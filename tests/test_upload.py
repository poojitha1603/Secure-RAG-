"""Uploaded documents get the same protection as seeded ones."""
import sys
import types

import numpy as np
import pytest

from app import embed, seed
from tests.conftest import auth

PHRASE = "zebra moonlight protocol quartz"       # unique words so we can find the upload
TEXT = (f"Project Zebra memo.\n\nThe {PHRASE} is a confidential internal plan. "
        f"Only senior staff may read about the {PHRASE}.")
QUERY = {"query": PHRASE, "top_k": 20}


def upload(client, who="acme-admin", roles="hr", users="", cls="confidential",
           name="zebra.txt", content=TEXT.encode(), **extra):
    return client.post("/admin/documents", headers=auth(who),
                       files={"file": (name, content)},
                       data={"classification": cls, "allowed_roles": roles,
                             "allowed_users": users, "title": "Zebra memo", **extra})


def docs_seen(client, who):
    r = client.post("/query", headers=auth(who), json=QUERY)
    return {s["doc_id"] for s in r.json()["sources"]}


# ---- who may upload / list / delete ---------------------------------------------------
def test_only_admins_can_manage_documents(client):
    for who in ("acme-employee", "acme-manager", "acme-hr"):
        assert upload(client, who).status_code == 403
        assert client.get("/admin/documents", headers=auth(who)).status_code == 403
        assert client.delete("/admin/documents/acme-hr-salary-sheet",
                             headers=auth(who)).status_code == 403
    assert client.post("/admin/documents", files={"file": ("a.txt", b"x")},
                       data={"classification": "internal", "allowed_roles": "hr"}).status_code == 401


# ---- uploaded docs obey their ACL -----------------------------------------------------
def test_uploaded_confidential_doc_is_visible_only_to_allowed_roles(client):
    r = upload(client, roles="hr")
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]
    assert doc_id.startswith("acme-") and r.json()["chunks"] >= 1

    assert doc_id in docs_seen(client, "acme-hr")
    for who in ("acme-employee", "acme-manager", "acme-admin"):
        assert doc_id not in docs_seen(client, who)
    for t in ("globex", "initech"):                     # other companies never see it
        for role in ("employee", "manager", "hr", "admin"):
            assert doc_id not in docs_seen(client, f"{t}-{role}")


def test_upload_to_single_user_via_allowed_users(client):
    doc_id = upload(client, roles="", users="acme-employee").json()["doc_id"]
    assert doc_id in docs_seen(client, "acme-employee")
    assert doc_id not in docs_seen(client, "acme-manager")


def test_tenant_comes_from_token_not_form(client):
    r = upload(client, who="acme-admin", roles="hr,employee,manager,admin",
               tenant_id="globex")                        # forged extra form field
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]
    assert doc_id.startswith("acme-")
    for role in ("employee", "manager", "hr", "admin"):
        assert doc_id not in docs_seen(client, f"globex-{role}")
    assert all(d["doc_id"] != doc_id for d in
               client.get("/admin/documents", headers=auth("globex-admin")).json()["documents"])


# ---- list / delete --------------------------------------------------------------------
def test_list_shows_only_own_tenant(client):
    doc_id = upload(client).json()["doc_id"]
    acme = client.get("/admin/documents", headers=auth("acme-admin")).json()["documents"]
    assert doc_id in {d["doc_id"] for d in acme}
    assert all(d["doc_id"].startswith("acme-") for d in acme)
    assert {d["doc_id"] for d in acme} >= {d.doc_id for d in seed.DOCUMENTS if d.tenant_id == "acme"}


def test_delete_removes_document_on_next_query(client):
    doc_id = upload(client).json()["doc_id"]
    assert doc_id in docs_seen(client, "acme-hr")
    r = client.delete(f"/admin/documents/{doc_id}", headers=auth("acme-admin"))
    assert r.status_code == 200 and r.json()["chunks_deleted"] >= 1
    assert doc_id not in docs_seen(client, "acme-hr")


def test_cannot_delete_other_tenants_document(client):
    r = client.delete("/admin/documents/acme-hr-salary-sheet", headers=auth("globex-admin"))
    assert r.status_code == 404
    assert "acme-hr-salary-sheet" in {
        d["doc_id"] for d in client.get("/admin/documents", headers=auth("acme-admin")).json()["documents"]}


# ---- input validation -----------------------------------------------------------------
def test_validation(client):
    assert upload(client, cls="top-secret").status_code == 422
    assert upload(client, roles="", users="").status_code == 422
    assert upload(client, name="x.exe", content=b"MZ").status_code == 415
    assert upload(client, content=b"   \n  ").status_code == 422
    assert upload(client, content=b"a" * 2_000_001).status_code == 413


def make_pdf(text: str) -> bytes:
    """Build a tiny valid one-page PDF (with a correct xref table) containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref_at}\n%%EOF").encode()
    return bytes(out)


def test_pdf_text_is_extracted(client):
    r = upload(client, name="memo.pdf", content=make_pdf(f"{PHRASE} budget"))
    assert r.status_code == 201, r.text
    assert r.json()["doc_id"] in docs_seen(client, "acme-hr")


# ---- embedder switch ------------------------------------------------------------------
def test_embedder_switch_uses_fastembed_when_asked(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for _ in texts:
                yield np.ones(384, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setenv("EMBEDDER", "fastembed")
    embed._reset_for_tests()
    try:
        assert len(embed.embed("hello")) == 384 and embed.dim() == 384
    finally:
        embed._reset_for_tests()
    monkeypatch.delenv("EMBEDDER")
    assert embed.dim() == 512 and len(embed.embed("hello")) == 512