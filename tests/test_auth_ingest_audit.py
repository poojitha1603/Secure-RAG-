import sqlite3
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app import seed
from app.auth import issue_token
from app.ingest import make_chunks
from app.store import UnprotectedChunkError
from tests.conftest import SECRET, auth

Q = {"query": "company overview"}


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


# ---- auth: identity comes from a verified token only ---------------------------------
def test_missing_and_garbage_token(client):
    assert client.post("/query", json=Q).status_code == 401
    assert client.post("/query", json=Q, headers=bearer("not-a-jwt")).status_code == 401


def test_wrong_signature_rejected(client):
    forged = jwt.encode({"sub": "x", "tenant_id": "globex", "roles": ["hr"],
                         "exp": int(time.time()) + 600}, "attacker-secret-xxxxxxxxxxxxxxxxxxxxxx", "HS256")
    assert client.post("/query", json=Q, headers=bearer(forged)).status_code == 401


def test_alg_none_rejected(client):
    tok = jwt.encode({"sub": "x", "tenant_id": "globex", "roles": ["hr"],
                      "exp": int(time.time()) + 600}, None, algorithm="none")
    assert client.post("/query", json=Q, headers=bearer(tok)).status_code == 401


def test_expired_token_rejected(client):
    tok = issue_token(seed.USERS["acme-hr"], SECRET, ttl_seconds=-10)
    assert client.post("/query", json=Q, headers=bearer(tok)).status_code == 401


def test_token_without_tenant_rejected(client):
    tok = jwt.encode({"sub": "x", "roles": ["hr"], "exp": int(time.time()) + 600}, SECRET, "HS256")
    assert client.post("/query", json=Q, headers=bearer(tok)).status_code == 401


# ---- ingestion: no chunk without a tenant ---------------------------------------------
def test_ingest_rejects_chunk_without_tenant(app):
    store = app.state.store
    before = store.count()
    chunks = make_chunks(seed.DOCUMENTS[0])
    bad = make_chunks(seed.DOCUMENTS[1])
    bad[0]["payload"]["tenant_id"] = ""          # the unprotected chunk
    with pytest.raises(UnprotectedChunkError):
        store.upsert_chunks(chunks + bad)
    assert store.count() == before               # all-or-nothing: nothing was written


def test_every_stored_chunk_has_tenant(app):
    points, _ = app.state.store.client.scroll("chunks", limit=1000, with_payload=True)
    assert points and all(p.payload.get("tenant_id") for p in points)


# ---- audit log ------------------------------------------------------------------------
def test_audit_records_and_admin_can_query(client):
    client.post("/query", headers=auth("acme-hr"),
                json={"query": "salary sheet", "top_k": 20})
    client.post("/query", headers=auth("acme-employee"),
                json={"query": "salary sheet", "top_k": 20})

    r = client.get("/admin/documents/acme-hr-salary-sheet/accesses?days=30",
                   headers=auth("acme-admin"))
    assert r.status_code == 200
    users = {a["user_id"] for a in r.json()["accesses"]}
    assert users == {"acme-hr"}                  # the employee could NOT retrieve it

    # another tenant's admin sees nothing about acme's documents
    r2 = client.get("/admin/documents/acme-hr-salary-sheet/accesses",
                    headers=auth("globex-admin"))
    assert r2.json()["accesses"] == []


def test_admin_endpoints_need_admin_role(client):
    assert client.get("/admin/documents/acme-hr-salary-sheet/accesses",
                      headers=auth("acme-hr")).status_code == 403


def test_audit_log_is_append_only(client):
    client.post("/query", headers=auth("acme-hr"), json=Q)
    conn = client.app.state.audit.conn
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit_log SET user_id='someone-else'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM audit_log")
