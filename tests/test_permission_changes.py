import time

from app import seed

from tests.conftest import auth

DOC = "acme-mgr-restructuring"
ASK = {"query": "restructuring plan next quarter", "top_k": 20}


def docs_seen(client, user_id):
    r = client.post("/query", headers=auth(user_id), json=ASK)
    return {s["doc_id"] for s in r.json()["sources"]}


def test_revoke_takes_effect_on_the_very_next_query(client):
    assert DOC in docs_seen(client, "acme-manager")                 # allowed at first

    t0 = time.perf_counter()
    r = client.put(f"/admin/documents/{DOC}/acl", headers=auth("acme-admin"),
                   json={"allowed_roles": ["hr"], "allowed_users": []})
    assert r.status_code == 200 and r.json()["chunks_updated"] >= 1
    still_visible = DOC in docs_seen(client, "acme-manager")        # very next query
    revoke_to_effect_ms = (time.perf_counter() - t0) * 1000

    assert not still_visible, "revoked document is still retrievable"
    assert DOC in docs_seen(client, "acme-hr")                      # hr keeps access
    print(f"revoke-to-effect: {revoke_to_effect_ms:.1f} ms")


def test_grant_to_single_user(client):
    assert DOC not in docs_seen(client, "acme-employee")
    client.put(f"/admin/documents/{DOC}/acl", headers=auth("acme-admin"),
               json={"allowed_roles": ["manager", "hr"], "allowed_users": ["acme-employee"]})
    assert DOC in docs_seen(client, "acme-employee")


def test_admin_cannot_touch_other_tenants_documents(client):
    r = client.put(f"/admin/documents/{DOC}/acl", headers=auth("globex-admin"),
                   json={"allowed_roles": ["employee"], "allowed_users": []})
    assert r.status_code == 404                                     # hides existence too
    assert DOC in docs_seen(client, "acme-manager")                 # unchanged


def test_non_admin_cannot_change_acl(client):
    r = client.put(f"/admin/documents/{DOC}/acl", headers=auth("acme-hr"),
                   json={"allowed_roles": ["employee"], "allowed_users": []})
    assert r.status_code == 403
