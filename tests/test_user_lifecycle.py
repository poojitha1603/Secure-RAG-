"""Create / disable / delete users and the dev quick sign-in dropdown's data source.

The dropdown is filled from GET /dev/users. These tests check that this list reflects
active users, never leaks other companies to a signed-in user, and that disabled or
deleted users really lose access (including tokens issued earlier).
"""
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app import seed
from app.auth import issue_token
from app.main import Settings, create_app
from app.models import User
from tests.conftest import SECRET, SEED_PASSWORD, auth

Q = {"query": "company overview", "top_k": 5}


def create_user(client, username="bob", roles=("employee",), who="acme-admin"):
    r = client.post("/admin/users", headers=auth(who),
                    json={"username": username, "password": "bobs-secret-12", "roles": list(roles)})
    assert r.status_code == 201, r.text
    return r.json()["user_id"]


def dev_users(client, headers=None):
    return client.get("/dev/users", headers=headers or {})


def ids(resp):
    return {u["user_id"] for u in resp.json()["users"]}


def dev_token(client, user_id):
    r = client.post("/token", json={"user_id": user_id})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def patch_user(client, user_id, who="acme-admin", **body):
    return client.patch(f"/admin/users/{user_id}", headers=auth(who), json=body)


# ---- the dropdown's data source ---------------------------------------------------------
def test_dev_users_is_404_when_dev_tokens_are_off():
    app = create_app(Settings(qdrant_url=None, audit_db=":memory:", users_db=":memory:",
                              jwt_secret=SECRET, seed=True, enable_dev_tokens=False,
                              seed_password=SEED_PASSWORD))
    c = TestClient(app)
    assert c.get("/dev/users").status_code == 404
    assert c.get("/dev/users", headers=auth("acme-hr")).status_code == 404


def test_signed_out_list_has_every_active_user_and_nothing_sensitive(client):
    r = dev_users(client)
    assert r.status_code == 200 and r.json()["scope"] == "all"
    assert ids(r) == set(seed.USERS)                         # comes from the database
    for u in r.json()["users"]:
        assert set(u) == {"user_id", "tenant_id", "roles"}   # no password hash, no flags


def test_created_user_appears_in_the_list(client):
    assert "acme-bob" not in ids(dev_users(client))
    uid = create_user(client)
    assert uid == "acme-bob"
    assert uid in ids(dev_users(client))                                       # signed out
    assert uid in ids(dev_users(client, auth("acme-admin")))                   # signed in, same company
    assert uid not in ids(dev_users(client, auth("globex-admin")))             # other company can't see it
    bob = next(u for u in dev_users(client).json()["users"] if u["user_id"] == uid)
    assert bob["tenant_id"] == "acme" and bob["roles"] == ["employee"]


def test_disabled_user_disappears_and_returns_when_enabled(client):
    uid = create_user(client)
    assert uid in ids(dev_users(client))
    patch_user(client, uid, disabled=True)
    assert uid not in ids(dev_users(client))
    assert uid not in ids(dev_users(client, auth("acme-admin")))
    patch_user(client, uid, disabled=False)
    assert uid in ids(dev_users(client))


def test_deleted_user_disappears_and_loses_all_access(client):
    uid = create_user(client)
    old = dev_token(client, uid)                      # a token issued while the user existed
    assert client.post("/query", headers=old, json=Q).status_code == 200

    r = client.delete(f"/admin/users/{uid}", headers=auth("acme-admin"))
    assert r.status_code == 200

    assert uid not in ids(dev_users(client))                                   # dropdown data
    assert uid not in {u["user_id"] for u in client.get("/admin/users", headers=auth("acme-admin")).json()["users"]}
    assert client.post("/token", json={"user_id": uid}).status_code == 404     # can't mint a new token
    assert client.post("/login", json={"username": uid, "password": "bobs-secret-12"}).status_code == 401
    assert client.post("/query", headers=old, json=Q).status_code == 401       # old token is dead
    assert dev_users(client, old).status_code == 401


def test_disabled_users_old_token_and_new_logins_are_rejected(client):
    uid = create_user(client)
    old = dev_token(client, uid)
    patch_user(client, uid, disabled=True)
    assert client.post("/query", headers=old, json=Q).status_code == 401
    assert dev_users(client, old).status_code == 401
    assert client.post("/token", json={"user_id": uid}).status_code == 404
    assert client.post("/login", json={"username": uid, "password": "bobs-secret-12"}).status_code == 401


def test_disabled_admin_loses_admin_powers_immediately(client):
    uid = create_user(client, "ops", roles=("employee", "admin"))
    old = dev_token(client, uid)
    assert client.get("/admin/users", headers=old).status_code == 200
    patch_user(client, uid, disabled=True)
    assert client.get("/admin/users", headers=old).status_code == 401


# ---- tenant isolation for the list ------------------------------------------------------
def test_signed_in_user_sees_only_their_own_company(client):
    for who in ("acme-employee", "acme-hr", "globex-employee"):
        r = dev_users(client, auth(who))
        tenant = seed.USERS[who].tenant_id
        assert r.status_code == 200 and r.json()["scope"] == "tenant"
        assert {u["tenant_id"] for u in r.json()["users"]} == {tenant}
        assert all(i.startswith(tenant + "-") for i in ids(r))


def test_a_bad_token_never_falls_back_to_the_wider_list(client):
    assert dev_users(client, {"Authorization": "Bearer not-a-jwt"}).status_code == 401
    forged = jwt.encode({"sub": "acme-hr", "tenant_id": "acme", "roles": ["hr"],
                         "exp": int(time.time()) + 600}, "attacker-secret-xxxxxxxxxxxxxxxxxxxxxxxx", "HS256")
    assert dev_users(client, {"Authorization": f"Bearer {forged}"}).status_code == 401
    expired = issue_token(seed.USERS["acme-hr"], SECRET, ttl_seconds=-10)
    assert dev_users(client, {"Authorization": f"Bearer {expired}"}).status_code == 401
    ghost = issue_token(User("acme-ghost", "acme", ("hr",)), SECRET)           # signed, but no such user
    assert dev_users(client, {"Authorization": f"Bearer {ghost}"}).status_code == 401


def test_the_dropdown_cannot_choose_a_tenant_or_grant_roles(client):
    """The dropdown only sends a user id. Extra fields are ignored; the token carries
    the tenant and roles stored in the database."""
    r = client.post("/token", json={"user_id": "acme-employee", "tenant_id": "globex",
                                    "roles": ["admin", "hr"]})
    claims = jwt.decode(r.json()["access_token"], SECRET, algorithms=["HS256"])
    assert claims["tenant_id"] == "acme" and claims["roles"] == ["employee"]
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert client.get("/admin/users", headers=h).status_code == 403            # still not an admin
    docs = client.post("/query", headers=h, json={"query": "salaries", "top_k": 20}).json()["sources"]
    assert docs and all(s["doc_id"].startswith("acme-") for s in docs)
    assert not any(s["doc_id"] == "acme-hr-salary-sheet" for s in docs)


# ---- DELETE /admin/users rules ----------------------------------------------------------
def test_delete_user_rules(client):
    uid = create_user(client)
    assert client.delete(f"/admin/users/{uid}").status_code == 401
    assert client.delete(f"/admin/users/{uid}", headers=auth("acme-hr")).status_code == 403
    assert client.delete(f"/admin/users/{uid}", headers=auth("globex-admin")).status_code == 404
    assert client.delete("/admin/users/acme-admin", headers=auth("acme-admin")).status_code == 400
    assert client.delete("/admin/users/acme-nobody", headers=auth("acme-admin")).status_code == 404
    assert uid in ids(dev_users(client))                                       # nothing happened yet
    assert client.delete(f"/admin/users/{uid}", headers=auth("acme-admin")).status_code == 200
    assert client.delete(f"/admin/users/{uid}", headers=auth("acme-admin")).status_code == 404


def test_delete_is_logged_without_secrets(client):
    uid = create_user(client)
    client.delete(f"/admin/users/{uid}", headers=auth("acme-admin"))
    events = client.get("/admin/events", headers=auth("acme-admin")).json()["events"]
    assert events[0]["action"] == "user.delete" and events[0]["target"] == uid
    assert "bobs-secret-12" not in str(events)


def test_deleting_a_user_removes_their_document_grants(client):
    """A new user who later gets a deleted user's id must NOT inherit that user's access."""
    up = client.post("/admin/documents", headers=auth("acme-admin"),
                     files={"file": ("n.txt", b"Private notes for bob only: tangerine-orbit plan.")},
                     data={"classification": "confidential", "allowed_roles": "hr",
                           "allowed_users": "acme-bob", "title": "Bob notes"})
    doc = up.json()["doc_id"]
    ask = {"query": "tangerine-orbit plan", "top_k": 20}

    create_user(client)                                                        # acme-bob, role employee
    bob = dev_token(client, "acme-bob")
    assert doc in {s["doc_id"] for s in client.post("/query", headers=bob, json=ask).json()["sources"]}

    r = client.delete("/admin/users/acme-bob", headers=auth("acme-admin"))
    assert r.json()["document_grants_removed"] == 1
    listed = {d["doc_id"]: d for d in client.get("/admin/documents", headers=auth("acme-admin")).json()["documents"]}
    assert listed[doc]["allowed_users"] == [] and listed[doc]["allowed_roles"] == ["hr"]

    create_user(client)                                                        # a DIFFERENT person, same id
    new_bob = dev_token(client, "acme-bob")
    assert doc not in {s["doc_id"] for s in client.post("/query", headers=new_bob, json=ask).json()["sources"]}
    assert doc in {s["doc_id"] for s in client.post("/query", headers=auth("acme-hr"), json=ask).json()["sources"]}


# ---- the page itself --------------------------------------------------------------------
def test_page_loads_users_from_the_server_not_from_a_hardcoded_list(client):
    html = client.get("/").text
    assert "/dev/users" in html
    assert "const TENANTS" not in html and "const ROLES" not in html
    assert "Object.entries(TENANTS)" not in html