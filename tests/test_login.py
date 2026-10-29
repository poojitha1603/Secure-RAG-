"""Real login + user management."""
import jwt
import pytest

from app import seed
from app.auth import issue_token
from app.main import Settings, create_app
from app.models import User
from fastapi.testclient import TestClient
from tests.conftest import SECRET, SEED_PASSWORD, auth

SALARY_Q = {"query": "What are the salaries?", "top_k": 20}


def login(client, username, password=SEED_PASSWORD):
    return client.post("/login", json={"username": username, "password": password})


def bearer(resp):
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def docs_seen(client, headers, q=SALARY_Q):
    return {s["doc_id"] for s in client.post("/query", headers=headers, json=q).json()["sources"]}


def create_user(client, who="acme-admin", **body):
    payload = {"username": "bob", "password": "bobs-secret-1", "roles": ["employee"]}
    payload.update(body)
    return client.post("/admin/users", headers=auth(who), json=payload)


# ---- /login ---------------------------------------------------------------------------
def test_login_success_returns_working_token(client):
    r = login(client, "acme-hr")
    assert r.status_code == 200
    claims = jwt.decode(r.json()["access_token"], SECRET, algorithms=["HS256"])
    assert claims["sub"] == "acme-hr" and claims["tenant_id"] == "acme"
    assert "acme-hr-salary-sheet" in docs_seen(client, bearer(r))


def test_wrong_password_and_unknown_user_look_identical(client):
    bad_pw = login(client, "acme-hr", "wrong-password")
    ghost = login(client, "nobody-here", "whatever-123")
    assert bad_pw.status_code == ghost.status_code == 401
    assert bad_pw.json() == ghost.json()            # no username enumeration


def test_passwords_are_hashed_not_stored(client):
    rows = client.app.state.users.conn.execute("SELECT password_hash FROM users").fetchall()
    assert rows
    for (h,) in rows:
        assert h.startswith("scrypt$") and SEED_PASSWORD not in h


# ---- create / list users --------------------------------------------------------------
def test_only_admins_can_manage_users(client):
    for who in ("acme-employee", "acme-manager", "acme-hr"):
        assert create_user(client, who).status_code == 403
        assert client.get("/admin/users", headers=auth(who)).status_code == 403
    assert client.post("/admin/users", json={}).status_code == 401


def test_new_user_joins_admins_tenant_and_can_login(client):
    r = create_user(client, roles=["hr"], tenant_id="globex")      # forged tenant is ignored
    assert r.status_code == 201
    assert r.json()["user_id"] == "acme-bob" and r.json()["tenant_id"] == "acme"

    ok = login(client, "acme-bob", "bobs-secret-1")
    assert ok.status_code == 200
    assert "acme-hr-salary-sheet" in docs_seen(client, bearer(ok))      # role hr -> allowed
    assert not any(d.startswith("globex-") for d in docs_seen(client, bearer(ok)))


def test_user_ids_are_namespaced_by_tenant(client):
    r = create_user(client, who="globex-admin", username="acme-hr")   # tries to imitate acme-hr
    assert r.status_code == 201 and r.json()["user_id"] == "globex-acme-hr"
    assert login(client, "acme-hr").status_code == 200                # the real one is untouched


def test_user_validation(client):
    assert create_user(client, password="short").status_code == 422
    assert create_user(client, username="Bad Name!").status_code == 422
    assert create_user(client, roles=[]).status_code == 422
    assert create_user(client, roles=["HR; DROP"]).status_code == 422
    assert create_user(client).status_code == 201
    assert create_user(client).status_code == 409                     # duplicate


def test_list_users_only_own_tenant(client):
    users = client.get("/admin/users", headers=auth("acme-admin")).json()["users"]
    assert users and all(u["tenant_id"] == "acme" for u in users)
    assert all("password_hash" not in u for u in users)               # never exposed


# ---- disable / role change take effect immediately ------------------------------------
def test_disabling_a_user_kills_their_existing_token(client):
    h = bearer(login(client, "acme-hr"))
    assert client.post("/query", headers=h, json=SALARY_Q).status_code == 200

    r = client.patch("/admin/users/acme-hr", headers=auth("acme-admin"), json={"disabled": True})
    assert r.status_code == 200 and r.json()["disabled"] is True
    assert client.post("/query", headers=h, json=SALARY_Q).status_code == 401   # old token dead
    assert login(client, "acme-hr").status_code == 401                          # can't log in

    client.patch("/admin/users/acme-hr", headers=auth("acme-admin"), json={"disabled": False})
    assert login(client, "acme-hr").status_code == 200


def test_role_change_applies_to_tokens_already_issued(client):
    h = bearer(login(client, "acme-hr"))          # token minted while the user was HR
    assert "acme-hr-salary-sheet" in docs_seen(client, h)
    client.patch("/admin/users/acme-hr", headers=auth("acme-admin"), json={"roles": ["employee"]})
    assert "acme-hr-salary-sheet" not in docs_seen(client, h)     # DB roles win over token roles


def test_password_change(client):
    client.patch("/admin/users/acme-hr", headers=auth("acme-admin"),
                 json={"password": "a-brand-new-pass"})
    assert login(client, "acme-hr").status_code == 401
    assert login(client, "acme-hr", "a-brand-new-pass").status_code == 200


def test_cannot_manage_other_tenants_users(client):
    r = client.patch("/admin/users/globex-hr", headers=auth("acme-admin"), json={"disabled": True})
    assert r.status_code == 404
    assert login(client, "globex-hr").status_code == 200


def test_admin_cannot_lock_themselves_out(client):
    me = auth("acme-admin")
    assert client.patch("/admin/users/acme-admin", headers=me, json={"disabled": True}).status_code == 400
    assert client.patch("/admin/users/acme-admin", headers=me,
                        json={"roles": ["employee"]}).status_code == 400


# ---- a validly signed token is not enough ---------------------------------------------
def test_valid_signature_for_nonexistent_user_is_rejected(client):
    ghost = issue_token(User("acme-ghost", "acme", ("hr",)), SECRET)
    r = client.post("/query", headers={"Authorization": f"Bearer {ghost}"}, json=SALARY_Q)
    assert r.status_code == 401


# ---- dev token switch -----------------------------------------------------------------
def test_dev_token_endpoint_can_be_turned_off():
    app = create_app(Settings(qdrant_url=None, audit_db=":memory:", users_db=":memory:",
                              jwt_secret=SECRET, seed=True, enable_dev_tokens=False,
                              seed_password=SEED_PASSWORD))
    c = TestClient(app)
    assert c.post("/token", json={"user_id": "acme-hr"}).status_code == 404
    assert c.get("/health").json()["dev_tokens"] is False
    assert c.post("/login", json={"username": "acme-hr", "password": SEED_PASSWORD}).status_code == 200