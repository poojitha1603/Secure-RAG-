"""Admin audit views: recent retrievals + what admins did. Tenant-scoped and tamper-proof."""
import sqlite3

import pytest

from tests.conftest import auth

Q = {"query": "What are the salaries?", "top_k": 20}


def test_audit_views_are_admin_only(client):
    for who in ("acme-employee", "acme-manager", "acme-hr"):
        assert client.get("/admin/audit", headers=auth(who)).status_code == 403
        assert client.get("/admin/events", headers=auth(who)).status_code == 403
    assert client.get("/admin/audit").status_code == 401


def test_recent_retrievals_are_tenant_scoped(client):
    client.post("/query", headers=auth("acme-hr"), json=Q)
    client.post("/query", headers=auth("globex-hr"), json=Q)

    acme = client.get("/admin/audit", headers=auth("acme-admin")).json()["events"]
    assert [e["user_id"] for e in acme] == ["acme-hr"]
    assert "acme-hr-salary-sheet" in acme[0]["doc_ids"]
    assert acme[0]["dropped_count"] == 0

    globex = client.get("/admin/audit", headers=auth("globex-admin")).json()["events"]
    assert [e["user_id"] for e in globex] == ["globex-hr"]
    assert all(d.startswith("globex-") for e in globex for d in e["doc_ids"])


def test_admin_actions_are_logged_without_secrets(client):
    admin = auth("acme-admin")
    up = client.post("/admin/documents", headers=admin,
                     files={"file": ("memo.txt", b"quarterly plan text")},
                     data={"classification": "internal", "allowed_roles": "hr", "title": "Memo"})
    doc_id = up.json()["doc_id"]
    client.put(f"/admin/documents/{doc_id}/acl", headers=admin,
               json={"allowed_roles": ["hr", "manager"], "allowed_users": []})
    client.post("/admin/users", headers=admin,
                json={"username": "carol", "password": "carols-secret-9", "roles": ["employee"]})
    client.patch("/admin/users/acme-carol", headers=admin,
                 json={"password": "carols-new-secret", "disabled": True})
    client.delete(f"/admin/documents/{doc_id}", headers=admin)

    events = client.get("/admin/events", headers=admin).json()["events"]
    actions = [e["action"] for e in events]          # newest first
    assert actions == ["document.delete", "user.update", "user.create",
                       "document.acl_change", "document.upload"]
    assert all(e["actor"] == "acme-admin" for e in events)

    blob = str(events)
    assert "carols-secret-9" not in blob and "carols-new-secret" not in blob
    assert "quarterly plan text" not in blob          # document text is never logged
    upd = next(e for e in events if e["action"] == "user.update")
    assert upd["detail"] == {"disabled": True, "password_changed": True}


def test_admin_events_are_tenant_scoped(client):
    client.post("/admin/users", headers=auth("acme-admin"),
                json={"username": "dave", "password": "daves-secret-1", "roles": ["employee"]})
    assert client.get("/admin/events", headers=auth("globex-admin")).json()["events"] == []


def test_admin_events_are_append_only(client):
    client.post("/admin/users", headers=auth("acme-admin"),
                json={"username": "erin", "password": "erins-secret-1", "roles": ["employee"]})
    conn = client.app.state.audit.conn
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE admin_events SET actor = 'someone-else'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM admin_events")


def test_failed_admin_actions_are_not_logged_as_success(client):
    client.delete("/admin/documents/globex-hr-salary-sheet", headers=auth("acme-admin"))   # 404
    assert client.get("/admin/events", headers=auth("acme-admin")).json()["events"] == []