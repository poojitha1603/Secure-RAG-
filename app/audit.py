"""Append-only audit log.

SQLite keeps the project zero-setup; the schema and queries port directly to
Postgres (swap json_each for jsonb_array_elements_text and use a
REVOKE UPDATE, DELETE ... FROM app_role instead of triggers).
"""
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    ts          REAL NOT NULL,          -- unix seconds, UTC
    user_id     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    query       TEXT NOT NULL,
    chunk_ids   TEXT NOT NULL,          -- JSON array returned to the LLM
    doc_ids     TEXT NOT NULL,          -- JSON array
    dropped_ids TEXT NOT NULL,          -- JSON array removed by the 2nd check (should be [])
    model       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_ts ON audit_log(tenant_id, ts);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

-- What admins DID (uploads, deletes, permission and user changes). Never stores passwords
-- or document text.
CREATE TABLE IF NOT EXISTS admin_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    tenant_id TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,            -- e.g. document.upload, user.disable
    target    TEXT NOT NULL,            -- doc id or user id
    detail    TEXT NOT NULL             -- JSON, small and non-sensitive
);
CREATE INDEX IF NOT EXISTS idx_events_tenant_ts ON admin_events(tenant_id, ts);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON admin_events
BEGIN SELECT RAISE(ABORT, 'admin_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON admin_events
BEGIN SELECT RAISE(ABORT, 'admin_events is append-only'); END;
"""


class AuditLog:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)

    def record(self, *, request_id: str, user, query: str, hits, dropped_ids, model: str):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO audit_log (request_id, ts, user_id, tenant_id, query,"
                " chunk_ids, doc_ids, dropped_ids, model) VALUES (?,?,?,?,?,?,?,?,?)",
                (request_id, time.time(), user.id, user.tenant_id, query,
                 json.dumps([h.chunk_id for h in hits]),
                 json.dumps(sorted({h.doc_id for h in hits})),
                 json.dumps(list(dropped_ids)), model))

    def who_retrieved(self, doc_id: str, tenant_id: str, days: int = 30) -> list[dict]:
        """Who retrieved this document in the last N days (scoped to one tenant)."""
        since = time.time() - days * 86400
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT a.request_id, a.ts, a.user_id, a.query "
                "FROM audit_log a, json_each(a.doc_ids) j "
                "WHERE j.value = ? AND a.tenant_id = ? AND a.ts >= ? ORDER BY a.ts DESC",
                (doc_id, tenant_id, since)).fetchall()
        return [{"request_id": r[0], "ts": r[1], "user_id": r[2], "query": r[3]} for r in rows]

    def recent(self, tenant_id: str, limit: int = 50) -> list[dict]:
        """Latest retrievals in one tenant, newest first."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT request_id, ts, user_id, query, doc_ids, dropped_ids FROM audit_log "
                "WHERE tenant_id = ? ORDER BY id DESC LIMIT ?", (tenant_id, limit)).fetchall()
        return [{"request_id": r[0], "ts": r[1], "user_id": r[2], "query": r[3],
                 "doc_ids": json.loads(r[4]), "dropped_count": len(json.loads(r[5]))}
                for r in rows]

    def record_admin(self, *, tenant_id: str, actor: str, action: str, target: str,
                     detail: dict | None = None):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO admin_events (ts, tenant_id, actor, action, target, detail)"
                " VALUES (?,?,?,?,?,?)",
                (time.time(), tenant_id, actor, action, target, json.dumps(detail or {})))

    def recent_admin(self, tenant_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, actor, action, target, detail FROM admin_events "
                "WHERE tenant_id = ? ORDER BY id DESC LIMIT ?", (tenant_id, limit)).fetchall()
        return [{"ts": r[0], "actor": r[1], "action": r[2], "target": r[3],
                 "detail": json.loads(r[4])} for r in rows]

    def rows(self) -> list[tuple]:
        with self._lock:
            return self.conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()


def new_request_id() -> str:
    return uuid.uuid4().hex