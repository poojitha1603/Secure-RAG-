"""User accounts: salted scrypt password hashes, stored in SQLite.

(Postgres later: same columns; keep the hashing as is.)
Passwords are never stored or logged in plain text.
"""
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

from .models import User

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
MIN_PASSWORD = 8
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
ROLE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,29}$")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                        dklen=32, maxmem=64 * 1024 * 1024)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, dk_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r),
                            p=int(p), dklen=len(bytes.fromhex(dk_hex)), maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(dk, bytes.fromhex(dk_hex))
    except (ValueError, TypeError):
        return False


# A real-looking hash to verify against when the username doesn't exist, so a login
# for an unknown user takes about as long as one for a known user.
DUMMY_HASH = hash_password("not-a-real-password")


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    tenant_id: str
    roles: tuple[str, ...]
    password_hash: str
    disabled: bool

    @property
    def user(self) -> User:
        return User(self.user_id, self.tenant_id, self.roles)

    def public(self) -> dict:
        return {"user_id": self.user_id, "tenant_id": self.tenant_id,
                "roles": list(self.roles), "disabled": self.disabled}


class UserExists(Exception):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    roles         TEXT NOT NULL,        -- JSON array
    password_hash TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
"""


class UserStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)

    @staticmethod
    def _row(r) -> UserRecord:
        return UserRecord(r[0], r[1], tuple(json.loads(r[2])), r[3], bool(r[4]))

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def create(self, user_id: str, tenant_id: str, roles, password_hash: str):
        try:
            with self._lock, self.conn:
                self.conn.execute(
                    "INSERT INTO users (user_id, tenant_id, roles, password_hash, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (user_id, tenant_id, json.dumps(list(roles)), password_hash, time.time()))
        except sqlite3.IntegrityError as e:
            raise UserExists(user_id) from e

    def get(self, user_id: str) -> UserRecord | None:
        with self._lock:
            r = self.conn.execute(
                "SELECT user_id, tenant_id, roles, password_hash, disabled FROM users"
                " WHERE user_id = ?", (user_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, tenant_id: str) -> list[UserRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT user_id, tenant_id, roles, password_hash, disabled FROM users"
                " WHERE tenant_id = ? ORDER BY user_id", (tenant_id,)).fetchall()
        return [self._row(r) for r in rows]

    def list_active(self, tenant_id: str | None = None) -> "list[UserRecord]":
        """Users that can sign in right now (not disabled). One tenant, or all if None."""
        sql = ("SELECT user_id, tenant_id, roles, password_hash, disabled FROM users"
               " WHERE disabled = 0")
        args: tuple = ()
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            args = (tenant_id,)
        with self._lock:
            rows = self.conn.execute(sql + " ORDER BY tenant_id, user_id", args).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, user_id: str, tenant_id: str) -> bool:
        """Delete a user of THIS tenant. Returns False if no such user in the tenant."""
        with self._lock, self.conn:
            cur = self.conn.execute("DELETE FROM users WHERE user_id = ? AND tenant_id = ?",
                                    (user_id, tenant_id))
        return cur.rowcount == 1

    def update(self, user_id: str, tenant_id: str, *, roles=None, disabled=None,
               password_hash=None) -> bool:
        """Update a user of THIS tenant. Returns False if no such user in the tenant."""
        sets, vals = [], []
        if roles is not None:
            sets.append("roles = ?"); vals.append(json.dumps(list(roles)))
        if disabled is not None:
            sets.append("disabled = ?"); vals.append(1 if disabled else 0)
        if password_hash is not None:
            sets.append("password_hash = ?"); vals.append(password_hash)
        if not sets:
            return self.get(user_id) is not None and self.get(user_id).tenant_id == tenant_id
        with self._lock, self.conn:
            cur = self.conn.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE user_id = ? AND tenant_id = ?",
                (*vals, user_id, tenant_id))
        return cur.rowcount == 1