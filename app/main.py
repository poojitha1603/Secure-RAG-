"""FastAPI service. Run: uvicorn app.main:app --reload"""
import os
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from . import rag, seed
from .acl import acl_filter, is_allowed
from .audit import AuditLog, new_request_id
from .auth import AuthError, issue_token, verify_token
from .extract import UnsupportedFile, extract_text
from .ingest import ingest_all, ingest_document
from .models import CLASSIFICATIONS, Document
from .store import VectorStore
from .throttle import LoginThrottle
from .users import (MIN_PASSWORD, ROLE_RE, USERNAME_RE, DUMMY_HASH, UserExists, UserStore,
                    hash_password, verify_password)

load_dotenv()   # reads a .env file in the folder you run from (real env vars win)
MAX_UPLOAD_BYTES = 2_000_000


@dataclass
class Settings:
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL"))
    audit_db: str = field(default_factory=lambda: os.getenv("AUDIT_DB", "audit.db"))
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "dev-only-change-me"))
    seed: bool = field(default_factory=lambda: os.getenv("SEED", "1") == "1")
    enable_dev_tokens: bool = field(default_factory=lambda: os.getenv("DEV_TOKENS", "1") == "1")
    users_db: str = field(default_factory=lambda: os.getenv("USERS_DB", "users.db"))
    # Password given to the seeded demo accounts (only when SEED=1). Change it in .env.
    seed_password: str = field(default_factory=lambda: os.getenv("SEED_PASSWORD", "demo-password"))
    login_max_per_ip: int = field(default_factory=lambda: int(os.getenv("LOGIN_MAX_PER_IP", "0")))  # 0 = auto
    login_max_failures: int = field(default_factory=lambda: int(os.getenv("LOGIN_MAX_FAILURES", "5")))
    login_window_seconds: float = field(default_factory=lambda: float(os.getenv("LOGIN_WINDOW_SECONDS", "300")))


@lru_cache(maxsize=4)
def _seed_hash(password: str) -> str:
    return hash_password(password)       # hashed once; reused for every seeded demo user


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class CreateUserRequest(BaseModel):
    # No tenant field: the new user always joins the ADMIN'S tenant (extras are ignored).
    model_config = ConfigDict(extra="ignore")
    username: str
    password: str
    roles: list[str]


class UpdateUserRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    disabled: bool | None = None
    roles: list[str] | None = None
    password: str | None = None


def _clean_roles(roles: list[str]) -> list[str]:
    out = sorted({r.strip().lower() for r in roles})
    if not out or len(out) > 10 or not all(ROLE_RE.match(r) for r in out):
        raise HTTPException(422, "roles: 1-10 names like 'employee', 'hr' (a-z, 0-9, _ -)")
    return out


def _check_password(pw: str):
    if len(pw) < MIN_PASSWORD or len(pw) > 200:
        raise HTTPException(422, f"password must be {MIN_PASSWORD}-200 characters")


class QueryRequest(BaseModel):
    # extra="ignore": a tenant_id / roles / user_id in the body is silently dropped.
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class AclUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_roles: list[str]
    allowed_users: list[str] = []


class DevTokenRequest(BaseModel):
    user_id: str


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="Permission-aware multi-tenant RAG")
    app.state.settings = s
    app.state.store = VectorStore(s.qdrant_url)
    app.state.audit = AuditLog(s.audit_db)
    if not s.enable_dev_tokens and (s.jwt_secret == "dev-only-change-me" or len(s.jwt_secret) < 32):
        # Production mode (DEV_TOKENS=0) with a guessable signing key would let anyone forge admin tokens.
        raise RuntimeError("JWT_SECRET must be set to a random value of at least 32 characters "
                           "when DEV_TOKENS=0. Generate one: python -c \"import secrets;print(secrets.token_urlsafe(48))\"")
    app.state.users = UserStore(s.users_db)
    app.state.throttle = LoginThrottle(s.login_max_failures,
                                       s.login_max_per_ip or max(30, 6 * s.login_max_failures),
                                       s.login_window_seconds)
    app.state.build_filter = acl_filter           # tests can swap this to simulate a bug
    if s.seed and app.state.store.count() == 0:
        ingest_all(app.state.store, seed.DOCUMENTS)
    if s.seed and app.state.users.count() == 0:
        pw_hash = _seed_hash(s.seed_password)
        for u in seed.USERS.values():
            app.state.users.create(u.id, u.tenant_id, u.roles, pw_hash)

    bearer = HTTPBearer(auto_error=False)

    def resolve_user(token: str):
        try:
            claims_user = verify_token(token, s.jwt_secret)
        except AuthError:
            raise HTTPException(401, "invalid token")
        # The signed token proves WHO you are; the database says what you are allowed
        # to be right now. Deleted/disabled users and role changes take effect at once.
        rec = app.state.users.get(claims_user.id)
        if rec is None or rec.disabled:
            raise HTTPException(401, "invalid token")
        return rec.user

    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if creds is None:
            raise HTTPException(401, "missing bearer token")
        return resolve_user(creds.credentials)

    def admin_user(user=Depends(current_user)):
        if "admin" not in user.roles:
            raise HTTPException(403, "admin role required")
        return user

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"ok": True, "chunks": app.state.store.count(),
                "dev_tokens": s.enable_dev_tokens}

    @app.post("/login")
    def login(body: LoginRequest, request: Request):
        """Username + password -> signed JWT. Same error for unknown user, wrong password
        and disabled account, so the response never reveals which usernames exist.
        Too many recent failures for the username or the client IP -> 429 (checked first)."""
        name = body.username.strip().lower()
        ip = request.client.host if request.client else "unknown"
        wait = app.state.throttle.retry_after(name, ip)
        if wait:
            raise HTTPException(429, f"too many failed logins, try again in {wait} seconds",
                                headers={"Retry-After": str(wait)})
        rec = app.state.users.get(body.username)
        ok = verify_password(body.password, rec.password_hash if rec else DUMMY_HASH)
        if rec is None or rec.disabled or not ok:
            app.state.throttle.failure(name, ip)
            raise HTTPException(401, "invalid username or password")
        app.state.throttle.success(name)
        return {"access_token": issue_token(rec.user, s.jwt_secret), "token_type": "bearer"}

    @app.get("/dev/users")
    def dev_users(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """DEV ONLY: active users for the quick sign-in dropdown (404 unless DEV_TOKENS=1).

        Signed out: all companies' active users, so a developer can pick any identity.
        Signed in: only the caller's own company. A token that is present but invalid,
        expired, or belongs to a disabled/deleted user gets 401 (never the wider list).
        Disabled users are never listed. It only lists; signing in still goes through
        /token, which takes the tenant and roles from the database, never from the client."""
        if not s.enable_dev_tokens:
            raise HTTPException(404)
        tenant = resolve_user(creds.credentials).tenant_id if creds is not None else None
        return {"scope": "tenant" if tenant else "all",
                "users": [{"user_id": r.user_id, "tenant_id": r.tenant_id, "roles": list(r.roles)}
                          for r in app.state.users.list_active(tenant)]}

    @app.post("/token")
    def dev_token(body: DevTokenRequest):
        """DEV ONLY (no password!): mint a token for an existing user.
        Turn it off with DEV_TOKENS=0 before deploying."""
        if not s.enable_dev_tokens:
            raise HTTPException(404)
        rec = app.state.users.get(body.user_id)
        if rec is None or rec.disabled:
            raise HTTPException(404, "unknown user")
        return {"access_token": issue_token(rec.user, s.jwt_secret), "token_type": "bearer"}

    @app.post("/query")
    def query(body: QueryRequest, user=Depends(current_user)):
        hits, dropped = rag.retrieve(app.state.store, user, body.query, body.top_k,
                                     build_filter=app.state.build_filter)
        answer = rag.generate(body.query, hits)
        request_id = new_request_id()
        app.state.audit.record(request_id=request_id, user=user, query=body.query,
                               hits=hits, dropped_ids=dropped, model=rag.model_name())
        return {"request_id": request_id, "answer": answer,
                "sources": [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title,
                             "classification": h.classification, "score": round(h.score, 4)}
                            for h in hits]}

    @app.get("/admin/documents/{doc_id}/accesses")
    def accesses(doc_id: str, days: int = Query(30, ge=1, le=365), admin=Depends(admin_user)):
        """Who retrieved document X in the last N days (own tenant only)."""
        return {"doc_id": doc_id, "days": days,
                "accesses": app.state.audit.who_retrieved(doc_id, admin.tenant_id, days)}

    @app.put("/admin/documents/{doc_id}/acl")
    def change_acl(doc_id: str, body: AclUpdate, admin=Depends(admin_user)):
        try:
            roles = [r.strip().lower() for r in body.allowed_roles if r.strip()]
            users = [u.strip().lower() for u in body.allowed_users if u.strip()]
            n = app.state.store.update_acl(doc_id, admin.tenant_id, roles, users)
        except KeyError:
            raise HTTPException(404, "document not found")   # also hides other tenants' docs
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="document.acl_change", target=doc_id,
            detail={"allowed_roles": roles, "allowed_users": users})
        return {"doc_id": doc_id, "chunks_updated": n}

    @app.get("/admin/documents")
    def list_documents(admin=Depends(admin_user)):
        """Documents in the admin's own tenant."""
        return {"documents": app.state.store.list_documents(admin.tenant_id)}

    @app.post("/admin/documents", status_code=201)
    def upload_document(
        file: UploadFile = File(...),
        classification: str = Form(...),
        allowed_roles: str = Form("", description="comma-separated, e.g. hr,manager"),
        allowed_users: str = Form("", description="comma-separated user ids"),
        title: str = Form(""),
        admin=Depends(admin_user),
    ):
        """Upload a TXT/MD/PDF. The tenant is the ADMIN'S tenant from the token;
        there is no tenant field in this form, and any extra form fields are ignored."""
        if classification not in CLASSIFICATIONS:
            raise HTTPException(422, f"classification must be one of {CLASSIFICATIONS}")
        roles = tuple(r.strip().lower() for r in allowed_roles.split(",") if r.strip())
        users = tuple(u.strip().lower() for u in allowed_users.split(",") if u.strip())
        if not roles and not users:
            raise HTTPException(422, "give at least one allowed role or user")

        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
        try:
            text = extract_text(file.filename or "", data)
        except UnsupportedFile as e:
            raise HTTPException(415, str(e))
        if not text:
            raise HTTPException(422, "no text could be extracted from the file")

        doc = Document(
            doc_id=f"{admin.tenant_id}-up-{uuid.uuid4().hex[:10]}",
            tenant_id=admin.tenant_id,
            title=title.strip() or (file.filename or "Untitled"),
            classification=classification, allowed_roles=roles, allowed_users=users, text=text)
        chunks = ingest_document(app.state.store, doc)
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="document.upload", target=doc.doc_id,
            detail={"title": doc.title, "classification": classification,
                    "allowed_roles": list(roles), "allowed_users": list(users), "chunks": chunks})
        return {"doc_id": doc.doc_id, "title": doc.title, "chunks": chunks,
                "classification": classification, "allowed_roles": list(roles),
                "allowed_users": list(users)}

    @app.get("/admin/audit")
    def audit_recent(limit: int = Query(50, ge=1, le=200), admin=Depends(admin_user)):
        """Latest retrievals in the admin's own tenant."""
        return {"events": app.state.audit.recent(admin.tenant_id, limit)}

    @app.get("/admin/events")
    def admin_events(limit: int = Query(50, ge=1, le=200), admin=Depends(admin_user)):
        """Latest admin actions (uploads, deletes, permission and user changes)."""
        return {"events": app.state.audit.recent_admin(admin.tenant_id, limit)}

    @app.get("/admin/documents/{doc_id}/explain")
    def explain_access(doc_id: str, user_id: str = Query(...), admin=Depends(admin_user)):
        """Why can (or can't) this user see this document? Reads the STORED permissions
        from the running server and applies the same rule retrieval uses."""
        rec = app.state.users.get(user_id)
        if rec is None or rec.tenant_id != admin.tenant_id:
            raise HTTPException(404, "user not found")           # also hides other tenants' users
        doc = next((d for d in app.state.store.list_documents(admin.tenant_id)
                    if d["doc_id"] == doc_id), None)
        if doc is None:
            raise HTTPException(404, "document not found")
        u = rec.user
        via_role = sorted(set(u.roles) & set(doc["allowed_roles"]))
        via_user = u.id in doc["allowed_users"]
        allowed = is_allowed(u, {"tenant_id": admin.tenant_id,
                                 "allowed_roles": doc["allowed_roles"],
                                 "allowed_users": doc["allowed_users"]})
        if via_role and via_user:
            reason = f"allowed by role ({', '.join(via_role)}) and by an explicit user grant"
        elif via_role:
            reason = f"allowed by role ({', '.join(via_role)})"
        elif via_user:
            reason = "allowed by an explicit user grant: this user id is in allowed_users " \
                     "(their roles do not match)"
        else:
            reason = "denied: no allowed role matches and there is no user grant"
        return {"user_id": u.id, "user_roles": list(u.roles), "doc_id": doc_id,
                "title": doc["title"], "classification": doc["classification"],
                "allowed_roles": doc["allowed_roles"], "allowed_users": doc["allowed_users"],
                "allowed": allowed, "via_role": via_role, "via_user_grant": via_user,
                "reason": reason}

    # ---- user management (admin, own tenant only) ---------------------------------
    @app.get("/admin/users")
    def list_users(admin=Depends(admin_user)):
        return {"users": [u.public() for u in app.state.users.list(admin.tenant_id)]}

    @app.post("/admin/users", status_code=201)
    def create_user(body: CreateUserRequest, admin=Depends(admin_user)):
        name = body.username.strip().lower()
        if not USERNAME_RE.match(name):
            raise HTTPException(422, "username: 3-40 chars, a-z 0-9 . _ - (start with a letter/digit)")
        _check_password(body.password)
        roles = _clean_roles(body.roles)
        # Every user id is prefixed with its tenant, so ids can't collide across companies.
        user_id = name if name.startswith(f"{admin.tenant_id}-") else f"{admin.tenant_id}-{name}"
        try:
            app.state.users.create(user_id, admin.tenant_id, roles, hash_password(body.password))
        except UserExists:
            raise HTTPException(409, "that user already exists")
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="user.create", target=user_id,
            detail={"roles": roles})
        return {"user_id": user_id, "tenant_id": admin.tenant_id, "roles": roles}

    @app.patch("/admin/users/{user_id}")
    def update_user(user_id: str, body: UpdateUserRequest, admin=Depends(admin_user)):
        roles = _clean_roles(body.roles) if body.roles is not None else None
        if body.password is not None:
            _check_password(body.password)
        if user_id == admin.id and (body.disabled or (roles is not None and "admin" not in roles)):
            raise HTTPException(400, "you can't disable yourself or remove your own admin role")
        ok = app.state.users.update(
            user_id, admin.tenant_id, roles=roles, disabled=body.disabled,
            password_hash=hash_password(body.password) if body.password is not None else None)
        if not ok:
            raise HTTPException(404, "user not found")      # also hides other tenants' users
        detail = {}
        if roles is not None:
            detail["roles"] = roles
        if body.disabled is not None:
            detail["disabled"] = body.disabled
        if body.password is not None:
            detail["password_changed"] = True
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="user.update", target=user_id,
            detail=detail)
        return app.state.users.get(user_id).public()

    @app.delete("/admin/users/{user_id}")
    def delete_user(user_id: str, admin=Depends(admin_user)):
        if user_id == admin.id:
            raise HTTPException(400, "you can't delete yourself")
        if not app.state.users.delete(user_id, admin.tenant_id):
            raise HTTPException(404, "user not found")      # also hides other tenants' users
        # Their old tokens already fail (user row is gone). Also remove their per-user
        # document grants so a future user re-using this id can't inherit them.
        n = app.state.store.remove_user_grants(admin.tenant_id, user_id)
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="user.delete", target=user_id,
            detail={"document_grants_removed": n})
        return {"user_id": user_id, "document_grants_removed": n}

    @app.delete("/admin/documents/{doc_id}")
    def delete_document(doc_id: str, admin=Depends(admin_user)):
        try:
            n = app.state.store.delete_document(doc_id, admin.tenant_id)
        except KeyError:
            raise HTTPException(404, "document not found")   # also hides other tenants' docs
        app.state.audit.record_admin(
            tenant_id=admin.tenant_id, actor=admin.id, action="document.delete", target=doc_id,
            detail={"chunks_deleted": n})
        return {"doc_id": doc_id, "chunks_deleted": n}

    return app


def __getattr__(name):           # lazy: `uvicorn app.main:app` builds the app on demand
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)