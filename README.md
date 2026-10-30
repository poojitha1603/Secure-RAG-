# SecureRAG: permission-aware multi-tenant RAG

![tests](https://img.shields.io/badge/tests-148%20passing-brightgreen) ![leaks](https://img.shields.io/badge/adversarial%20leaks-0%20%2F%20624-brightgreen)

A question-answering system for several companies at once, where **what you can retrieve depends on who you are**.
Identity comes only from a signed JWT. The access rule is applied *inside* the vector search, a second check
in Python catches any filter bug, every retrieval is audited, and a CI-run adversarial suite proves zero
cross-tenant leaks.

```
request + JWT -> verify (user, tenant, roles) -> build ACL filter -> Qdrant search WITH filter
              -> re-check every chunk in Python (drop + alert) -> prompt -> answer -> audit log
```

| Employee asks "What are the salaries?" | HR asks the same question |
|---|---|
| ![employee](docs/employee-chat.png) | ![hr](docs/hr-chat.png) |

Same question, same company, different people: the employee never receives the confidential salary
sheet, not even as text inside the prompt sent to the LLM.

## Features

- **Tenant isolation and role ACLs** on every chunk (`tenant_id`, `allowed_roles`, `allowed_users`, `classification`).
- **Pre-filter + second check**: the filter runs inside Qdrant; `is_allowed` re-checks each result and fails closed on malformed data.
- **Identity from the token only**: tenant/roles/user in the request body are ignored. The user is re-checked in the database on every request, so disabling a user or changing roles takes effect immediately, even for old tokens.
- **Real login**: salted scrypt password hashes, `POST /login`, and rate limiting with lockout on repeated failures.
- **Admin dashboard**: upload TXT/MD/PDF with an ACL, edit who can see a document, delete documents, manage users (create, change roles, disable, delete), see who retrieved what, and "explain why this user can or can't see this document".
- **Append-only audit**: every retrieval, and every admin action, in SQLite tables protected by triggers (never passwords or document text).
- **Chat UI** with a live "what the server reads from your token" panel and a forged-tenant demo.
- **Optional real parts**: Groq LLM answers, semantic embeddings (`fastembed`), Qdrant server.
- **Tests**: 148 tests, including a 624-query adversarial leakage suite (prompt injection, forged tokens, planted instructions, cross-tenant probes) checked against an independent oracle.

## Run it

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env            # then set JWT_SECRET and SEED_PASSWORD
python -m pytest -q               # expect: 148 passed, 0 forbidden chunks returned
uvicorn app.main:app --reload     # http://127.0.0.1:8000  (seeds 3 fictional companies)
python scripts/benchmark.py       # filter latency + revoke-to-effect time
```

Open http://127.0.0.1:8000/ and sign in. Seeded users are `{acme,globex,initech}-{employee,manager,hr,admin}`;
all use the password in `SEED_PASSWORD` (default `demo-password`, **change it**).

Try it from the command line (dev shortcut, see below):

```bash
TOKEN=$(curl -s localhost:8000/token -H 'content-type: application/json' \
        -d '{"user_id":"acme-employee"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -s localhost:8000/query -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"query":"What are the salaries?"}'
```

## How access works

- **Identity**: `POST /login` returns a signed JWT (HS256, algorithm pinned). Each request is re-checked against the users table.
- **A user may see a chunk if** it is in their tenant **and** (they have one of its `allowed_roles` **or** they are in its `allowed_users`). Everything else is invisible.
- **Seed data** per company: public overview, internal handbook, HR-only salary sheet, manager-only restructuring plan, HR-only performance review. Each has a canary marker so tests can detect leaked text.
- **Uploads** always go to the admin's own tenant, taken from the token.
- **Deleting a user** also removes their per-user document grants, so a new person who later gets the same id cannot inherit the old access.

## Login protection

`POST /login` is rate limited. After `LOGIN_MAX_FAILURES` (default 5) failed attempts for a username within
`LOGIN_WINDOW_SECONDS` (default 300) it answers **429 with `Retry-After`**, even for the right password.
A per-IP limit stops password spraying across many usernames. Unknown and real usernames are throttled
identically, so it never reveals which usernames exist. It is in memory and per process, and it ignores
`X-Forwarded-For`.

## Dev quick sign-in (development only)

`POST /token` (no password) and the Chat page's "Dev quick sign-in" dropdown only exist while `DEV_TOKENS=1`
(the default). **Set `DEV_TOKENS=0` before deploying.** The dropdown is loaded live from `GET /dev/users`
(active users only; signed-in users see only their own company). It sends just a user id: tenant and roles
always come from the database. It refreshes after create/disable/delete in Admin > Users.

## Admin tab

Sign in as an admin (e.g. `acme-admin`) and open **Admin** (hidden for others, and the server refuses non-admins anyway).

![admin](docs/admin-users.png)

- **Documents**: upload TXT/MD/PDF with a classification and allowed roles/users, edit who can see a document (applies from the next query), delete, see who retrieved it.
- **Users**: add users to your company, change roles, disable/enable, delete.
- **Activity**: recent questions in your company (with how many chunks the second ACL check had to block, which should always be 0) and the append-only log of admin actions.

## Real semantic search (optional)

The default `hash` embedder is offline and only for tests and demos. For real semantic search:

```powershell
pip install fastembed
$env:EMBEDDER='fastembed'     # or put EMBEDDER=fastembed in .env
```

Tests always use the hash embedder, whatever `.env` says. If a Qdrant collection was built with the other
embedder, delete it first (vector sizes differ):
`Invoke-RestMethod -Method Delete http://127.0.0.1:6333/collections/chunks`.

## Real LLM answers with Groq (optional)

```powershell
$env:GROQ_API_KEY='gsk_...'     # free key at console.groq.com
uvicorn app.main:app --reload
```

The model only ever sees chunks that passed both ACL checks. If the call fails, the endpoint falls back to
the offline stub answer. Tests never call Groq, even if the key is set.

## Config (env vars)

`JWT_SECRET` (**set a real one**), `SEED_PASSWORD`, `SEED` (default 1), `DEV_TOKENS` (default 1),
`QDRANT_URL` (unset = embedded in-memory, re-seeded on every start), `AUDIT_DB`, `USERS_DB`,
`EMBEDDER` (`hash` or `fastembed`), `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`),
`LOGIN_MAX_FAILURES` (default 5), `LOGIN_WINDOW_SECONDS` (default 300). See `.env.example`. Never commit `.env`.

## Map to the 8 build steps

| Step | Where |
|---|---|
| 1 Access model + seed | `app/models.py`, `app/seed.py` |
| 2 ACLs in ingestion, reject no-tenant, indexes | `app/ingest.py`, `app/store.py` |
| 3 Identity from token only | `app/auth.py`, `QueryRequest` ignores extra body fields in `app/main.py` |
| 4 Pre-filter | `app/acl.py::acl_filter`, used in `app/rag.py::retrieve` |
| 5 Second check + alert | `app/acl.py::is_allowed`, `app/rag.py::retrieve` |
| 6 Audit log + admin endpoint | `app/audit.py`, `GET /admin/documents/{id}/accesses` |
| 7 Permission changes + revoke test | `store.update_acl`, `PUT /admin/documents/{id}/acl`, `tests/test_permission_changes.py` |
| 8 Leakage suite + CI | `tests/test_leakage.py`, `.github/workflows/ci.yml` |

## Known limits and future work

- Logout only clears the token in the browser (JWTs are stateless). To cut a user off at once, disable or delete them.
- Login throttling is per process; use Redis if you run several workers.
- Users and audit logs use SQLite. For production, move to Postgres (same columns; use `REVOKE UPDATE, DELETE` instead of the SQLite triggers).
- The embedded Qdrant re-seeds on every restart. Use a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`, then `QDRANT_URL=http://localhost:6333`) to persist data.
- No answer cache exists. If you add one, put `tenant_id` and the role set in the cache key; summaries, whole-document embeddings and chat history would need the same ACL payload.
- Not yet: Dockerfile, SSO/OIDC, per-document expiry.
![CI](https://github.com/poojitha1603/Secure-RAG-/actions/workflows/ci.yml/badge.svg)
## Deploy (Render, free plan)

The repo includes `render.yaml`, so deploying is a few clicks:

1. Push to GitHub, then on https://render.com choose **New > Blueprint** and pick this repo.
2. When asked, type a value for `SEED_PASSWORD`. It becomes the password of all demo accounts (`acme-employee`, `acme-hr`, `acme-admin`, ...), so treat it as a public demo login.
3. Deploy. Open the URL Render gives you and sign in. The app answers `/health` once it is up.

What the blueprint sets: `DEV_TOKENS=0` (no password-less sign-in), a random 48-character `JWT_SECRET`, `EMBEDDER=hash` (fastembed needs more RAM than the free plan has), and a higher per-IP login limit because behind Render's proxy all visitors can look like one IP (the per-username lockout still applies).

The app refuses to start with `DEV_TOKENS=0` unless `JWT_SECRET` is at least 32 random characters, so a guessable signing key cannot reach production by mistake.

Free-plan behaviour: the service sleeps after about 15 minutes idle (the first request then takes ~30-60 s), and the disk is wiped on every restart, so users, uploads and the audit log reset to the seed data. That is fine for a demo. For real use, use a Qdrant server, Postgres and a persistent disk. Changing `SEED_PASSWORD` has no effect once a `users.db` already exists.

Anyone with the demo password can sign in as an admin and upload or delete documents *in the fictional demo companies*. Keep real or sensitive documents out of a public demo.