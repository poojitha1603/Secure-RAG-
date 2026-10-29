"""Measure (1) latency overhead of the ACL filter and (2) revoke-to-effect time.

    python scripts/benchmark.py --filler 3000 --runs 200

NOTE: this uses Qdrant's embedded mode (brute force). On a real Qdrant server with
payload indexes the filtered path is much closer to unfiltered, so re-run with
QDRANT_URL=http://localhost:6333 before quoting a number on your resume.
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import rag, seed  # noqa: E402
from app.acl import acl_filter  # noqa: E402
from app.auth import issue_token  # noqa: E402
from app.embed import embed  # noqa: E402
from app.ingest import make_chunks  # noqa: E402
from app.main import Settings, create_app  # noqa: E402
from app.models import Document  # noqa: E402

WORDS = ("invoice ledger roster policy audit budget vendor contract shipment forecast "
         "pipeline release incident backlog sprint quota renewal onboarding compliance "
         "inventory warehouse refund ticket escalation roadmap hiring benefits").split()


def filler_docs(n_chunks: int):
    rng = random.Random(7)
    docs = []
    for i in range(n_chunks):
        t = seed.TENANT_IDS[i % len(seed.TENANT_IDS)]
        text = " ".join(rng.choices(WORDS, k=30))
        docs.append(Document(f"{t}-filler-{i}", t, f"filler {i}", "internal",
                             ("employee", "manager", "hr", "admin"), (), text))
    return docs


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", type=int, default=3000)
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()

    app = create_app(Settings(qdrant_url=os.getenv("QDRANT_URL"), audit_db=":memory:",
                              users_db=":memory:", jwt_secret="bench-secret-bench-secret-bench-secret", seed=True))
    store = app.state.store
    # Load in big batches (one request per 200 chunks, not one per document).
    batch = []
    for d in filler_docs(a.filler):
        batch += make_chunks(d)
        if len(batch) >= 200:
            store.upsert_chunks(batch)
            batch = []
    if batch:
        store.upsert_chunks(batch)
    print(f"chunks in store: {store.count()}")

    user = seed.USERS["acme-employee"]
    queries = [" ".join(random.Random(i).choices(WORDS, k=4)) for i in range(a.runs)]
    vecs = [embed(q) for q in queries]

    def timed(fn):
        out = []
        for v in vecs:
            t = time.perf_counter()
            fn(v)
            out.append((time.perf_counter() - t) * 1000)
        return out

    timed(lambda v: store.search(v, None, 10))                       # warm-up
    base = timed(lambda v: store.search(v, None, 10))
    filt = timed(lambda v: store.search(v, acl_filter(user), 10))
    full = []
    for q in queries:                                                # filter + 2nd check
        t = time.perf_counter()
        rag.retrieve(store, user, q, top_k=10)
        full.append((time.perf_counter() - t) * 1000)

    print(f"{'':28}{'p50 ms':>9}{'p95 ms':>9}")
    for name, xs in (("no filter (baseline)", base), ("ACL pre-filter", filt),
                     ("pre-filter + 2nd check", full)):
        print(f"{name:28}{statistics.median(xs):9.2f}{pct(xs, .95):9.2f}")
    print(f"filtering overhead (p50): {statistics.median(filt) - statistics.median(base):+.2f} ms")

    # revoke-to-effect
    c = TestClient(app)
    tok = lambda uid: {"Authorization": "Bearer " + issue_token(seed.USERS[uid], "bench-secret-bench-secret-bench-secret")}
    doc = "acme-mgr-restructuring"
    ask = {"query": "restructuring plan next quarter", "top_k": 20}
    seen = lambda: doc in {s["doc_id"] for s in c.post("/query", headers=tok("acme-manager"), json=ask).json()["sources"]}
    assert seen(), "manager should see the doc before revoke"
    t = time.perf_counter()
    c.put(f"/admin/documents/{doc}/acl", headers=tok("acme-admin"),
          json={"allowed_roles": ["hr"], "allowed_users": []})
    gone = not seen()
    print(f"revoke-to-effect: {(time.perf_counter() - t) * 1000:.1f} ms "
          f"(gone on the very next query: {gone})")


if __name__ == "__main__":
    main()