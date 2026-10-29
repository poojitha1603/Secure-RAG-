"""Prove WHY we have two checks and why we pre-filter."""
from fastapi.testclient import TestClient
from qdrant_client import models as qm

from app import rag, seed
from app.acl import is_allowed
from app.embed import embed
from tests.conftest import auth


def test_second_check_catches_a_broken_prefilter(app):
    """Simulate a developer bug: the filter returns 'match everything'."""
    app.state.build_filter = lambda user: qm.Filter()      # the bug
    client = TestClient(app)
    before = rag.metrics["acl_violations"]

    r = client.post("/query", headers=auth("acme-employee"),
                    json={"query": "What are the salaries?", "top_k": 20})
    assert r.status_code == 200
    body = r.json()
    user = seed.USERS["acme-employee"]

    # Nothing forbidden reaches the response/LLM ...
    for s in body["sources"]:
        assert s["doc_id"].startswith("acme-") and s["classification"] != "confidential"
    # ... and the bug is loud: alert counter rises and the audit row records the drops.
    assert rag.metrics["acl_violations"] > before
    dropped = app.state.audit.rows()[-1][8]
    assert dropped != "[]"


def test_prefilter_beats_postfilter(app):
    """Post-filtering starves the user of results; pre-filtering does not."""
    user = seed.USERS["acme-employee"]
    q, k = "salary compensation pay bands", 3
    store = app.state.store

    pre, _ = rag.retrieve(store, user, q, top_k=k)           # filter INSIDE the search
    raw = store.search(embed(q), None, k)                    # unfiltered top-k ...
    post = [p for p in raw if is_allowed(user, p.payload)]   # ... then drop forbidden

    assert len(pre) == k
    assert len(post) < k, "expected the forbidden salary sheets to crowd out the top-k"
