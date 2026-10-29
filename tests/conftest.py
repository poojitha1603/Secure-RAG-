import os

# Tests are always offline and deterministic: use the hash embedder even if your .env says
# EMBEDDER=fastembed. (Must happen before app modules read the environment.)
os.environ["EMBEDDER"] = "hash"

import pytest
from fastapi.testclient import TestClient

from app import seed
from app.main import Settings, create_app
from app.auth import issue_token
from tests.stats import STATS

SECRET = "test-secret-at-least-32-bytes-long-xxxxxxxx"
SEED_PASSWORD = "demo-password"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Tests must never call a real LLM, even if GROQ_API_KEY is set on your machine."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def make_app():
    return create_app(Settings(qdrant_url=None, audit_db=":memory:", users_db=":memory:",
                               jwt_secret=SECRET, seed=True, enable_dev_tokens=True,
                               seed_password=SEED_PASSWORD))


@pytest.fixture()
def app():
    """Fresh app (fresh store + audit log) per test. Use for tests that mutate state."""
    return make_app()


@pytest.fixture(scope="module")
def shared_app():
    """One app per module. Use only for read-only tests."""
    return make_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


def auth(user_id: str) -> dict:
    return {"Authorization": f"Bearer {issue_token(seed.USERS[user_id], SECRET)}"}


def pytest_terminal_summary(terminalreporter):
    terminalreporter.section("leakage summary")
    terminalreporter.write_line(
        f"adversarial queries run: {STATS['adversarial_queries']} | "
        f"forbidden chunks returned: {STATS['forbidden_chunks_returned']}")