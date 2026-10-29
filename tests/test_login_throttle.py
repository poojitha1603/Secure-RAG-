"""Failed-login throttling: lockout, no username enumeration, recovery, window expiry."""
import pytest
from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.throttle import LoginThrottle
from tests.conftest import SECRET, SEED_PASSWORD


def make(max_failures=3):
    return TestClient(create_app(Settings(
        qdrant_url=None, audit_db=":memory:", users_db=":memory:", jwt_secret=SECRET, seed=True,
        enable_dev_tokens=True, seed_password=SEED_PASSWORD, login_max_failures=max_failures)))


def login(c, user, pw):
    return c.post("/login", json={"username": user, "password": pw})


def test_locks_after_max_failures_even_for_the_correct_password():
    c = make(3)
    for _ in range(3):
        assert login(c, "acme-hr", "wrong-password").status_code == 401
    r = login(c, "acme-hr", SEED_PASSWORD)                  # correct, but locked
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    assert "access_token" not in r.text


def test_below_the_limit_is_not_locked_and_success_resets_the_counter():
    c = make(3)
    for _ in range(2):
        login(c, "acme-hr", "wrong-password")
    assert login(c, "acme-hr", SEED_PASSWORD).status_code == 200       # resets
    for _ in range(2):
        assert login(c, "acme-hr", "wrong-password").status_code == 401  # 2 fresh failures, not 4
    assert login(c, "acme-hr", SEED_PASSWORD).status_code == 200


def test_unknown_and_real_usernames_are_throttled_identically():
    c = make(3)
    for u in ("acme-hr", "nobody-here"):
        codes = [login(c, u, "wrong-password").status_code for _ in range(4)]
        assert codes == [401, 401, 401, 429], u


def test_username_case_variants_share_one_counter():
    c = make(3)
    for u in ("acme-hr", "ACME-HR", "Acme-Hr"):
        login(c, u, "wrong-password")
    assert login(c, "acme-hr", SEED_PASSWORD).status_code == 429


def test_one_users_lockout_does_not_block_another_user():
    c = make(3)
    for _ in range(3):
        login(c, "acme-hr", "wrong-password")
    assert login(c, "acme-employee", SEED_PASSWORD).status_code == 200


def test_ip_limit_blocks_password_spraying_across_many_usernames():
    c = make(3)                                   # per-IP limit = max(30, 18) = 30
    codes = [login(c, f"user-{i}", "wrong-password").status_code for i in range(31)]
    assert codes[:30] == [401] * 30 and codes[30] == 429
    assert login(c, "acme-hr", SEED_PASSWORD).status_code == 429       # same IP is blocked


def test_x_forwarded_for_cannot_be_used_to_dodge_the_limit():
    c = make(3)
    for i in range(3):
        c.post("/login", json={"username": "acme-hr", "password": "wrong-password"},
               headers={"X-Forwarded-For": f"10.0.0.{i}"})
    r = c.post("/login", json={"username": "acme-hr", "password": SEED_PASSWORD},
               headers={"X-Forwarded-For": "10.9.9.9"})
    assert r.status_code == 429


def test_lockout_expires_after_the_window():
    now = [0.0]
    t = LoginThrottle(max_per_user=3, window_seconds=300, clock=lambda: now[0])
    for _ in range(3):
        t.failure("acme-hr", "1.2.3.4")
    assert 0 < t.retry_after("acme-hr", "1.2.3.4") <= 301
    now[0] = 299; assert t.retry_after("acme-hr", "1.2.3.4") > 0
    now[0] = 301; assert t.retry_after("acme-hr", "1.2.3.4") == 0


def test_throttle_memory_is_pruned():
    now = [0.0]
    t = LoginThrottle(max_per_user=3, window_seconds=10, clock=lambda: now[0])
    for i in range(50):
        t.failure(f"u{i}", "ip")
    now[0] = 11
    for i in range(50):
        t.retry_after(f"u{i}", "ip")
    t.retry_after("x", "ip")
    assert len(t._fails) <= 1


def test_tokens_and_other_endpoints_are_unaffected_by_login_lockout():
    c = make(3)
    for _ in range(3):
        login(c, "acme-hr", "wrong-password")
    assert c.post("/token", json={"user_id": "acme-hr"}).status_code == 200