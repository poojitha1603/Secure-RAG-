"""Failed-login throttling (in memory, per process).

Two counters over a sliding window: one per username, one per client IP. When either is
over its limit, /login answers 429 with Retry-After BEFORE any password is checked.
A successful login clears that username's counter. Unknown usernames are counted exactly
like real ones, so the throttle never reveals which usernames exist.

Limits: it is per process (use Redis if you run several workers), and it never trusts
X-Forwarded-For (behind a proxy, every client would share the proxy's IP).
"""
import threading
import time
from collections import deque


class LoginThrottle:
    def __init__(self, max_per_user: int = 5, max_per_ip: int = 30,
                 window_seconds: float = 300, clock=time.monotonic):
        self.max_per_user, self.max_per_ip = max_per_user, max_per_ip
        self.window, self.clock = window_seconds, clock
        self._fails: dict[tuple[str, str], deque] = {}
        self._lock = threading.Lock()

    def _prune(self, key, now) -> deque:
        q = self._fails.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if not q:
            self._fails.pop(key, None)
        return q

    def retry_after(self, username: str, ip: str) -> int:
        """Seconds to wait (0 = allowed)."""
        now = self.clock()
        waits = [0]
        with self._lock:
            for key, limit in ((("u", username), self.max_per_user), (("ip", ip), self.max_per_ip)):
                q = self._prune(key, now)
                if len(q) >= limit:
                    waits.append(int(self.window - (now - q[len(q) - limit])) + 1)
        return max(waits)

    def failure(self, username: str, ip: str):
        now = self.clock()
        with self._lock:
            for key in (("u", username), ("ip", ip)):
                self._fails.setdefault(key, deque()).append(now)

    def success(self, username: str):
        with self._lock:
            self._fails.pop(("u", username), None)