"""Optional shared-key access control and a small in-memory rate limiter.

Set RETAILVISION_API_KEY to require a key on every route except the page, static files,
/login and /health. The key may be sent as the X-API-Key header or the HttpOnly cookie that
POST /login sets. With no key configured the server is open (fine on a trusted laptop).
"""
import hmac
import os
import time
from collections import defaultdict, deque

COOKIE = "rv_key"
PUBLIC_PATHS = {"/", "/login", "/health"}
PUBLIC_PREFIXES = ("/static/",)


def configured_key():
    return os.getenv("RETAILVISION_API_KEY", "")


def key_matches(supplied):
    key = configured_key()
    return bool(key) and hmac.compare_digest(supplied.encode(), key.encode())


def is_public(path):
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def is_authorized(request):
    if not configured_key():
        return True
    supplied = request.headers.get("x-api-key") or request.cookies.get(COOKIE) or ""
    return key_matches(supplied)


class RateLimiter:
    """Sliding window: at most `limit` hits per `window` seconds per client."""

    def __init__(self, limit, window=60.0):
        self.limit, self.window = limit, window
        self._hits = defaultdict(deque)

    def allow(self, client):
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
