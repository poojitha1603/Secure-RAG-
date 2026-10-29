"""JWT auth. Identity (user, tenant, roles) comes from the verified token ONLY."""
import time

import jwt

from .models import User

ALGO = "HS256"


class AuthError(Exception):
    pass


def issue_token(user: User, secret: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": user.id, "tenant_id": user.tenant_id, "roles": list(user.roles),
         "iat": now, "exp": now + ttl_seconds},
        secret, algorithm=ALGO)


def verify_token(token: str, secret: str) -> User:
    try:
        # algorithms=[...] pins HS256, which blocks the classic alg=none trick.
        claims = jwt.decode(token, secret, algorithms=[ALGO],
                            options={"require": ["exp", "sub"]})
    except jwt.PyJWTError as e:
        raise AuthError(str(e)) from e

    tenant, roles = claims.get("tenant_id"), claims.get("roles")
    if not isinstance(tenant, str) or not tenant:
        raise AuthError("token has no tenant_id")
    if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
        raise AuthError("token roles malformed")
    return User(id=claims["sub"], tenant_id=tenant, roles=tuple(roles))
