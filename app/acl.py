"""The two independent ACL checks (defense in depth).

1. acl_filter(user)  -> pushed INTO the vector query (pre-filter).
2. is_allowed(user, payload) -> re-checked in Python after retrieval.

They are written separately on purpose: a bug in one should not be a bug in both.
"""
from qdrant_client import models as qm


def acl_filter(user) -> qm.Filter:
    """tenant must match AND (one of my roles is allowed OR my user id is allowed).

    The OR is nested in its own Filter so it can never be misread as
    "should" sitting beside "must" (Qdrant treats that combination subtly).
    """
    either = []
    if user.roles:
        either.append(qm.FieldCondition(
            key="allowed_roles", match=qm.MatchAny(any=list(user.roles))))
    either.append(qm.FieldCondition(
        key="allowed_users", match=qm.MatchValue(value=user.id)))

    return qm.Filter(must=[
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=user.tenant_id)),
        qm.Filter(should=either),
    ])


def _is_str_list(value) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value)


def is_allowed(user, payload) -> bool:
    """Plain-Python version of the same rule, applied to each retrieved chunk.

    Fails CLOSED: anything missing or of the wrong type means "not allowed".
    (A string where a list is expected would otherwise make `in` do substring
    matching, e.g. "initech-emp" in "initech-employee".)
    """
    try:
        tenant = payload.get("tenant_id")
        if not isinstance(tenant, str) or not tenant or tenant != user.tenant_id:
            return False
        roles, users = payload.get("allowed_roles"), payload.get("allowed_users")
        if not _is_str_list(roles) or not _is_str_list(users):
            return False
        return bool(set(user.roles) & set(roles)) or user.id in users
    except Exception:
        return False