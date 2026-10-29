"""Core data shapes: who is asking (User) and what is protected (Document)."""
from dataclasses import dataclass

CLASSIFICATIONS = ("public", "internal", "confidential")


@dataclass(frozen=True)
class User:
    id: str
    tenant_id: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: str
    title: str
    classification: str          # public | internal | confidential
    allowed_roles: tuple[str, ...]
    allowed_users: tuple[str, ...]
    text: str
