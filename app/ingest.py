"""Ingestion: split a document into chunks and copy its ACL onto every chunk."""
import re

from .embed import embed
from .models import Document


def chunk_text(text: str, max_chars: int = 350) -> list[str]:
    chunks: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if cur and len(cur) + len(sent) + 1 > max_chars:
                chunks.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            chunks.append(cur)
    return chunks


def make_chunks(doc: Document) -> list[dict]:
    """Every chunk inherits the document's ACL as payload metadata."""
    return [{
        "chunk_id": f"{doc.doc_id}#{i}",
        "vector": embed(text),
        "payload": {
            "chunk_id": f"{doc.doc_id}#{i}",
            "doc_id": doc.doc_id,
            "doc_title": doc.title,
            "tenant_id": doc.tenant_id,
            "classification": doc.classification,
            "allowed_roles": list(doc.allowed_roles),
            "allowed_users": list(doc.allowed_users),
            "text": text,
        },
    } for i, text in enumerate(chunk_text(doc.text))]


def ingest_document(store, doc: Document) -> int:
    chunks = make_chunks(doc)
    store.upsert_chunks(chunks)      # store rejects chunks without tenant_id
    return len(chunks)


def ingest_all(store, docs) -> int:
    return sum(ingest_document(store, d) for d in docs)
