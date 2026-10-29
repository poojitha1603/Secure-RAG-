"""Qdrant wrapper. Every write path enforces: no chunk without a tenant_id."""
import uuid

from qdrant_client import QdrantClient, models as qm

from .embed import dim
from .models import CLASSIFICATIONS

COLLECTION = "chunks"
KEYWORD_INDEXES = ("tenant_id", "allowed_roles", "allowed_users", "doc_id")


class UnprotectedChunkError(ValueError):
    """Raised when a chunk has no tenant_id (or an otherwise invalid ACL)."""


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, url: str | None = None):
        # No URL -> embedded in-memory Qdrant (great for tests and CI).
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        want = dim()
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=qm.VectorParams(size=want, distance=qm.Distance.COSINE))
        else:
            have = self.client.get_collection(COLLECTION).config.params.vectors.size
            if have != want:
                raise RuntimeError(
                    f"Collection '{COLLECTION}' has vector size {have} but the current "
                    f"embedder produces {want}. Delete the collection and re-ingest "
                    f"(DELETE /collections/{COLLECTION} on your Qdrant server).")
        # Payload indexes keep filtered search fast on a real Qdrant server
        # (the embedded mode ignores them and prints a harmless warning).
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(
                COLLECTION, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)

    # ---- writes -----------------------------------------------------------
    @staticmethod
    def _validate(chunk: dict):
        p = chunk.get("payload") or {}
        tenant = p.get("tenant_id")
        if not isinstance(tenant, str) or not tenant.strip():
            raise UnprotectedChunkError(f"chunk {chunk.get('chunk_id')!r} has no tenant_id")
        if p.get("classification") not in CLASSIFICATIONS:
            raise UnprotectedChunkError(f"chunk {chunk.get('chunk_id')!r}: bad classification")
        for key in ("allowed_roles", "allowed_users"):
            if not isinstance(p.get(key), list):
                raise UnprotectedChunkError(f"chunk {chunk.get('chunk_id')!r}: {key} must be a list")

    def upsert_chunks(self, chunks: list[dict]):
        for c in chunks:                       # validate ALL before writing ANY
            self._validate(c)
        self.client.upsert(COLLECTION, wait=True, points=[
            qm.PointStruct(id=point_id(c["chunk_id"]), vector=c["vector"], payload=c["payload"])
            for c in chunks])

    def update_acl(self, doc_id: str, tenant_id: str,
                   allowed_roles: list[str], allowed_users: list[str]) -> int:
        """Rewrite the ACL on EVERY chunk of a document. Returns chunks updated."""
        flt = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
        ])
        n = self.client.count(COLLECTION, count_filter=flt, exact=True).count
        if n == 0:
            raise KeyError(doc_id)
        self.client.set_payload(
            COLLECTION, wait=True, points=flt,
            payload={"allowed_roles": list(allowed_roles), "allowed_users": list(allowed_users)})
        return n

    def remove_user_grants(self, tenant_id: str, user_id: str) -> int:
        """Remove `user_id` from allowed_users on every document in the tenant.
        Used when a user is deleted, so a NEW user who later gets the same id does not
        inherit the old user's access. Returns the number of documents changed."""
        flt = qm.Filter(must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="allowed_users", match=qm.MatchValue(value=user_id)),
        ])
        new_lists: dict[str, list[str]] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(COLLECTION, scroll_filter=flt, limit=256,
                                             offset=offset, with_payload=["doc_id", "allowed_users"])
            for p in pts:
                new_lists[p.payload["doc_id"]] = [u for u in p.payload["allowed_users"] if u != user_id]
            if offset is None:
                break
        for doc_id, users in new_lists.items():
            self.client.set_payload(
                COLLECTION, wait=True, payload={"allowed_users": users},
                points=qm.Filter(must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                    qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]))
        return len(new_lists)

    def delete_document(self, doc_id: str, tenant_id: str) -> int:
        """Delete every chunk of a document (own tenant only). Returns chunks deleted."""
        flt = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
        ])
        n = self.client.count(COLLECTION, count_filter=flt, exact=True).count
        if n == 0:
            raise KeyError(doc_id)
        self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=flt), wait=True)
        return n

    def list_documents(self, tenant_id: str) -> list[dict]:
        """One row per document in this tenant (aggregated from chunk payloads)."""
        flt = qm.Filter(must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
        docs: dict[str, dict] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(
                COLLECTION, scroll_filter=flt, limit=256, offset=offset,
                with_payload=["doc_id", "doc_title", "classification",
                              "allowed_roles", "allowed_users"])
            for p in pts:
                pl = p.payload
                d = docs.setdefault(pl["doc_id"], {
                    "doc_id": pl["doc_id"], "title": pl["doc_title"],
                    "classification": pl["classification"],
                    "allowed_roles": pl["allowed_roles"],
                    "allowed_users": pl["allowed_users"], "chunks": 0})
                d["chunks"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d["doc_id"])

    # ---- reads ------------------------------------------------------------
    def search(self, vector: list[float], flt: qm.Filter | None, limit: int):
        return self.client.query_points(
            COLLECTION, query=vector, query_filter=flt, limit=limit,
            with_payload=True).points

    def count(self) -> int:
        return self.client.count(COLLECTION, exact=True).count