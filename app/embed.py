"""Embeddings, switchable with the EMBEDDER env var.

  EMBEDDER=hash        (default) tiny offline hashed bag-of-words. Used by tests/CI.
  EMBEDDER=fastembed   real semantic model (needs `pip install fastembed`, downloads
                       the model once). Model: FASTEMBED_MODEL, default BAAI/bge-small-en-v1.5.

Nothing else in the project cares how vectors are made, except that a Qdrant
collection is tied to one vector size: change embedder -> delete the collection.
"""
import hashlib
import os
import re

import numpy as np

HASH_DIM = 512
DIM = HASH_DIM   # kept for backwards compatibility; use dim() for the real value
_STOP = {"the", "a", "an", "of", "to", "and", "is", "are", "in", "for", "on", "all",
         "what", "me", "my", "show", "your", "you", "list", "with", "from", "at"}

_fast_model = None
_fast_dim = None


def backend() -> str:
    return os.getenv("EMBEDDER", "hash").lower()


def _hash_embed(text: str) -> list[float]:
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        v[h % HASH_DIM] += 1.0
    v = np.where(v > 0, 1.0 + np.log(np.maximum(v, 1.0)), 0.0).astype(np.float32)
    norm = float(np.linalg.norm(v)) or 1.0
    return (v / norm).tolist()


def _fast_embed(text: str) -> list[float]:
    global _fast_model, _fast_dim
    if _fast_model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError("EMBEDDER=fastembed needs: pip install fastembed") from e
        _fast_model = TextEmbedding(
            model_name=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    vec = next(iter(_fast_model.embed([text])))
    out = [float(x) for x in vec]
    _fast_dim = len(out)
    return out


def embed(text: str) -> list[float]:
    return _fast_embed(text) if backend() == "fastembed" else _hash_embed(text)


def dim() -> int:
    """Vector size of the active embedder."""
    if backend() == "fastembed":
        return len(embed("dimension probe"))
    return HASH_DIM


def _reset_for_tests():
    global _fast_model, _fast_dim
    _fast_model = None
    _fast_dim = None