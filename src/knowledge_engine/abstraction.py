from typing import Dict, Optional

# Try to use sentence-transformers for semantic matching, but fall back to simple
# substring matching when the package is not available.
_EMBEDDING_AVAILABLE = False
_EMBED_MODEL = None
_EMBED_CACHE = {}
try:
    from sentence_transformers import SentenceTransformer
    import numpy as _np
    from numpy.linalg import norm as _norm
    _EMBEDDING_AVAILABLE = True
except Exception:
    _EMBEDDING_AVAILABLE = False


def _ensure_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def abstract_variable(name: str, mapping: Dict[str, str], default: Optional[str] = None, similarity_threshold: float = 0.65) -> str:
    """Map a domain-specific variable name to a domain-agnostic primitive using mapping.

    If `sentence-transformers` is available, performs semantic matching via
    cosine similarity between embeddings of `name` and mapping keys and returns
    the mapped value with highest similarity above `similarity_threshold`.
    Otherwise falls back to substring matching as before.
    """
    low = name.lower()
    # quick substring match first (keeps existing behavior fast)
    for key, val in mapping.items():
        if key.lower() in low:
            return val

    if not _EMBEDDING_AVAILABLE:
        return default or name

    # semantic matching
    model = _ensure_model()
    # compute embedding for name (cache by exact string)
    if name in _EMBED_CACHE:
        emb_name = _EMBED_CACHE[name]
    else:
        emb_name = model.encode(name)
        _EMBED_CACHE[name] = emb_name

    best_score = -1.0
    best_val = None
    # compute embeddings for keys and compare
    for key, val in mapping.items():
        if key in _EMBED_CACHE:
            emb_key = _EMBED_CACHE[key]
        else:
            emb_key = model.encode(key)
            _EMBED_CACHE[key] = emb_key
        score = float(_np.dot(emb_name, emb_key) / (_norm(emb_name) * _norm(emb_key) + 1e-12))
        if score > best_score:
            best_score = score
            best_val = val

    if best_score >= similarity_threshold:
        return best_val
    return default or name
