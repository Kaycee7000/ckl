import os
import json
import sqlite3
from typing import List, Optional

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from knowledge_library.repository import ArtifactRepository


class VectorArtifactRepository(ArtifactRepository):
    """Extends ArtifactRepository with FAISS dense vector search over preconditions and metadata.

    If FAISS or sentence-transformers are not available, raises RuntimeError on semantic methods.
    """

    def __init__(self, root: str = '.', model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(root)
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.index = None
        self.artifact_ids: List[str] = []
        if FAISS_AVAILABLE:
            # lazy model init
            self.model = SentenceTransformer(self.model_name)
            # infer dim
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self._rebuild_vector_index()

    def _text_for_artifact(self, art: dict) -> str:
        title = art.get('name', '')
        desc = art.get('description', '') or ''
        pre = art.get('preconditions', '') or ''
        if isinstance(pre, list):
            pre = ' '.join(pre)
        return f"{title}. {desc}. Preconditions: {pre}"

    def _rebuild_vector_index(self):
        if not FAISS_AVAILABLE:
            return
        artifacts = self.list_artifacts()
        if not artifacts:
            self.index.reset()
            self.artifact_ids = []
            return
        texts = [self._text_for_artifact(a) for a in artifacts]
        ids = [a['id'] for a in artifacts]
        emb = self.model.encode(texts, normalize_embeddings=True)
        emb = np.array(emb, dtype=np.float32)
        self.index.reset()
        if emb.size > 0:
            self.index.add(emb)
        self.artifact_ids = ids

    def add_artifact(self, artifact: dict) -> str:
        aid = super().add_artifact(artifact)
        if FAISS_AVAILABLE:
            txt = self._text_for_artifact(artifact)
            vec = self.model.encode([txt], normalize_embeddings=True)
            vec = np.array(vec, dtype=np.float32)
            self.index.add(vec)
            self.artifact_ids.append(aid)
        return aid

    def search_preconditions_semantic(self, query: str, top_k: int = 5) -> List[dict]:
        if not FAISS_AVAILABLE:
            raise RuntimeError('FAISS or sentence-transformers not available')
        if self.index.ntotal == 0:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.array(qv, dtype=np.float32)
        # FAISS expects shape (n, d)
        scores, indices = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self.artifact_ids):
                continue
            aid = self.artifact_ids[idx]
            art = self.get_artifact(aid)
            results.append({'artifact': art, 'similarity_score': float(score)})
        return results
