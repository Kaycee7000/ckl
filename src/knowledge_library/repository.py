import os
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from jsonschema import validate, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema", "artifact_schema.json")


class ArtifactRepository:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.artifacts_dir = os.path.join(self.root, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.db_path = os.path.join(self.root, "artifacts.db")
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                name TEXT,
                version TEXT,
                path TEXT,
                preconditions TEXT
            )
            """
        )
        self.conn.commit()

    def validate_artifact(self, artifact: Dict) -> None:
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        validate(instance=artifact, schema=schema)

    def add_artifact(self, artifact: Dict) -> str:
        # validate
        try:
            self.validate_artifact(artifact)
        except ValidationError as e:
            raise
        aid = artifact["id"]
        path = os.path.join(self.artifacts_dir, f"{aid}.json")
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO artifacts (id, name, version, path, preconditions) VALUES (?, ?, ?, ?, ?)",
            (aid, artifact.get("name"), artifact.get("version"), path, artifact.get("preconditions", "")),
        )
        self.conn.commit()
        return aid

    def list_artifacts(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, version, path FROM artifacts")
        rows = cur.fetchall()
        return [dict(id=r[0], name=r[1], version=r[2], path=r[3]) for r in rows]

    def get_artifact(self, aid: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT path FROM artifacts WHERE id = ?", (aid,))
        row = cur.fetchone()
        if not row:
            return None
        with open(row[0], "r") as f:
            return json.load(f)

    def _build_index(self) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, preconditions FROM artifacts")
        rows = cur.fetchall()
        ids = [r[0] for r in rows]
        docs = [r[1] or "" for r in rows]
        if not docs:
            return TfidfVectorizer(), np.zeros((0, 0)), ids
        vec = TfidfVectorizer().fit(docs)
        X = vec.transform(docs).toarray()
        return vec, X, ids

    def search_by_preconditions(self, query: str, top_k: int = 5) -> List[Dict]:
        vec, X, ids = self._build_index()
        if X.size == 0:
            return []
        qv = vec.transform([query]).toarray()[0]
        sims = X.dot(qv)
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            aid = ids[i]
            art = self.get_artifact(aid)
            results.append(dict(artifact=art, score=float(sims[i])))
        return results
