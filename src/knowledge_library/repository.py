import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Tuple
from jsonschema import validate, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema", "artifact_schema.json")

# AWS RDS Endpoint provided
DEFAULT_RDS_ENDPOINT = "database-1.cvsuygmckyxi.us-east-2.rds.amazonaws.com"
DEFAULT_DB_PORT = "5432"


class ArtifactRepository:
    def __init__(self, root: str = None, db_url: str = None):
        self.root = root or os.path.dirname(__file__)
        
        # Reads DATABASE_URL from environment or falls back to AWS RDS string
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            f"postgresql://postgres:YOUR_PASSWORD@{DEFAULT_RDS_ENDPOINT}:{DEFAULT_DB_PORT}/postgres"
        )
        self.schema_path = SCHEMA_PATH
        self._init_db()

    def _get_connection(self):
        """Creates a connection to PostgreSQL."""
        return psycopg2.connect(self.db_url)

    def _init_db(self):
        """Initializes the PostgreSQL table schema if it does not exist."""
        query = """
        CREATE TABLE IF NOT EXISTS artifacts (
            id VARCHAR(128) PRIMARY KEY,
            name VARCHAR(255),
            version VARCHAR(64),
            preconditions TEXT,
            data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def validate_artifact(self, artifact: Dict) -> None:
        if os.path.exists(self.schema_path):
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            validate(instance=artifact, schema=schema)

    def add_artifact(self, artifact: Dict) -> str:
        # Validate JSON schema
        self.validate_artifact(artifact)
        
        aid = artifact["id"]
        name = artifact.get("name", "")
        version = artifact.get("version", "v1.0.0")
        preconditions = artifact.get("preconditions", "")
        
        if isinstance(preconditions, list):
            preconditions_str = " ".join(preconditions)
        else:
            preconditions_str = str(preconditions)

        # Insert or update in PostgreSQL
        query = """
        INSERT INTO artifacts (id, name, version, preconditions, data)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE 
        SET name = EXCLUDED.name,
            version = EXCLUDED.version,
            preconditions = EXCLUDED.preconditions,
            data = EXCLUDED.data;
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (aid, name, version, preconditions_str, json.dumps(artifact)))
            conn.commit()
        return aid

    def list_artifacts(self) -> List[Dict]:
        query = "SELECT id, name, version, preconditions FROM artifacts;"
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_artifact(self, aid: str) -> Optional[Dict]:
        query = "SELECT data FROM artifacts WHERE id = %s;"
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (aid,))
                row = cur.fetchone()
                if not row:
                    return None
                data = row["data"]
                return json.loads(data) if isinstance(data, str) else data

    def _build_index(self) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
        query = "SELECT id, preconditions FROM artifacts;"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
        
        ids = [r[0] for r in rows]
        docs = [r[1] or "" for r in rows]
        if not docs or not any(d.strip() for d in docs):
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
            if art:
                results.append(dict(artifact=art, score=float(sims[i])))
        return results
