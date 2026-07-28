import duckdb
import pandas as pd
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DuckDBStorage:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(database=db_path)

    def create_table(self, table_name: str = "data"):
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                domain TEXT,
                entity_id TEXT,
                timestamp TIMESTAMP,
                value DOUBLE,
                metadata JSON
            )
            """
        )

    def write_dataframe(self, df: pd.DataFrame, table_name: str = "data"):
        # Expect columns: domain, entity_id, timestamp, value, metadata
        # Enforce UTC for timestamp column to avoid timezone leakage
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self.conn.register("_tmp_df", df)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM _tmp_df")

    def query(self, sql: str, params: Optional[List[Any]] = None, partition_granularity: str = "D") -> pd.DataFrame:
        """Execute a SQL query. If a timestamp predicate is present (and params provided),
        log partition-pruning information using the provided partition granularity.
        """
        # Simple heuristic: if query references `timestamp` and params include a datetime-like cutoff,
        # log partition pruning intent using UTC-normalized value.
        if "timestamp" in sql.lower() and params:
            # try to find a datetime in params
            for p in params:
                try:
                    ts = pd.to_datetime(p, utc=True)
                    logger.info("Partition pruning: detected timestamp predicate, cutoff=%s (UTC), granularity=%s", ts.isoformat(), partition_granularity)
                    break
                except Exception:
                    continue
        if params:
            return self.conn.execute(sql, params).fetchdf()
        return self.conn.execute(sql).fetchdf()
