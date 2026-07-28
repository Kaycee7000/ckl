import pandas as pd
from datetime import datetime
from .storage import DuckDBStorage

class PointInTimeQuery:
    def __init__(self, storage: DuckDBStorage, table_name: str = "data"):
        self.storage = storage
        self.table = table_name

    def fetch(self, domain: str, entity_id: str, cutoff_timestamp: datetime, lookback: int = 100) -> pd.DataFrame:
        # Enforce strict cutoff: only rows with timestamp < cutoff
        sql = (
            f"SELECT * FROM {self.table} WHERE domain = ? AND entity_id = ? "
            f"AND timestamp < ? ORDER BY timestamp DESC LIMIT {lookback}"
        )
        df = self.storage.conn.execute(sql, [domain, entity_id, cutoff_timestamp]).fetchdf()
        # Validate none >= cutoff
        if not df.empty:
            max_ts = pd.to_datetime(df["timestamp"]).max()
            if max_ts >= pd.to_datetime(cutoff_timestamp):
                raise RuntimeError("Data leakage detected: row timestamp >= cutoff")
        return df
