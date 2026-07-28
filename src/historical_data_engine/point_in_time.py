import pandas as pd
from datetime import datetime
from typing import Union
from .storage import DuckDBStorage

from dateutil import tz

class PointInTimeQuery:
    def __init__(self, storage: DuckDBStorage, table_name: str = "data"):
        self.storage = storage
        self.table = table_name

    def fetch(self, domain: str, entity_id: str, cutoff_timestamp: datetime, lookback: int = 100) -> pd.DataFrame:
        # Enforce strict cutoff: only rows with timestamp < cutoff
        # Normalize cutoff to UTC to avoid timezone mismatch
        if isinstance(cutoff_timestamp, str):
            cutoff_ts = pd.to_datetime(cutoff_timestamp, utc=True)
        else:
            cutoff_ts = pd.to_datetime(cutoff_timestamp, utc=True)

        sql = (
            f"SELECT * FROM {self.table} WHERE domain = ? AND entity_id = ? "
            f"AND timestamp < ? ORDER BY timestamp DESC LIMIT {lookback}"
        )
        # Use storage.query so partition-pruning logs can be emitted
        df = self.storage.query(sql, params=[domain, entity_id, cutoff_ts])
        # Ensure returned timestamps are timezone-aware UTC
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]) \
                .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")

        # Validate none >= cutoff
        if not df.empty:
            max_ts = pd.to_datetime(df["timestamp"]).max()
            if max_ts >= cutoff_ts:
                raise RuntimeError("Data leakage detected: row timestamp >= cutoff")
        return df
