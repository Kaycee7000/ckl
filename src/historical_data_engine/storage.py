import duckdb
import pandas as pd
from typing import Optional

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
        self.conn.register("_tmp_df", df)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM _tmp_df")

    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).fetchdf()
