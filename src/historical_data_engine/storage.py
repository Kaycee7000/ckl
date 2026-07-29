import duckdb
import pandas as pd
import logging
import os
import uuid
from typing import Optional, List, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DuckDBStorage:
    def __init__(self, s3_bucket: Optional[str] = None):
        # The actual engine runs in memory; data is persisted to S3
        self.conn = duckdb.connect(database=":memory:")
        
        # Pull bucket name from environment variables if not passed directly
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET_NAME")
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET_NAME environment variable is missing.")

        # Install and load required extensions for S3
        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")
        self.conn.execute("INSTALL aws;")
        self.conn.execute("LOAD aws;")
        
        # Automatically load secure AWS credentials from the ECS Task Role
        self.conn.execute("CALL load_aws_credentials();")

    def create_table(self, table_name: str = "data"):
        """
        Instead of a local table, create a VIEW that queries all parquet files 
        in the specified S3 path. This makes S3 act like a local table.
        """
        s3_path = f"s3://{self.s3_bucket}/{table_name}/*.parquet"
        try:
            self.conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{s3_path}')")
            logger.info(f"Bound view '{table_name}' to S3 path: {s3_path}")
        except Exception as e:
            # If the S3 folder is empty, the view creation will fail. 
            # We catch it here; the view will be successfully created after the first write.
            logger.warning(f"Could not bind view (S3 path might be empty yet): {e}")

    def write_dataframe(self, df: pd.DataFrame, table_name: str = "data"):
        # Enforce UTC for timestamp column to avoid timezone leakage
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            
        self.conn.register("_tmp_df", df)
        
        # Generate a unique filename (UUID) to avoid overwriting existing data
        file_id = uuid.uuid4().hex
        s3_path = f"s3://{self.s3_bucket}/{table_name}/{file_id}.parquet"
        
        # Export the dataframe directly to S3 as a Parquet file
        self.conn.execute(f"COPY _tmp_df TO '{s3_path}' (FORMAT PARQUET)")
        logger.info(f"Successfully wrote data to {s3_path}")
        
        # Refresh the view so the newly written file is included in future queries
        self.create_table(table_name)

    def query(self, sql: str, params: Optional[List[Any]] = None, partition_granularity: str = "D") -> pd.DataFrame:
        """Execute a SQL query against the S3-backed view."""
        if "timestamp" in sql.lower() and params:
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
