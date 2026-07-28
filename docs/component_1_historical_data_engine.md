# Component 1 — Historical Data Engine

## Purpose
Provide a reproducible, leak-free historical data engine for time-series and event data across domains (macroeconomic, financial, supply chain, SaaS metrics). Core features: data normalization, point-in-time (temporal) isolation for model validation, and efficient analytical storage using DuckDB + Parquet + Apache Arrow.

---

## Technical Specification

### High-level flow
1. Raw data ingestion (S3/MinIO, databases, APIs) -> landing zone (Parquet).
2. Normalization pipeline: resample, align to target frequency, impute missing values, unit standardization, metadata enrichment.
3. Persist normalized data as partitioned Parquet files (domain/entity_id/partition) and register in DuckDB as optimized analytical tables or views.
4. PointInTimeQuery interface: queries data with a strict cutoff_timestamp to prevent leakage.

### Storage schema (recommended)
- Files: Parquet files partitioned on: domain, year, month (or domain/entity_id/year/month) for efficient time-range pruning.
- DuckDB tables: use external Parquet files for large historical datasets and create lightweight views for common queries.

Suggested canonical table layout (logical):

- historical_timeseries(domain TEXT, entity_id TEXT, ts TIMESTAMP, metric TEXT, value DOUBLE, metadata JSON)

Notes:
- `domain` groups dataset types ("macro", "finance", "supply_chain", "saas").
- `entity_id` is the canonical identifier (company, sku, series id).
- `ts` is timezone-aware UTC timestamp.
- `metric` allows storing multiple metrics in a narrow table; optionally pivot to wide when necessary for models.
- `metadata` stores provenance, units, source_id as JSON (use Postgres JSONB for cataloging if you add a knowledge store).

Example DuckDB SQL (register Parquet files as view):

```sql
-- Use a path pattern for partitioned Parquet files
CREATE VIEW historical_timeseries AS
SELECT domain, entity_id, ts, metric, value, metadata
FROM read_parquet('s3://my-bucket/normalized/domain=*/year=*/month=*/data-*.parquet');
```

(If using local files: replace S3 path with `./data/normalized/domain=*/year=*/month=*/data-*.parquet`.)

### Point-in-Time semantics
- Every query used for training/validation must be filtered by `ts < cutoff_timestamp` (exclusive of cutoff) and, when needed, by `as_of` windows for feature aggregation.
- Support two modes:
  - Temporal isolation: global cutoff applied to all tables in a query (recommended).
  - Per-entity cutoff (advanced): different entities have different last-observed timestamps.

### Performance & engineering notes
- Use Parquet + Arrow for columnar, compressed storage and zero-copy transfers between DuckDB and Python (pyarrow arrays).
- For moderately sized production datasets, use DuckDB for ad-hoc analysis and testing; push-to-TimescaleDB/Postgres for high-concurrency OLTP/time-series needs.
- When implementing resampling, choose deterministic imputation (forward-fill/backward-fill or model-based with fixed random seeds) to ensure reproducible validation.

---

## API surface (Python sketches)

1) Data ingestion / normalizer (sketch)

```python
# docs/examples/normalizer.py (sketch)
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def normalize_timeseries(df: pd.DataFrame, target_freq: str = 'D') -> pd.DataFrame:
    """Resample, align, and standardize units on a single entity/metric dataframe.
    Expects `df` with columns: ['ts', 'value'] and a `meta` dict outside.
    """
    df = df.set_index('ts').sort_index()
    # resample to target frequency
    df_resampled = df['value'].resample(target_freq).mean()
    # simple imputation
    df_imputed = df_resampled.fillna(method='ffill').fillna(method='bfill')
    df_out = df_imputed.reset_index().rename(columns={'index': 'ts', 0: 'value'})
    return df_out

# persist to parquet partition path
```

2) PointInTimeQuery interface (sketch)

```python
# ckl/data/point_in_time.py (sketch)
from datetime import datetime
import duckdb

class PointInTimeQuery:
    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        self.conn = duckdb_conn

    def query_timeseries(self, cutoff_timestamp: datetime, domain: str, entity_ids: list[str], metric: str):
        # Enforce cutoff at SQL layer
        ids = ','.join([f"'{e}'" for e in entity_ids])
        sql = f"""
        SELECT domain, entity_id, ts, metric, value
        FROM historical_timeseries
        WHERE domain = '{domain}'
          AND entity_id IN ({ids})
          AND metric = '{metric}'
          AND ts < TIMESTAMP '{cutoff_timestamp.isoformat()}'
        ORDER BY entity_id, ts
        """
        return self.conn.execute(sql).fetch_df()
```

Important: Always pass `cutoff_timestamp` as a parameter at binding time to avoid string-injection and to document temporal isolation.

---

## Development Tasks (Epic 1)

Task 1.1: Set up DuckDB-backed time-series storage schema with domain partitioning
- Subtasks:
  - Define canonical schema for `historical_timeseries` and metadata catalog (metadata table mapping file paths, sources, ingestion time, units).
  - Implement Parquet-based partition layout and a helper to register partitions as DuckDB views.
  - Provide example SQL & scripts to attach S3/MinIO endpoints and credentials.
- Acceptance criteria:
  - Able to run a sample query over a partitioned dataset and prune partitions by domain/year/month.
  - Catalog table exists with ingestion provenance.
- Estimate: 1–2 days.

Task 1.2: Build data ingestion and normalization pipeline
- Subtasks:
  - Implement connectors for file, DB, and API sources (pluggable loader interface).
  - Implement normalization steps: timezone normalization, resampling, imputation strategies, unit conversions, metric canonicalization.
  - Write tests (unit + integration) using small sample CSVs and DuckDB.
- Acceptance criteria:
  - Pipeline can take raw CSV/JSON input and produce a partitioned Parquet output that matches the canonical schema.
  - Unit tests for resampling and imputation with deterministic behavior.
- Estimate: 3–5 days.

Task 1.3: Implement a PointInTimeQuery interface
- Subtasks:
  - Implement a Python class (duckdb-backed) that enforces global cutoff_timestamp and optionally `as_of` windows for feature aggregation.
  - Add query logging and explain plans for auditability.
  - Add tests ensuring queries with different cutoffs produce non-leaking results (e.g., synthetic examples where later values exist).
- Acceptance criteria:
  - API exists to request historical slices with cutoff and returns identical results to SQL queries with explicit cutoff filters.
  - Integration tests demonstrating no leakage.
- Estimate: 1–2 days.

---

## Implementation notes & recommendations
- Reproducibility: pin versions of pyarrow, duckdb, pandas; include example Dockerfile for local runs.
- Packaging: start with a minimal package `ckl.data` with modules `ingest`, `normalize`, `storage`, `point_in_time`.
- CI: add GitHub Actions job that runs unit tests and the DuckDB-based integration tests (using local files; mock S3 via localstack if needed).

---

## Next actions I can take now (choose one or I'll pick the first):
1. Scaffold a `ckl/data` package with skeleton modules and tests for Task 1.1–1.3.
2. Create an `ARCHITECTURE.md` or `docs/component_1_historical_data_engine.md` in the repo (this file has been added as a start).
3. Implement a working `PointInTimeQuery` class and a small integration test using DuckDB and example CSVs.

If you want (3), I will add code, tests, and a GitHub Actions workflow to run them.
