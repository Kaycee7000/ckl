import pandas as pd
from datetime import datetime, timedelta
from historical_data_engine.storage import DuckDBStorage
from historical_data_engine.point_in_time import PointInTimeQuery


def setup_sample_storage():
    s = DuckDBStorage()
    s.create_table()
    now = datetime.utcnow()
    rows = []
    for i in range(5):
        rows.append({
            "domain": "test",
            "entity_id": "E1",
            "timestamp": now - timedelta(minutes=10 * i),
            "value": float(i),
            "metadata": None,
        })
    df = pd.DataFrame(rows)
    s.write_dataframe(df)
    return s, now


def test_fetch_respects_cutoff():
    storage, now = setup_sample_storage()
    pit = PointInTimeQuery(storage)
    cutoff = now - timedelta(minutes=15)
    df = pit.fetch("test", "E1", cutoff_timestamp=cutoff, lookback=10)
    assert not df.empty
    assert pd.to_datetime(df["timestamp"]).max() < pd.to_datetime(cutoff, utc=True)
