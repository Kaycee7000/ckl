from typing import List, Dict, Any
import numpy as np


class ResidualMonitor:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record(self, artifact_id: str, prediction_mean: float, truth: float, timestamp: Any = None):
        res = truth - prediction_mean
        self.records.append({"artifact_id": artifact_id, "pred": float(prediction_mean), "truth": float(truth), "residual": float(res), "ts": timestamp})

    def summary_for_artifact(self, artifact_id: str) -> Dict[str, Any]:
        recs = [r for r in self.records if r["artifact_id"] == artifact_id]
        if not recs:
            return {"count": 0}
        res = np.array([r["residual"] for r in recs])
        return {
            "count": int(len(res)),
            "mean_residual": float(np.mean(res)),
            "std_residual": float(np.std(res)),
        }
