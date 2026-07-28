from typing import Dict, Any
from validation_engine.backtester import WalkForwardBacktester
import copy


class Gatekeeper:
    def __init__(self, backtester: WalkForwardBacktester, series: list):
        self.backtester = backtester
        self.series = series

    def evaluate_proposal(self, artifact: Dict[str, Any], proposal: Dict[str, Any], model_factory) -> Dict[str, Any]:
        """Evaluate proposal by comparing backtest CRPS before and after applying change.

        model_factory(artifact) -> model_fn(train, horizon) expected by backtester
        """
        # baseline
        baseline_model = model_factory(artifact)
        baseline_results = self.backtester.run(self.series)
        baseline_crps = float(sum([sum(r['crps']) for r in baseline_results]) / max(1, sum(len(r['crps']) for r in baseline_results)))

        # apply change to a copy of artifact
        new_art = copy.deepcopy(artifact)
        change = proposal.get("change", {})
        # naive deep update
        for k, v in change.items():
            new_art[k] = v

        new_model = model_factory(new_art)
        new_results = self.backtester.run(self.series)
        new_crps = float(sum([sum(r['crps']) for r in new_results]) / max(1, sum(len(r['crps']) for r in new_results)))

        accept = new_crps <= baseline_crps
        return {"baseline_crps": baseline_crps, "new_crps": new_crps, "accept": accept}
