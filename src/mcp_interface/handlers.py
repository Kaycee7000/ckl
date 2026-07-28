import json
import os
from jsonschema import validate, ValidationError
from typing import Dict, Any, List
from simulation_engine.scenario import ScenarioRunner
from knowledge_library.repository import ArtifactRepository
from validation_engine.backtester import WalkForwardBacktester
from causal_engine.scm import SCM
from knowledge_engine.confidence import confidence_from_backtests
import numpy as np

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")


def _load_schema(name: str) -> Dict:
    path = os.path.join(SCHEMA_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def simulate_scenario(request: Dict[str, Any], artifact_repo: ArtifactRepository = None) -> Dict[str, Any]:
    schema = _load_schema("simulate_scenario_request.json")
    validate(instance=request, schema=schema)
    params = request["params"]
    horizon = request["horizon"]
    # If artifact_id and artifact_repo provided, try to use artifact bias
    bias = 0.0
    if "artifact_id" in request and artifact_repo:
        art = artifact_repo.get_artifact(request["artifact_id"])
        if art:
            bias = art.get("structural_equations", {}).get("bias", 0.0)

    def model_fn(p):
        tot = float(np.sum(p)) + float(bias)
        t = np.linspace(0, 1, horizon)
        return (tot * t).tolist()

    runner = ScenarioRunner(model_fn)
    param_matrix = np.array([params])
    results = runner.run_scenarios(param_matrix, parallel=False)
    return {"trajectory": results.tolist()}


def explain_causal_chain(request: Dict[str, Any], artifact_repo: ArtifactRepository = None) -> Dict[str, Any]:
    schema = _load_schema("explain_causal_chain_request.json")
    validate(instance=request, schema=schema)
    aid = request["artifact_id"]
    if not artifact_repo:
        raise RuntimeError("artifact_repo required")
    art = artifact_repo.get_artifact(aid)
    if not art:
        return {"error": "artifact_not_found"}
    graph = art.get("causal_graph", {})
    # simple key drivers: use nodes with highest out-degree if present
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    outdeg = {}
    for n in nodes:
        outdeg[n] = sum(1 for e in edges if e[0] == n)
    sorted_nodes = sorted(outdeg.items(), key=lambda x: -x[1])
    key_drivers = [n for n, _ in sorted_nodes[:5]]
    # confidence: placeholder from artifact metadata
    confidence = art.get("metadata", {}).get("confidence", 0.5)
    return {"graph": graph, "key_drivers": key_drivers, "confidence": confidence}


def query_knowledge_library(request: Dict[str, Any], artifact_repo: ArtifactRepository) -> Dict[str, Any]:
    schema = _load_schema("query_knowledge_library_request.json")
    validate(instance=request, schema=schema)
    q = request["query"]
    top_k = request.get("top_k", 5)
    results = artifact_repo.search_by_preconditions(q, top_k=top_k)
    return {"results": results}


def validate_historical_analogue(request: Dict[str, Any], artifact_repo: ArtifactRepository) -> Dict[str, Any]:
    schema = _load_schema("validate_historical_analogue_request.json")
    validate(instance=request, schema=schema)
    aid = request["artifact_id"]
    series = np.array(request["series"])
    window = request.get("window_size", 50)
    art = artifact_repo.get_artifact(aid)
    if not art:
        return {"error": "artifact_not_found"}
    # create model from artifact bias if present
    bias = art.get("structural_equations", {}).get("bias", 0.0)

    def model_fn(train, horizon):
        last = float(train[-1])
        mean = np.array([last + bias for _ in range(horizon)])
        std = np.array([np.std(train) for _ in range(horizon)])
        return {"mean": mean, "std": std}

    backtester = WalkForwardBacktester(model_fn, window_size=window, horizon=1, step=10)
    results = backtester.run(series)
    # compute aggregate CRPS
    crps_vals = [v for r in results for v in r['crps']]
    mean_crps = float(np.mean(crps_vals)) if crps_vals else None
    return {"mean_crps": mean_crps, "n_folds": len(results)}
