import tempfile
from mcp_interface.handlers import simulate_scenario, explain_causal_chain, query_knowledge_library, validate_historical_analogue
from knowledge_library.repository import ArtifactRepository


def test_simulate_and_query_and_validate():
    with tempfile.TemporaryDirectory() as tmp:
        repo = ArtifactRepository(tmp)
        art = {"id": "a1", "name": "Test", "version": "v1.0.0", "preconditions": "price shock", "structural_equations": {"bias": -0.5}}
        repo.add_artifact(art)
        sim_req = {"domain": "test", "params": [1.0, 2.0], "horizon": 3, "artifact_id": "a1"}
        sim_out = simulate_scenario(sim_req, artifact_repo=repo)
        assert "trajectory" in sim_out

        explain_out = explain_causal_chain({"artifact_id": "a1"}, artifact_repo=repo)
        assert "confidence" in explain_out

        q_out = query_knowledge_library({"query": "price", "top_k": 1}, artifact_repo=repo)
        assert len(q_out["results"]) >= 1

        series = [i + 0.1 for i in range(200)]
        val_out = validate_historical_analogue({"artifact_id": "a1", "series": series, "window_size": 50}, artifact_repo=repo)
        assert "mean_crps" in val_out
