import tempfile
import os
from knowledge_library.repository import ArtifactRepository


def sample_artifact(aid: str, name: str, version: str, preconditions: str):
    return {
        "id": aid,
        "name": name,
        "version": version,
        "preconditions": preconditions,
        "structural_equations": {"x": "2*y + u"}
    }


def test_artifact_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        repo = ArtifactRepository(tmp)
        a1 = sample_artifact("art1", "PriceShock", "v1.0.0", "price increase leads to demand drop")
        a2 = sample_artifact("art2", "DemandElasticity", "v1.0.0", "price elasticity affects demand")
        repo.add_artifact(a1)
        repo.add_artifact(a2)
        results = repo.search_by_preconditions("price increase", top_k=2)
        assert len(results) >= 1
        assert any(r['artifact']['id'] == 'art1' for r in results)
