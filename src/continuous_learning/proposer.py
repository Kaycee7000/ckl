from typing import Dict, Any


def propose_adjustment_from_residuals(artifact: Dict[str, Any], residual_summary: Dict[str, Any], threshold: float = 0.1) -> Dict[str, Any]:
    """Generate a simple proposal: adjust artifact bias if mean residual exceeds threshold."""
    mean_res = residual_summary.get("mean_residual", 0.0)
    if abs(mean_res) < threshold:
        return {"action": "none"}
    # propose to adjust a 'bias' parameter inside structural_equations if present
    se = artifact.get("structural_equations", {})
    bias = se.get("bias", 0.0)
    # propose changing bias to correct mean residual
    proposed_bias = bias + mean_res
    return {
        "action": "update_artifact",
        "artifact_id": artifact.get("id"),
        "change": {"structural_equations": {**se, "bias": proposed_bias}},
    }
