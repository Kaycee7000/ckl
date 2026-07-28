"""Attribution utilities for SCMs, including SHAP-based path attribution.

This module exposes a `shap_path_attribution` function which computes per-feature
attributions using SHAP. If `shap` is not installed, it raises an informative error.
"""
from typing import Callable, Sequence, Any, Dict


def shap_path_attribution(model_fn: Callable[[Sequence[float]], float], background: Sequence[Sequence[float]], samples: Sequence[float]) -> Dict[str, float]:
    """Compute SHAP values for `model_fn` at `samples` given `background` data.

    - `model_fn`: callable mapping feature vector -> scalar output
    - `background`: background dataset (list/array of feature vectors)
    - `samples`: a single feature vector to explain
    Returns a dict with keys 'shap_values' and 'expected_value'.
    """
    try:
        import shap
        import numpy as np
    except Exception:
        raise RuntimeError("shap library not installed. Install shap to use SHAP-based attribution.")

    # Use KernelExplainer for model-agnostic explanations
    explainer = shap.KernelExplainer(model_fn, background)
    shap_vals = explainer.shap_values(np.array(samples))
    expected = explainer.expected_value
    # Convert to a simple serializable dict
    return {"shap_values": list(shap_vals), "expected_value": float(expected)}
