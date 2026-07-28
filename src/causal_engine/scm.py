from typing import Callable, Dict, List, Optional, Any, Tuple
import copy
import numpy as np


class SCMNode:
    def __init__(self, name: str, parents: List[str], deterministic_fn: Callable[[Dict[str, Any]], float], noise_fn: Optional[Callable[[], float]] = None):
        self.name = name
        self.parents = parents
        self.deterministic_fn = deterministic_fn
        self.noise_fn = noise_fn


class SCM:
    """Simple Structural Causal Model with additive noise (value = f(parents) + U).

    deterministic_fn: callable(parents_dict) -> deterministic value (without noise)
    noise_fn: callable() -> noise sample (U)
    """

    def __init__(self):
        self.nodes: Dict[str, SCMNode] = {}

    def add_node(self, name: str, parents: List[str], deterministic_fn: Callable[[Dict[str, Any]], float], noise_fn: Optional[Callable[[], float]] = None):
        self.nodes[name] = SCMNode(name, parents, deterministic_fn, noise_fn)

    def topological_order(self) -> List[str]:
        # simple Kahn's algorithm
        deps = {n: set(node.parents) for n, node in self.nodes.items()}
        order: List[str] = []
        no_deps = [n for n, p in deps.items() if len(p) == 0]
        while no_deps:
            n = no_deps.pop()
            order.append(n)
            for m in list(deps.keys()):
                if n in deps[m]:
                    deps[m].remove(n)
                    if len(deps[m]) == 0:
                        no_deps.append(m)
        if len(order) != len(self.nodes):
            raise RuntimeError("Graph has cycles or missing parent references")
        return order

    def sample(self, interventions: Optional[Dict[str, float]] = None, n: int = 1) -> List[Dict[str, float]]:
        interventions = interventions or {}
        order = self.topological_order()
        samples = []
        for _ in range(n):
            vals: Dict[str, float] = {}
            for node in order:
                if node in interventions:
                    vals[node] = interventions[node]
                    continue
                node_obj = self.nodes[node]
                parent_vals = {p: vals[p] for p in node_obj.parents}
                det = node_obj.deterministic_fn(parent_vals)
                U = node_obj.noise_fn() if node_obj.noise_fn is not None else 0.0
                vals[node] = det + U
            samples.append(vals)
        return samples

    def do(self, interventions: Dict[str, float]) -> "SCM":
        new = copy.deepcopy(self)
        for var, val in interventions.items():
            # replace deterministic function with constant and remove noise
            new.nodes[var].deterministic_fn = lambda parents, _val=val: float(_val)
            new.nodes[var].noise_fn = lambda: 0.0
            new.nodes[var].parents = []
        return new

    def abduct_noise(self, observed: Dict[str, float]) -> Dict[str, float]:
        # assumes additive noise: observed[v] = f(parents) + U_v
        order = self.topological_order()
        inferred_u: Dict[str, float] = {}
        vals: Dict[str, float] = {}
        for node in order:
            node_obj = self.nodes[node]
            parent_vals = {p: vals[p] for p in node_obj.parents}
            det = node_obj.deterministic_fn(parent_vals)
            U = observed[node] - det
            inferred_u[node] = U
            vals[node] = observed[node]
        return inferred_u

    def counterfactual(self, observed: Dict[str, float], interventions: Dict[str, float], query: str) -> float:
        # abduct
        inferred_u = self.abduct_noise(observed)
        # copy and apply do
        scm_do = self.do(interventions)
        # evaluate forward deterministically using inferred_u
        order = scm_do.topological_order()
        vals: Dict[str, float] = {}
        for node in order:
            if node in interventions:
                vals[node] = interventions[node]
                continue
            node_obj = scm_do.nodes[node]
            parent_vals = {p: vals[p] for p in node_obj.parents}
            det = node_obj.deterministic_fn(parent_vals)
            U = inferred_u.get(node, 0.0)
            vals[node] = det + U
        return vals[query]

    def causal_attribution(self, upstream: str, target: str, trials: int = 1000, baseline: float = 0.0) -> float:
        # approximate attribution by sampling noises and measuring effect of setting upstream to baseline
        orig = self.sample(n=trials)
        # compute original expected target
        orig_vals = np.array([s[target] for s in orig])
        # for each sample, compute target with upstream set to baseline but same noises
        diffs = []
        order = self.topological_order()
        for s in orig:
            # s contains sampled values; abduct noises for this sample
            us = self.abduct_noise(s)
            # build do scm where upstream set to baseline
            scm_do = self.do({upstream: baseline})
            # forward evaluate using us
            vals: Dict[str, float] = {}
            for node in order:
                if node in scm_do.nodes and node == upstream:
                    vals[node] = baseline
                    continue
                node_obj = scm_do.nodes[node]
                parent_vals = {p: vals[p] for p in node_obj.parents}
                det = node_obj.deterministic_fn(parent_vals)
                U = us.get(node, 0.0)
                vals[node] = det + U
            diffs.append(s[target] - vals[target])
        return float(np.mean(diffs))
