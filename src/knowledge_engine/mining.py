from typing import List, Dict, Tuple
import networkx as nx
from collections import Counter
import itertools
import numpy as _np


def enumerate_connected_subgraphs(G: nx.Graph, k: int):
    # yield sorted tuples of nodes for each connected induced subgraph of size k
    nodes = list(G.nodes())
    for subset in __import__('itertools').combinations(nodes, k):
        H = G.subgraph(subset)
        if nx.is_connected(H):
            yield H


def canonical_graph_str(H: nx.Graph) -> str:
    """Return a canonical string for graph H.

    Strategy (in order):
    - For small graphs (<=8 nodes), compute canonical adjacency by enumerating
      all node order permutations and taking the lexicographically smallest
      adjacency bitstring (exact canonical labeling without external deps).
    - Otherwise, fall back to a spectral+degree signature as a graceful fallback.
    """
    n = H.number_of_nodes()
    # exact canonical labeling by permutation for small graphs
    if n <= 8:
        nodes = list(H.nodes())
        best = None
        for perm in itertools.permutations(range(n)):
            order = [nodes[i] for i in perm]
            A = nx.to_numpy_array(H, nodelist=order).astype(int)
            s = ''.join(map(str, A.flatten().tolist()))
            if best is None or s < best:
                best = s
        return f"canon_adj:{best}"

    # fallback: degree sequence + adjacency eigenvalues
    deg = sorted([d for _, d in H.degree()])
    A = nx.to_numpy_array(H)
    try:
        eigs = sorted([float(round(x.real, 6)) for x in _np.linalg.eigvals(A)])
    except Exception:
        eigs = []
    return f"deg:{deg}|eigs:{eigs}"


def frequent_subgraphs(graphs: List[nx.Graph], k: int = 3, min_support: int = 2) -> Dict[str, int]:
    """Find frequent connected induced subgraphs of size k across a list of graphs.
    Returns mapping from canonical signature to count (support).
    """
    counter = Counter()
    for G in graphs:
        seen = set()
        for H in enumerate_connected_subgraphs(G, k):
            sig = canonical_graph_str(H)
            if sig not in seen:
                counter[sig] += 1
                seen.add(sig)
    # filter by min_support
    return {s: c for s, c in counter.items() if c >= min_support}
