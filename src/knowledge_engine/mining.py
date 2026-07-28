from typing import List, Dict, Tuple
import networkx as nx
from collections import Counter


def enumerate_connected_subgraphs(G: nx.Graph, k: int):
    # yield sorted tuples of nodes for each connected induced subgraph of size k
    nodes = list(G.nodes())
    for subset in __import__('itertools').combinations(nodes, k):
        H = G.subgraph(subset)
        if nx.is_connected(H):
            yield H


def canonical_graph_str(H: nx.Graph) -> str:
    # create an isomorphism-invariant signature using degree sequence and
    # sorted adjacency eigenvalues (rounded)
    import numpy as _np

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
