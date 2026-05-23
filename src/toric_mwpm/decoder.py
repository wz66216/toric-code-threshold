from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.matching import min_weight_matching

from .lattice import ToricChain, Vertex, shortest_path_chain


def toric_distance(a: Vertex, b: Vertex, L: int) -> int:
    dx_raw = abs(a[0] - b[0])
    dy_raw = abs(a[1] - b[1])
    dx = min(dx_raw, L - dx_raw)
    dy = min(dy_raw, L - dy_raw)
    return int(dx + dy)


def _defects_from_syndrome(syndrome: np.ndarray) -> list[Vertex]:
    array = np.asarray(syndrome, dtype=bool)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("syndrome must be a square two-dimensional array")
    xs, ys = np.nonzero(array)
    defects = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if len(defects) % 2 != 0:
        raise ValueError("a toric-code syndrome must contain an even number of defects")
    return defects


def decode_mwpm(syndrome: np.ndarray) -> ToricChain:
    """Decode a toric-code syndrome with minimum-weight perfect matching."""
    defects = _defects_from_syndrome(syndrome)
    L = int(np.asarray(syndrome).shape[0])
    if not defects:
        return ToricChain.empty(L)

    graph = nx.Graph()
    for index, defect in enumerate(defects):
        graph.add_node(index, defect=defect)

    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            graph.add_edge(i, j, weight=toric_distance(defects[i], defects[j], L))

    matching = min_weight_matching(graph, weight="weight")
    recovery = ToricChain.empty(L)
    for i, j in matching:
        path = shortest_path_chain(L, defects[i], defects[j])
        recovery = recovery + path

    return recovery
