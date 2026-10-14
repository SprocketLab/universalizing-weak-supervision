"""A small weighted digraph used to turn pairwise preferences into a ranking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Edge:
    """Directed edge ``u -> v`` (``u`` preferred to ``v``)."""

    def __init__(self, u, v, weight=0, confidence=0):
        self.u = u
        self.v = v
        self.weight = weight
        self.confidence = confidence

    def __hash__(self):
        return hash((self.u, self.v))

    def __str__(self):
        return f"{self.u}->{self.v}, {self.weight} @ {self.confidence}"

    __repr__ = __str__

    def __eq__(self, e2):
        return self.u == e2.u and self.v == e2.v


class RankingDiGraph:
    """Digraph over items; optionally initialized with the complete DAG of a ranking."""

    def __init__(self, ranking_perm=None, rng: Optional[np.random.Generator] = None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.in_edges = defaultdict(list)  # loser -> edges from winners
        self.out_edges = defaultdict(list)  # winner -> edges to losers
        if ranking_perm is None:
            self.nodes = []
        else:
            perm = list(ranking_perm)
            self.nodes = perm
            for i, u in enumerate(perm):
                for v in perm[i + 1 :]:
                    self.add_edge(u, v, 0)

    def add_edge(self, u, v, weight=0, confidence=0):
        e = Edge(u, v, weight, confidence)
        self.out_edges[u].append(e)
        self.in_edges[v].append(e)

    def print_edges(self):
        for u in self.nodes:
            for e in self.out_edges[u]:
                print(str(e))

    def remove_edge(self, e: Edge):
        self.out_edges[e.u].remove(e)
        self.in_edges[e.v].remove(e)

    def get_nodes_with_no_in_edges(self):
        return [n for n in self.nodes if len(self.in_edges[n]) == 0]

    def _shuffled(self, lst):
        return [lst[i] for i in self.rng.permutation(len(lst))]

    def topo_sort(self):
        """Greedy ordering: repeatedly take the heaviest remaining edge (ties: out-degree, confidence)."""
        L = []
        nodes_ranked = []
        candidate_edges = []
        for u in list(self.out_edges.keys()):
            for e in self.out_edges[u]:
                for x in (u, e.v):
                    if x not in nodes_ranked:
                        nodes_ranked.append(x)
                candidate_edges.append(e)

        out_edge_count = {u: len(self.out_edges[u]) for u in self.nodes}
        visited = set()
        masked_nodes = self._shuffled([v for v in self.nodes if v not in nodes_ranked])
        nodes_left = list(nodes_ranked)

        while len(L) < len(nodes_ranked) and candidate_edges:
            e = max(candidate_edges, key=lambda e: (out_edge_count[e.u], e.weight, e.confidence))
            candidate_edges.remove(e)
            for x in (e.u, e.v):
                if x not in visited:
                    L.append(x)
                    visited.add(x)
                    nodes_left.remove(x)
                    out_edge_count[x] = 1
                    break
        return L + self._shuffled(nodes_left) + masked_nodes

    def mask_node(self, u):
        """Remove every edge touching ``u``."""
        for e in list(self.out_edges[u]) + list(self.in_edges[u]):
            self.remove_edge(e)
