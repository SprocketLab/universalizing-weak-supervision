from collections import defaultdict
import random
import copy
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Edge:
    def __init__(self, u, v, weight=0, confidence=0):
        """
        Edge object
        Parameters
        ----------
        u: src node
        v: dst node
        weight
        confidence
        """
        # u --> v
        self.u = u
        self.v = v
        self.weight = weight
        self.confidence = confidence

    def __hash__(self):
        x = '{}->{}'.format(self.u,self.v)
        return hash(x)

    def __str__(self):
        return '{}->{}, {} @ {}'.format(self.u,self.v,self.weight,self.confidence)

    def __eq__(self, e2):
        return self.u == e2.u and self.v == e2.v


class RankingDiGraph:
    def __init__(self, ranking_perm=None):
        """
        Initialize digraph (
        Parameters
        ----------
        ranking_perm: Ranking object
        """
        if not ranking_perm is None:
            self.nodes = ranking_perm
            self.in_edges = defaultdict(list)
            self.out_edges = defaultdict(list)

            n = len(ranking_perm)

            for i, u in enumerate(ranking_perm):
                for j in range(i+1, n):
                    self.add_edge(Edge(u, ranking_perm[j]))
        else:
            logging.info('ranking perm is None')
            self.nodes = []
            self.in_edges = defaultdict(list)  # index: loser node, the value: win node
            self.out_edges = defaultdict(list) # index: win node, the value: loser node

    def add_edge(self, u, v, weight, confidence=0):
        """
        add edges u -> v (self.in_edges and self.out_edges)
        Parameters
        ----------
        u
        v
        weight
        confidence

        Returns
        -------

        """
        self.out_edges[u].append(Edge(u, v, weight, confidence))
        self.in_edges[v].append(Edge(u, v, weight, confidence))

    def print_edges(self):
        """
        Print edges
        Returns
        -------

        """
        for u in self.nodes:
            for e in self.out_edges[u]:
                print(str(e))

    def remove_edge(self,e):
        """
        remove edge e from self.our_edges, self.in_edges
        Parameters
        ----------
        e

        Returns
        -------

        """

        self.out_edges[e.u].remove(e)
        self.in_edges[e.v].remove(e)

    def get_nodes_with_no_in_edges(self):
        """
        get nodes with no in edges
        Returns
        -------

        """
        S = []
        for n in self.nodes:
            if len(self.in_edges[n]) == 0:
                S.append(n)
        return S

    def topo_sort(self):
        """
        topological sort,
        Returns
        -------

        """
        L = []
        nodes_ranked = set()
        candidate_edges = []

        # organize nodes_ranked, candidate_edges
        # assumption: winning items com first (?)
        for u in self.out_edges.keys():
            if len(self.out_edges[u]) > 0:
                nodes_ranked.add(u)
                for e in self.out_edges[u]:
                    nodes_ranked.add(e.v)
                    candidate_edges.append(e)

        nodes_ranked = list(nodes_ranked)
        logger.debug("nodes_ranked {}".format(nodes_ranked))

        out_edge_count = dict([(u,len(self.out_edges[u])) for u in self.nodes])

        # initialize visited count
        visited = [0 for i in self.nodes]

        # mask unranked nodes
        masked_nodes = []
        for v in self.nodes:
            if v not in nodes_ranked:
                masked_nodes.append(v)

        random.shuffle(masked_nodes)
        nodes_left = copy.deepcopy(nodes_ranked)

        n = len(nodes_ranked)
        k = 0
        while k < n and len(candidate_edges) > 0:
            # pick edge with max weight, if tie use confidence
            e = max([(e, out_edge_count[e.u], e.weight, e.confidence)
                     for e in candidate_edges], key=lambda x:[x[1], x[2], x[3]])[0]
            logger.debug('selected {}'.format(e))

            # mark unvisited nodes in edge e (u->v)
            candidate_edges.remove(e)
            if not visited[e.u]:
                L.append(e.u)
                visited[e.u] = 1
                nodes_left.remove(e.u)
                out_edge_count[e.u] = 1
                k += 1

            elif not visited[e.v]:
                L.append(e.v)
                visited[e.v] = 1
                nodes_left.remove(e.v)
                out_edge_count[e.v] = 1
                k += 1

        random.shuffle(nodes_left)

        L = L + nodes_left + masked_nodes

        return L

    def mask_node(self, u):
        """
        remove maskes related to the node u
        Parameters
        ----------
        u

        Returns
        -------

        """
        edges_to_remove = []
        for v in self.out_edges[u]:
            edges_to_remove.append((u, v))
        for w in self.in_edges[u]:
            edges_to_remove.append((w, u))
        for e in edges_to_remove:
            self.remove_edge(Edge(e[0], e[1]))
