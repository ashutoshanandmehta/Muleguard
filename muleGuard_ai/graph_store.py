from typing import List, Dict, Set
from .models import Edge, FeatureVector, Subgraph

class GraphStore:
    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: List[Edge] = []
        self._node_feats: Dict[str, FeatureVector] = {}

    def upsert(self, nodes: List[str], edges: List[Edge], node_features: Dict[str, FeatureVector]):
        self._nodes.update(nodes)
        self._edges.extend(edges)
        self._node_feats.update(node_features)

    def fetch_subgraph(self, focal: str, horizon_days: int) -> Subgraph:
        nbr_edges = [e for e in self._edges if e.src == focal or e.dst == focal]
        nbr_nodes = {focal}
        for e in nbr_edges:
            nbr_nodes.add(e.src)
            nbr_nodes.add(e.dst)
        feats = {n: self._node_feats.get(n, FeatureVector({})) for n in nbr_nodes}
        ordered_nodes = [focal] + sorted(n for n in nbr_nodes if n != focal)
        return Subgraph(nodes=ordered_nodes, edges=nbr_edges, node_features=feats, focal_node_id=focal)
