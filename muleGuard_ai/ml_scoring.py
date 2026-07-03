from .config import Config
from .models import Subgraph, FeatureVector, EventData

class MLScoring:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def gnn_score(self, subgraph: Subgraph) -> float:
        # Expect upstream to provide 'gnn_score' in node features if available; otherwise 0.0
        focal = subgraph.focal_node_id or subgraph.nodes[0]
        fv = subgraph.node_features.get(focal)
        return fv.values.get("gnn_score", 0.0) if fv else 0.0

    def timeseries_score(self, fv: FeatureVector) -> float:
        return fv.values.get("ts_anomaly", 0.0)

    def rule_score(self, event: EventData, fv: FeatureVector, policies: dict) -> float:
        return fv.values.get("rule_uplift", 0.0)

    def fuse(self, gnn: float, ts: float, rule: float) -> float:
        w = self.cfg.fusion
        s = max(w.w_gnn + w.w_ts + w.w_rule, 1e-12)
        return (w.w_gnn * gnn + w.w_ts * ts + w.w_rule * rule) / s
