from typing import Optional, Dict, List
from .config import Config
from .models import (
    EventData, EntityMap, NodeFeatures, FeatureVector,
    DecisionOutcome
)
from .features import FeatureService
from .identity_graph import GraphBuilder
from .graph_store import GraphStore
from .ml_scoring import MLScoring
from .decisioning import DecisionEngine
from .analyst import AnalystInterface

class MuleGuardAI:
    def __init__(self, cfg: Config, store: GraphStore,
                 featsvc: FeatureService, builder: GraphBuilder,
                 scorer: MLScoring, decider: DecisionEngine, analyst: AnalystInterface):
        self.cfg = cfg
        self.store = store
        self.featsvc = featsvc
        self.builder = builder
        self.scorer = scorer
        self.decider = decider
        self.analyst = analyst

    def _node_feature_map(self, emap: EntityMap, node_feats: NodeFeatures) -> Dict[str, FeatureVector]:
        m: Dict[str, FeatureVector] = {emap.account_id: node_feats.account}
        if node_feats.customer and emap.customer_id: m[emap.customer_id] = node_feats.customer
        if node_feats.device and emap.device_id:     m[emap.device_id] = node_feats.device
        if node_feats.ip and emap.ip_id:             m[emap.ip_id] = node_feats.ip
        if node_feats.merchant and emap.merchant_id: m[emap.merchant_id] = node_feats.merchant
        return m

    def process(self, event: EventData, emap: EntityMap,
                node_feats: NodeFeatures, focal_node_id: str,
                focal_fv: FeatureVector) -> DecisionOutcome:

        efe = self.featsvc.compute_delta(event, node_feats)
        edges = self.builder.upsert(emap, event, efe)

        nodes = [n for n in [emap.account_id, emap.customer_id, emap.device_id,
                             emap.ip_id, emap.merchant_id, emap.session_id] if n]
        self.store.upsert(nodes, edges, self._node_feature_map(emap, node_feats))

        subg = self.store.fetch_subgraph(focal_node_id, self.cfg.time_horizon_days)
        gnn = self.scorer.gnn_score(subg)
        ts = self.scorer.timeseries_score(focal_fv)
        rule = self.scorer.rule_score(event, focal_fv, self.cfg.policies)
        fused = self.scorer.fuse(gnn, ts, rule)

        action = self.decider.decide(fused)
        explanation = self.analyst.build_explanation(gnn, ts, rule, subg)
        case_id = self.analyst.create_case(focal_node_id, fused, action, explanation, subg) if action in ("STEP_UP", "HOLD", "BLOCK") else None

        return DecisionOutcome(action=action, score=fused, case_id=case_id, explanation=explanation)
