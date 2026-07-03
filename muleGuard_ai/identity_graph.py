from typing import List
from math import log1p
from .config import Config
from .models import EntityMap, EventData, EdgeFeatures, Edge

class GraphBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _w(self, amount: float, freq: float, recency: float, risk: float) -> float:
        p = self.cfg.edge_weight
        amt = log1p(max(amount, 0.0))
        return p.alpha * amt + p.beta * freq + p.gamma * recency + p.delta * risk

    def upsert(self, emap: EntityMap, event: EventData, efeats: EdgeFeatures) -> List[Edge]:
        edges: List[Edge] = []
        if event.kind == "transaction" and event.amount is not None and event.dst_account:
            edges.append(Edge(emap.account_id, event.dst_account, "txn",
                              self._w(event.amount, efeats.txn_freq, efeats.recency_decay, efeats.session_risk)))
        if event.kind == "login" and emap.device_id:
            edges.append(Edge(emap.account_id, emap.device_id, "login",
                              self._w(0.0, efeats.txn_freq, efeats.recency_decay, efeats.session_risk)))
        if emap.ip_id and emap.device_id:
            edges.append(Edge(emap.device_id, emap.ip_id, "device_use",
                              self._w(0.0, 0.0, efeats.recency_decay, efeats.device_entropy)))
        if emap.session_id:
            edges.append(Edge(emap.session_id, emap.account_id, "session_to_account",
                              self._w(0.0, 0.0, efeats.recency_decay, efeats.session_risk)))
        return edges
