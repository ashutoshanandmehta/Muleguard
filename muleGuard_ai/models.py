from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass(frozen=True)
class EventData:
    kind: str
    timestamp: int
    amount: Optional[float]
    src_account: Optional[str]
    dst_account: Optional[str]
    merchant_id: Optional[str]
    device_id: Optional[str]
    ip_id: Optional[str]
    session_id: Optional[str]
    channel: Optional[str]

@dataclass(frozen=True)
class EntityMap:
    account_id: str
    customer_id: str
    device_id: Optional[str]
    ip_id: Optional[str]
    merchant_id: Optional[str]
    session_id: Optional[str]

@dataclass
class FeatureVector:
    values: Dict[str, float]

@dataclass
class NodeFeatures:
    account: FeatureVector
    customer: Optional[FeatureVector]
    device: Optional[FeatureVector]
    ip: Optional[FeatureVector]
    merchant: Optional[FeatureVector]

@dataclass
class EdgeFeatures:
    txn_freq: float
    recency_decay: float
    device_entropy: float
    session_risk: float

@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    weight: float

@dataclass
class Subgraph:
    nodes: List[str]
    edges: List[Edge]
    node_features: Dict[str, FeatureVector]
    focal_node_id: Optional[str] = None

@dataclass
class DecisionOutcome:
    action: str
    score: float
    case_id: Optional[str]
    explanation: Dict[str, Any]
