
from dataclasses import dataclass
from typing import Dict

@dataclass
class Thresholds:
    ALLOW_T: float      # set value
    STEP_UP_T: float    # set value
    HOLD_T: float       # set value
    BLOCK_T: float      # set value

@dataclass
class FusionWeights:
    w_gnn: float        # set value
    w_ts: float         # set value
    w_rule: float       # set value

@dataclass
class EdgeWeightParams:
    alpha: float        # set value
    beta: float        # set value
    gamma: float        # set value
    delta: float        # set value

@dataclass
class Config:
    thresholds: Thresholds
    fusion: FusionWeights
    edge_weight: EdgeWeightParams
    time_horizon_days: int    # set value
    model_parameters: Dict
    policies: Dict
