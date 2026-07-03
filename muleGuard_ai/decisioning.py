from .config import Config

class DecisionEngine:
    def __init__(self, cfg: Config):
        t = cfg.thresholds
        if not (t.ALLOW_T <= t.STEP_UP_T <= t.HOLD_T <= t.BLOCK_T):
            raise ValueError("Decision thresholds must be ordered: ALLOW_T <= STEP_UP_T <= HOLD_T <= BLOCK_T")
        self.thresholds = t

    def decide(self, fused_score: float) -> str:
        t = self.thresholds
        if fused_score <= t.ALLOW_T:
            return "ALLOW"
        if fused_score <= t.STEP_UP_T:
            return "STEP_UP"
        if fused_score < t.BLOCK_T:
            return "HOLD"
        return "BLOCK"
