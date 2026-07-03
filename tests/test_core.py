import unittest

from muleGuard_ai.config import Config, EdgeWeightParams, FusionWeights, Thresholds
from muleGuard_ai.decisioning import DecisionEngine
from muleGuard_ai.graph_store import GraphStore
from muleGuard_ai.ml_scoring import MLScoring
from muleGuard_ai.models import Edge, FeatureVector


def cfg() -> Config:
    return Config(
        thresholds=Thresholds(ALLOW_T=0.2, STEP_UP_T=0.5, HOLD_T=0.7, BLOCK_T=0.9),
        fusion=FusionWeights(w_gnn=0.6, w_ts=0.3, w_rule=0.1),
        edge_weight=EdgeWeightParams(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1),
        time_horizon_days=30,
        model_parameters={},
        policies={},
    )


class CoreBehaviorTest(unittest.TestCase):
    def test_gnn_score_uses_focal_node(self):
        store = GraphStore()
        store.upsert(
            ["ACC123", "ACC987"],
            [Edge("ACC123", "ACC987", "txn", 1.0)],
            {
                "ACC123": FeatureVector({"gnn_score": 0.88}),
                "ACC987": FeatureVector({}),
            },
        )
        subgraph = store.fetch_subgraph("ACC123", 30)
        self.assertEqual(subgraph.nodes[0], "ACC123")
        self.assertEqual(MLScoring(cfg()).gnn_score(subgraph), 0.88)

    def test_decision_thresholds_keep_hold_until_block(self):
        decider = DecisionEngine(cfg())
        self.assertEqual(decider.decide(0.19), "ALLOW")
        self.assertEqual(decider.decide(0.21), "STEP_UP")
        self.assertEqual(decider.decide(0.71), "HOLD")
        self.assertEqual(decider.decide(0.89), "HOLD")
        self.assertEqual(decider.decide(0.90), "BLOCK")


if __name__ == "__main__":
    unittest.main()
