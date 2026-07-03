from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .graph_dataset import GraphDataset


@dataclass
class AccountRisk:
    account_id: str
    score: float
    contributors: Dict[str, float]
    evidence: List[str]


def _squash(value: float, scale: float) -> float:
    return float(1.0 - np.exp(-max(value, 0.0) / max(scale, 1e-6)))


def score_accounts(dataset: GraphDataset) -> List[AccountRisk]:
    risks: List[AccountRisk] = []
    feature_pos = {name: idx for idx, name in enumerate(dataset.feature_names)}

    for account_id in dataset.account_ids():
        fv = dataset.features[("account", account_id)]
        ts = float(fv[feature_pos["ts_anomaly"]])
        rule = float(fv[feature_pos["rule_uplift"]])
        velocity = _squash(float(fv[feature_pos["txn_velocity_24h"]]), 10.0)
        amount = _squash(float(fv[feature_pos["total_amount_log"]]), 25.0)
        out_degree = _squash(float(fv[feature_pos["out_degree"]]), 8.0)
        in_degree = _squash(float(fv[feature_pos["in_degree"]]), 8.0)
        graph_pressure = min(1.0, 0.4 * velocity + 0.25 * amount + 0.2 * out_degree + 0.15 * in_degree)
        score = min(1.0, 0.45 * graph_pressure + 0.35 * ts + 0.20 * rule)

        evidence = []
        for edge in dataset.edges:
            if edge.src_id == account_id or edge.dst_id == account_id:
                evidence.append(f"{edge.src_id} -[{edge.relation}]-> {edge.dst_id}")
            if len(evidence) >= 5:
                break

        risks.append(
            AccountRisk(
                account_id=account_id,
                score=round(score, 4),
                contributors={
                    "graph_pressure": round(graph_pressure, 4),
                    "ts_anomaly": round(ts, 4),
                    "rule_uplift": round(rule, 4),
                },
                evidence=evidence,
            )
        )

    return sorted(risks, key=lambda item: item.score, reverse=True)


def score_account_values(dataset: GraphDataset) -> Dict[str, float]:
    feature_pos = {name: idx for idx, name in enumerate(dataset.feature_names)}
    scores: Dict[str, float] = {}
    for account_id in dataset.account_ids():
        fv = dataset.features[("account", account_id)]
        ts = float(fv[feature_pos["ts_anomaly"]])
        rule = float(fv[feature_pos["rule_uplift"]])
        velocity = _squash(float(fv[feature_pos["txn_velocity_24h"]]), 10.0)
        amount = _squash(float(fv[feature_pos["total_amount_log"]]), 25.0)
        out_degree = _squash(float(fv[feature_pos["out_degree"]]), 8.0)
        in_degree = _squash(float(fv[feature_pos["in_degree"]]), 8.0)
        graph_pressure = min(1.0, 0.4 * velocity + 0.25 * amount + 0.2 * out_degree + 0.15 * in_degree)
        scores[account_id] = round(min(1.0, 0.45 * graph_pressure + 0.35 * ts + 0.20 * rule), 4)
    return scores
