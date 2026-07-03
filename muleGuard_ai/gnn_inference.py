from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .graph_dataset import GraphDataset
from .gnn_model import build_account_gnn, require_gnn_dependencies
from .risk_baseline import AccountRisk
from .pyg_adapter import to_pyg_heterodata
from .tabular_teacher import apply_teacher_logits


@dataclass
class GNNInferenceResult:
    account_id: str
    score: float


def _load_checkpoint(torch, checkpoint_path: str):
    try:
        # Checkpoints may carry a pickled tabular teacher (numpy/sklearn objects).
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _teacher_offset(torch, checkpoint: dict, dataset: GraphDataset, node_ids: List[str]):
    teacher_payload = checkpoint.get("tabular_teacher")
    hyperparameters = checkpoint.get("hyperparameters", {})
    if not teacher_payload or not hyperparameters.get("use_tabular_teacher"):
        return None
    expected_features = checkpoint.get("feature_names")
    if expected_features and list(expected_features) != list(dataset.feature_names):
        raise RuntimeError(
            "Checkpoint tabular teacher feature names do not match the current dataset. "
            "Rebuild features with the same pipeline used at training time."
        )
    x_all = np.array([dataset.features[("account", aid)] for aid in node_ids], dtype=np.float32)
    alpha = float(hyperparameters.get("teacher_alpha", 1.0))
    return alpha * torch.tensor(apply_teacher_logits(teacher_payload, x_all), dtype=torch.float)


def score_accounts_with_checkpoint(dataset: GraphDataset, checkpoint_path: str) -> List[AccountRisk]:
    torch, F, *_ = require_gnn_dependencies()
    checkpoint = _load_checkpoint(torch, checkpoint_path)
    hyperparameters = checkpoint.get("hyperparameters", {})
    data = to_pyg_heterodata(dataset, graph_view=hyperparameters.get("graph_view", "full"))
    expected_metadata = checkpoint.get("metadata")
    if expected_metadata and expected_metadata != data.metadata():
        raise RuntimeError("Checkpoint graph metadata does not match the current dataset graph schema.")
    hidden_channels = int(hyperparameters.get("hidden_channels", 32))
    architecture = hyperparameters.get("architecture", "hetero_sage")
    layers = int(hyperparameters.get("layers", 2))
    dropout = float(hyperparameters.get("dropout", 0.0))
    residual = bool(hyperparameters.get("residual", False))
    input_skip = bool(hyperparameters.get("input_skip", True))
    head_layers = int(hyperparameters.get("head_layers", 1))
    model = build_account_gnn(
        data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=2,
        architecture=architecture,
        dropout=dropout,
        num_layers=layers,
        residual=residual,
        input_skip=input_skip,
        head_layers=head_layers,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    teacher_offset = _teacher_offset(torch, checkpoint, dataset, data["account"].node_ids)
    with torch.no_grad():
        if getattr(model, "uses_edge_attr", False):
            logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)["account"]
        else:
            logits = model(data.x_dict, data.edge_index_dict)["account"]
        if teacher_offset is not None:
            logits = logits.clone()
            logits[:, 1] = logits[:, 1] + teacher_offset
        probs = F.softmax(logits, dim=-1)[:, 1].tolist()

    baseline_features: Dict[str, Dict[str, float]] = {}
    feature_pos = {name: idx for idx, name in enumerate(dataset.feature_names)}
    for account_id in dataset.account_ids():
        fv = dataset.features[("account", account_id)]
        baseline_features[account_id] = {
            "ts_anomaly": round(float(fv[feature_pos["ts_anomaly"]]), 4),
            "rule_uplift": round(float(fv[feature_pos["rule_uplift"]]), 4),
        }

    risks: List[AccountRisk] = []
    for account_id, score in zip(dataset.account_ids(), probs):
        evidence = []
        for edge in dataset.edges:
            if edge.src_id == account_id or edge.dst_id == account_id:
                evidence.append(f"{edge.src_id} -[{edge.relation}]-> {edge.dst_id}")
            if len(evidence) >= 5:
                break
        risks.append(
            AccountRisk(
                account_id=account_id,
                score=round(float(score), 4),
                contributors={
                    "gnn_mule_probability": round(float(score), 4),
                    **baseline_features[account_id],
                },
                evidence=evidence,
            )
        )
    return sorted(risks, key=lambda item: item.score, reverse=True)
