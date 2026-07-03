import argparse
import copy
from pathlib import Path
import sys

import numpy as np

from .evaluate import _metrics
from .gnn_model import build_account_gnn, require_gnn_dependencies
from .graph_dataset import build_graph_dataset
from .pyg_adapter import to_pyg_heterodata
from .risk_baseline import score_account_values
from .tabular_teacher import TEACHER_MODELS, compute_teacher_logits


def _forward(model, data):
    if getattr(model, "uses_edge_attr", False):
        return model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)["account"]
    return model(data.x_dict, data.edge_index_dict)["account"]


def _forward_scores(model, data, teacher_alpha: float):
    """Forward pass with the tabular teacher added as a class-1 logit offset.

    ``out[:, 1] += alpha * teacher_logit`` anchors predictions at the tabular
    baseline so the GNN head only has to learn the graph residual.
    """
    out = _forward(model, data)
    teacher_logit = getattr(data["account"], "teacher_logit", None)
    if teacher_logit is not None and teacher_alpha != 0.0:
        out = out.clone()
        out[:, 1] = out[:, 1] + teacher_alpha * teacher_logit
    return out


def _fit_tabular_teacher(torch, dataset, data, args):
    """Fit the teacher on the train mask and attach per-account logits to ``data``."""
    node_ids = data["account"].node_ids
    x_all = np.array([dataset.features[("account", aid)] for aid in node_ids], dtype=np.float32)
    y_all = data["account"].y.detach().cpu().numpy()
    train_idx = data["account"].train_mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy()
    teacher_logit, payload = compute_teacher_logits(
        x_all,
        y_all,
        train_idx,
        model_type=args.teacher_model,
        folds=args.teacher_cv_folds,
        seed=args.seed,
        n_estimators=args.teacher_n_estimators,
        max_depth=args.teacher_max_depth,
    )
    data["account"].teacher_logit = torch.tensor(teacher_logit, dtype=torch.float)
    return payload


def _teacher_test_metrics(torch, data, top_k: int, cutoffs):
    teacher_logit = getattr(data["account"], "teacher_logit", None)
    if teacher_logit is None:
        return {}
    test_mask = data["account"].test_mask
    idxs = [idx for idx, keep in enumerate(test_mask.tolist()) if keep]
    if not idxs:
        return {}
    probs = torch.sigmoid(teacher_logit[test_mask]).detach().cpu().tolist()
    labels = [int(data["account"].y[idx]) for idx in idxs]
    return _metrics(labels, probs, threshold=0.5, top_k=min(top_k, len(labels)), cutoffs=cutoffs)


def _weighted_focal_loss(F, logits, labels, weight=None, gamma: float = 2.0):
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    pt = (-ce).exp()
    return (((1.0 - pt) ** gamma) * ce).mean()


def _pairwise_ranking_loss(torch, logits, labels, max_pairs: int):
    scores = logits[:, 1] - logits[:, 0]
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.sum() * 0.0
    pair_count = positive.numel() * negative.numel()
    if max_pairs > 0 and pair_count > max_pairs:
        pos_idx = torch.randint(positive.numel(), (max_pairs,), device=scores.device)
        neg_idx = torch.randint(negative.numel(), (max_pairs,), device=scores.device)
        diffs = positive[pos_idx] - negative[neg_idx]
    else:
        diffs = (positive[:, None] - negative[None, :]).reshape(-1)
    return torch.nn.functional.softplus(-diffs).mean()


def _score_metric(F, logits, labels, mask, metric: str, top_k: int, cutoffs):
    if int(mask.sum()) == 0:
        return 0.0, {}
    probs = F.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().tolist()
    y_true = labels[mask].detach().cpu().tolist()
    metrics = _metrics(y_true, probs, threshold=0.5, top_k=min(top_k, len(y_true)), cutoffs=cutoffs)
    return float(metrics.get(metric, 0.0) or 0.0), metrics


def _baseline_metrics(dataset, data, mask, top_k: int, cutoffs):
    baseline_scores = score_account_values(dataset)
    account_ids = data["account"].node_ids
    idxs = [idx for idx, keep in enumerate(mask.tolist()) if keep]
    if not idxs:
        return {}
    labels = [int(data["account"].y[idx]) for idx in idxs]
    scores = [baseline_scores.get(account_ids[idx], 0.0) for idx in idxs]
    return _metrics(labels, scores, threshold=0.5, top_k=min(top_k, len(labels)), cutoffs=cutoffs)


def _save_checkpoint(torch, payload: dict, output: Path) -> None:
    try:
        torch.save(payload, output)
    except OSError as exc:
        if "could not get source code" not in str(exc):
            raise
        torch.save(payload, output, _use_new_zipfile_serialization=False)


def _split_summary(dataset, data):
    account_ids = data["account"].node_ids

    def part(mask):
        idxs = [idx for idx, keep in enumerate(mask.tolist()) if keep]
        times = [dataset.account_first_seen.get(account_ids[idx], 0) for idx in idxs]
        positives = sum(int(data["account"].y[idx]) for idx in idxs)
        return {
            "accounts": len(idxs),
            "positives": positives,
            "negatives": len(idxs) - positives,
            "first_seen_min": min(times) if times else None,
            "first_seen_max": max(times) if times else None,
        }

    return {
        "split_strategy": getattr(data["account"], "split_strategy", "time"),
        "train": part(data["account"].train_mask),
        "validation": part(data["account"].val_mask),
        "test": part(data["account"].test_mask),
    }


def train(args: argparse.Namespace) -> dict:
    torch, F, *_ = require_gnn_dependencies()
    torch.manual_seed(args.seed)
    selection_metric = args.selection_metric or args.validation_metric

    dataset = build_graph_dataset(
        args.transactions,
        args.telemetry,
        args.entity_map,
        args.node_features,
    )
    data = to_pyg_heterodata(dataset, graph_view=args.graph_view)
    if int(data["account"].train_mask.sum()) == 0:
        raise RuntimeError("No labeled account nodes found. Add an `is_mule` column to node features.")
    teacher_payload = None
    if args.use_tabular_teacher:
        teacher_payload = _fit_tabular_teacher(torch, dataset, data, args)
    model = build_account_gnn(
        data.metadata(),
        hidden_channels=args.hidden_channels,
        out_channels=2,
        architecture=args.architecture,
        dropout=args.dropout,
        num_layers=args.layers,
        residual=args.residual,
        input_skip=args.input_skip,
        head_layers=args.head_layers,
    )

    with torch.no_grad():
        _forward(model, data)
    if args.use_tabular_teacher and args.teacher_zero_init_head:
        model.zero_init_head()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 2))
    class_weight = None
    if args.class_weighting == "balanced":
        labels = data["account"].y[data["account"].train_mask]
        counts = torch.bincount(labels, minlength=2).float()
        class_weight = torch.ones(2, dtype=torch.float)
        present = counts > 0
        class_weight[present] = counts[present].sum() / (present.sum() * counts[present])

    best_state = None
    best_epoch = 0
    best_score = float("-inf")
    best_metrics = {}
    bad_epochs = 0
    val_mask = data["account"].val_mask
    if int(val_mask.sum()) == 0:
        val_mask = data["account"].test_mask
    if int(val_mask.sum()) == 0:
        val_mask = data["account"].train_mask

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = _forward_scores(model, data, args.teacher_alpha)
        mask = data["account"].train_mask
        if args.loss == "focal":
            loss = _weighted_focal_loss(F, out[mask], data["account"].y[mask], class_weight, args.focal_gamma)
        else:
            loss = F.cross_entropy(out[mask], data["account"].y[mask], weight=class_weight)
        if args.ranking_loss_weight > 0:
            loss = loss + args.ranking_loss_weight * _pairwise_ranking_loss(
                torch,
                out[mask],
                data["account"].y[mask],
                args.ranking_max_pairs,
            )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_out = _forward_scores(model, data, args.teacher_alpha)
            val_score, val_metrics = _score_metric(
                F,
                eval_out,
                data["account"].y,
                val_mask,
                selection_metric,
                args.top_k,
                args.cutoffs,
            )
        scheduler.step(val_score)
        if val_score > best_score + args.min_delta:
            best_score = val_score
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            test_mask = data["account"].test_mask
            test_score, test_metrics = _score_metric(
                F,
                eval_out,
                data["account"].y,
                test_mask,
                selection_metric,
                args.top_k,
                args.cutoffs,
            )
            print(
                f"epoch={epoch} loss={loss.item():.4f} "
                f"val_{selection_metric}={val_score:.4f} "
                f"test_{selection_metric}={test_score:.4f}"
            )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_{selection_metric}={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_out = _forward_scores(model, data, args.teacher_alpha)
        _, final_test_metrics = _score_metric(
            F,
            final_out,
            data["account"].y,
            data["account"].test_mask,
            selection_metric,
            args.top_k,
            args.cutoffs,
        )
    baseline_test_metrics = _baseline_metrics(dataset, data, data["account"].test_mask, args.top_k, args.cutoffs)
    teacher_test_metrics = _teacher_test_metrics(torch, data, args.top_k, args.cutoffs)
    split_summary = _split_summary(dataset, data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        torch,
        {
            "model_state": model.state_dict(),
            "metadata": data.metadata(),
            "hyperparameters": {
                "hidden_channels": args.hidden_channels,
                "out_channels": 2,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "class_weighting": args.class_weighting,
                "architecture": args.architecture,
                "layers": args.layers,
                "dropout": args.dropout,
                "residual": args.residual,
                "input_skip": args.input_skip,
                "head_layers": args.head_layers,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma,
                "ranking_loss_weight": args.ranking_loss_weight,
                "ranking_max_pairs": args.ranking_max_pairs,
                "split_strategy": getattr(data["account"], "split_strategy", "time"),
                "graph_view": args.graph_view,
                "validation_metric": args.validation_metric,
                "selection_metric": selection_metric,
                "best_epoch": best_epoch,
                "best_selection_score": round(best_score, 6) if best_score != float("-inf") else 0.0,
                "use_tabular_teacher": bool(args.use_tabular_teacher),
                "teacher_alpha": args.teacher_alpha,
                "teacher_model": args.teacher_model,
                "teacher_cv_folds": args.teacher_cv_folds,
                "teacher_n_estimators": args.teacher_n_estimators,
                "teacher_max_depth": args.teacher_max_depth,
                "teacher_zero_init_head": bool(args.teacher_zero_init_head),
            },
            "feature_names": dataset.feature_names,
            "edge_attr_dim": 3,
            "tabular_teacher": teacher_payload,
            "best_selection_metrics": best_metrics,
            "best_validation_metrics": best_metrics,
            "test_metrics": final_test_metrics,
            "baseline_test_metrics": baseline_test_metrics,
            "teacher_test_metrics": teacher_test_metrics,
            "split": split_summary,
        },
        output,
    )
    print(f"saved checkpoint: {output} best_epoch={best_epoch} best_{selection_metric}={best_score:.4f}")
    return {
        "checkpoint": str(output),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_selection_score": round(best_score, 6) if best_score != float("-inf") else 0.0,
        "best_selection_metrics": best_metrics,
        "best_validation_score": round(best_score, 6) if best_score != float("-inf") else 0.0,
        "best_validation_metrics": best_metrics,
        "test_metrics": final_test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "teacher_test_metrics": teacher_test_metrics,
        "split": split_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MuleGuard's account-level heterogeneous GraphSAGE MVP.")
    parser.add_argument("--transactions", default="muleguard_core_transactions.csv")
    parser.add_argument("--telemetry", default="muleguard_digital_telemetry.csv")
    parser.add_argument("--entity-map", default="muleguard_entity_map_full.csv")
    parser.add_argument("--node-features", default="muleguard_node_features_full.csv")
    parser.add_argument("--output", default="models/account_graphsage.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--architecture", choices=["hetero_sage", "gatv2", "edge_transformer"], default="hetero_sage")
    parser.add_argument("--graph-view", choices=["full", "account_only", "transaction"], default="full")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--residual", dest="residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.add_argument("--input-skip", dest="input_skip", action="store_true", default=True)
    parser.add_argument("--no-input-skip", dest="input_skip", action="store_false")
    parser.add_argument("--head-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.1)
    parser.add_argument("--ranking-max-pairs", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--use-tabular-teacher", action="store_true", default=False)
    parser.add_argument("--teacher-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-model", choices=list(TEACHER_MODELS), default="numpy_logistic")
    parser.add_argument("--teacher-cv-folds", type=int, default=5)
    parser.add_argument("--teacher-n-estimators", type=int, default=200)
    parser.add_argument("--teacher-max-depth", type=int, default=None)
    parser.add_argument("--teacher-zero-init-head", dest="teacher_zero_init_head", action="store_true", default=True)
    parser.add_argument("--no-teacher-zero-init-head", dest="teacher_zero_init_head", action="store_false")
    parser.add_argument("--validation-metric", default="capture_at_5pct")
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cutoffs", default=[0.01, 0.02, 0.05], type=lambda value: [float(item.strip()) for item in value.split(",") if item.strip()])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        train(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
