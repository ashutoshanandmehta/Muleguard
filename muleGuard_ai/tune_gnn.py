import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from .evaluate import _parse_cutoffs
from .train_gnn import build_parser as build_train_parser, train


DEFAULT_GRID = {
    "architecture": ["hetero_sage", "gatv2", "edge_transformer"],
    "graph_view": ["full"],
    "loss": ["cross_entropy", "focal"],
    "hidden_channels": [16, 32, 64],
    "layers": [2, 3],
    "dropout": [0.2, 0.4],
    "lr": [0.001, 0.005, 0.01],
}

SMOKE_GRID = {
    "architecture": ["hetero_sage", "gatv2"],
    "graph_view": ["full"],
    "loss": ["cross_entropy"],
    "hidden_channels": [8],
    "layers": [1],
    "dropout": [0.2],
    "lr": [0.01],
}


@dataclass(frozen=True)
class GNNConfig:
    architecture: str
    graph_view: str
    loss: str
    hidden_channels: int
    layers: int
    dropout: float
    lr: float
    focal_gamma: float
    ranking_loss_weight: float

    def key(self) -> str:
        return (
            f"{self.architecture}__{self.graph_view}__{self.loss}__h{self.hidden_channels}"
            f"__l{self.layers}__d{str(self.dropout).replace('.', '_')}"
            f"__lr{str(self.lr).replace('.', '_')}"
            f"__fg{str(self.focal_gamma).replace('.', '_')}"
            f"__rw{str(self.ranking_loss_weight).replace('.', '_')}"
        )

    def as_dict(self) -> Dict:
        return {
            "architecture": self.architecture,
            "graph_view": self.graph_view,
            "loss": self.loss,
            "hidden_channels": self.hidden_channels,
            "layers": self.layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "focal_gamma": self.focal_gamma,
            "ranking_loss_weight": self.ranking_loss_weight,
        }


def expand_grid(smoke: bool = False) -> List[GNNConfig]:
    grid = SMOKE_GRID if smoke else DEFAULT_GRID
    return [
        GNNConfig(
            architecture=architecture,
            graph_view=graph_view,
            loss=loss,
            hidden_channels=hidden_channels,
            layers=layers,
            dropout=dropout,
            lr=lr,
            focal_gamma=2.0,
            ranking_loss_weight=0.1,
        )
        for architecture, graph_view, loss, hidden_channels, layers, dropout, lr in product(
            grid["architecture"],
            grid["graph_view"],
            grid["loss"],
            grid["hidden_channels"],
            grid["layers"],
            grid["dropout"],
            grid["lr"],
        )
    ]


def _parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def filter_grid(
    configs: List[GNNConfig],
    architectures: Optional[List[str]] = None,
    graph_views: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    hidden_channels: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    dropouts: Optional[List[float]] = None,
    learning_rates: Optional[List[float]] = None,
    focal_gammas: Optional[List[float]] = None,
    ranking_loss_weights: Optional[List[float]] = None,
) -> List[GNNConfig]:
    def keep(config: GNNConfig) -> bool:
        if architectures and config.architecture not in architectures:
            return False
        if graph_views and config.graph_view not in graph_views:
            return False
        if losses and config.loss not in losses:
            return False
        if hidden_channels and config.hidden_channels not in hidden_channels:
            return False
        if layers and config.layers not in layers:
            return False
        if dropouts and config.dropout not in dropouts:
            return False
        if learning_rates and config.lr not in learning_rates:
            return False
        if focal_gammas and config.focal_gamma not in focal_gammas:
            return False
        if ranking_loss_weights and config.ranking_loss_weight not in ranking_loss_weights:
            return False
        return True

    return [config for config in configs if keep(config)]


def expand_training_grids(configs: List[GNNConfig], focal_gammas: Optional[List[float]], ranking_loss_weights: Optional[List[float]]) -> List[GNNConfig]:
    gammas = focal_gammas or sorted({config.focal_gamma for config in configs})
    weights = ranking_loss_weights or sorted({config.ranking_loss_weight for config in configs})
    return [
        GNNConfig(
            architecture=config.architecture,
            graph_view=config.graph_view,
            loss=config.loss,
            hidden_channels=config.hidden_channels,
            layers=config.layers,
            dropout=config.dropout,
            lr=config.lr,
            focal_gamma=gamma,
            ranking_loss_weight=ranking_weight,
        )
        for config in configs
        for gamma in gammas
        for ranking_weight in weights
    ]


def expand_graph_views(configs: List[GNNConfig], graph_views: Optional[List[str]]) -> List[GNNConfig]:
    views = graph_views or sorted({config.graph_view for config in configs})
    return [
        GNNConfig(
            architecture=config.architecture,
            graph_view=view,
            loss=config.loss,
            hidden_channels=config.hidden_channels,
            layers=config.layers,
            dropout=config.dropout,
            lr=config.lr,
            focal_gamma=config.focal_gamma,
            ranking_loss_weight=config.ranking_loss_weight,
        )
        for config in configs
        for view in views
    ]


def _data_paths(data_dir: str) -> Dict[str, str]:
    data = Path(data_dir)
    return {
        "transactions": str(data / "muleguard_core_transactions.csv"),
        "telemetry": str(data / "muleguard_digital_telemetry.csv"),
        "entity_map": str(data / "muleguard_entity_map_full.csv"),
        "node_features": str(data / "muleguard_node_features_full.csv"),
    }


def _train_args(args: argparse.Namespace, config: GNNConfig, seed: int, checkpoint: Path) -> argparse.Namespace:
    parser = build_train_parser()
    paths = _data_paths(args.data)
    cli_args = [
        "--transactions", paths["transactions"],
        "--telemetry", paths["telemetry"],
        "--entity-map", paths["entity_map"],
        "--node-features", paths["node_features"],
        "--output", str(checkpoint),
        "--epochs", str(args.epochs),
        "--hidden-channels", str(config.hidden_channels),
        "--architecture", config.architecture,
        "--graph-view", config.graph_view,
        "--layers", str(config.layers),
        "--dropout", str(config.dropout),
        "--input-skip" if args.input_skip else "--no-input-skip",
        "--head-layers", str(args.head_layers),
        "--lr", str(config.lr),
        "--weight-decay", str(args.weight_decay),
        "--log-every", str(args.log_every),
        "--seed", str(seed),
        "--class-weighting", "balanced",
        "--loss", config.loss,
        "--focal-gamma", str(config.focal_gamma),
        "--ranking-loss-weight", str(config.ranking_loss_weight),
        "--ranking-max-pairs", str(args.ranking_max_pairs),
        "--grad-clip", str(args.grad_clip),
        "--patience", str(args.patience),
        "--validation-metric", args.metric,
        "--selection-metric", args.selection_metric or args.metric,
        "--top-k", str(args.top_k),
        "--cutoffs", ",".join(str(cutoff) for cutoff in args.cutoffs),
    ]
    if args.use_tabular_teacher:
        cli_args.extend([
            "--use-tabular-teacher",
            "--teacher-alpha", str(args.teacher_alpha),
            "--teacher-model", args.teacher_model,
            "--teacher-cv-folds", str(args.teacher_cv_folds),
        ])
    return parser.parse_args(cli_args)


def valid_metric_names(cutoffs: Iterable[float]) -> List[str]:
    names = [
        "precision",
        "recall",
        "f1",
        "top_k_recall",
        "pr_auc",
        "ks",
    ]
    for cutoff in cutoffs:
        pct = cutoff * 100.0
        if pct.is_integer():
            label = f"{int(pct)}pct"
        else:
            label = f"{str(round(pct, 3)).replace('.', '_')}pct"
        names.extend([
            f"capture_at_{label}",
            f"precision_at_{label}",
            f"lift_at_{label}",
        ])
    return names


def _summarize(runs: List[Dict], metric: str) -> Dict:
    values = [float(run["gnn_metrics"].get(metric, 0.0) or 0.0) for run in runs]
    best_run = max(runs, key=lambda run: float(run["gnn_metrics"].get(metric, 0.0) or 0.0))
    return {
        "runs": len(runs),
        "metric": metric,
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "best": round(float(max(values)), 6),
        "best_seed": best_run["seed"],
        "best_checkpoint": best_run["checkpoint"],
        "best_metrics": best_run["gnn_metrics"],
    }


def _progress_line(completed: int, total: int, start_time: float, status: str) -> str:
    width = 24
    ratio = completed / total if total else 1.0
    filled = min(width, int(round(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    return f"[{bar}] {completed}/{total} elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m {status}"


def _load_tabular_baseline(path: Optional[str], metric: str) -> Dict:
    if not path or not Path(path).exists():
        return {"available": False, "metric": metric, "score": None, "path": path}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    score = payload.get("best_score")
    if score is None:
        score = payload.get("metrics", {}).get(metric)
    return {
        "available": score is not None,
        "metric": metric,
        "score": float(score) if score is not None else None,
        "path": path,
        "model": payload.get("best_model") or payload.get("metadata", {}).get("model"),
        "score_basis": payload.get("best_score_basis", "single_run"),
    }


def promotion_decision(best_gnn_score: float, tabular_score: Optional[float], margin: float, target: float, target_lift: float, best_metrics: Dict) -> str:
    if tabular_score is None:
        return "NEEDS_MORE_DATA"
    lift = float(best_metrics.get("lift_at_5pct", 0.0) or 0.0)
    if best_gnn_score >= tabular_score + margin and best_gnn_score >= target and lift >= target_lift:
        return "PROMOTE_GNN"
    if best_gnn_score >= tabular_score + margin:
        return "NEEDS_MORE_DATA"
    return "KEEP_TABULAR"


def _build_report(args: argparse.Namespace, configs: List[GNNConfig], candidates: Dict, best_config_key: Optional[str], best_score: float, best_summary: Optional[Dict], tabular: Dict, warnings: List[str], status: str) -> Dict:
    decision = "IN_PROGRESS"
    if best_summary is not None and status == "complete":
        decision = promotion_decision(
            best_score,
            tabular["score"],
            args.promotion_margin,
            args.target_capture_at_5pct,
            args.target_lift_at_5pct,
            best_summary["best_metrics"],
        )
    return {
        "status": status,
        "data_dir": args.data,
        "metric": args.metric,
        "selection_metric": args.selection_metric or args.metric,
        "input_skip": args.input_skip,
        "head_layers": args.head_layers,
        "cutoffs": args.cutoffs,
        "smoke": args.smoke,
        "grid_filters": {
            "architectures": args.architectures,
            "graph_views": args.graph_views,
            "losses": args.losses,
            "hidden_channels": args.hidden_channels_grid,
            "layers": args.layers_grid,
            "dropouts": args.dropout_grid,
            "learning_rates": args.lr_grid,
            "focal_gammas": args.focal_gamma_grid,
            "ranking_loss_weights": args.ranking_loss_weight_grid,
        },
        "tabular_teacher": {
            "enabled": args.use_tabular_teacher,
            "alpha": args.teacher_alpha,
            "model": args.teacher_model,
            "cv_folds": args.teacher_cv_folds,
        },
        "runs_per_config": args.runs,
        "config_count": len(configs),
        "completed_config_count": len(candidates),
        "best_config": best_config_key,
        "best_score": round(best_score, 6) if best_summary is not None else None,
        "best_score_basis": "mean_across_runs",
        "metric_scope": "held_out_time_test",
        "best_summary": best_summary,
        "tabular_baseline": tabular,
        "promotion_margin": args.promotion_margin,
        "ranking_loss_weight": args.ranking_loss_weight,
        "focal_gamma_grid": args.focal_gamma_grid,
        "ranking_loss_weight_grid": args.ranking_loss_weight_grid,
        "ranking_max_pairs": args.ranking_max_pairs,
        "target_capture_at_5pct": args.target_capture_at_5pct,
        "target_lift_at_5pct": args.target_lift_at_5pct,
        "promotion_decision": decision,
        "candidates": candidates,
        "warnings": warnings,
    }


def _write_report(path: str, payload: Dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _restore_candidates(path: str, metric: str) -> tuple:
    report_path = Path(path)
    if not report_path.exists():
        return {}, None, float("-inf"), None, []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", {})
    warnings = payload.get("warnings", [])
    best_key = None
    best_score = float("-inf")
    best_summary = None
    for key, candidate in candidates.items():
        summary = candidate.get("summary", {})
        score = float(summary.get("mean", 0.0) or 0.0)
        if summary.get("metric") == metric and score > best_score:
            best_key = key
            best_score = score
            best_summary = summary
    return candidates, best_key, best_score, best_summary, warnings


def tune(args: argparse.Namespace) -> Dict:
    if args.metric not in valid_metric_names(args.cutoffs):
        allowed = ", ".join(valid_metric_names(args.cutoffs))
        raise RuntimeError(f"Invalid metric `{args.metric}`. Choose one of: {allowed}")
    if args.selection_metric and args.selection_metric not in valid_metric_names(args.cutoffs):
        allowed = ", ".join(valid_metric_names(args.cutoffs))
        raise RuntimeError(f"Invalid selection metric `{args.selection_metric}`. Choose one of: {allowed}")
    configs = expand_grid(args.smoke)
    configs = expand_graph_views(configs, args.graph_views)
    configs = expand_training_grids(configs, args.focal_gamma_grid, args.ranking_loss_weight_grid)
    configs = filter_grid(
        configs,
        architectures=args.architectures,
        graph_views=args.graph_views,
        losses=args.losses,
        hidden_channels=args.hidden_channels_grid,
        layers=args.layers_grid,
        dropouts=args.dropout_grid,
        learning_rates=args.lr_grid,
        focal_gammas=args.focal_gamma_grid,
        ranking_loss_weights=args.ranking_loss_weight_grid,
    )
    if args.max_configs:
        configs = configs[:args.max_configs]
    if not configs:
        raise RuntimeError("No GNN tuning configs matched the requested grid filters.")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.output).parent
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        candidates, best_config_key, best_score, best_summary, warnings = _restore_candidates(args.output, args.metric)
    else:
        candidates = {}
        best_config_key = None
        best_score = float("-inf")
        best_summary = None
        warnings = []
    total_runs = len(configs) * args.runs
    completed_runs = 0
    started_at = time.time()
    tabular = _load_tabular_baseline(args.tabular_report, args.metric)

    for config_index, config in enumerate(configs, start=1):
        if args.resume and config.key() in candidates and int(candidates[config.key()].get("summary", {}).get("runs", 0)) >= args.runs:
            if args.progress:
                print(_progress_line(completed_runs, total_runs, started_at, f"resume_skip config={config_index}/{len(configs)} {config.key()}"))
            completed_runs += args.runs
            continue
        if args.progress:
            print(_progress_line(completed_runs, total_runs, started_at, f"config={config_index}/{len(configs)} {config.key()}"))
        runs = []
        for offset in range(args.runs):
            seed = args.seed + offset
            checkpoint = checkpoint_dir / f"{config.key()}__seed{seed}.pt"
            eval_path = report_dir / f"{config.key()}__seed{seed}_metrics.json"
            try:
                if args.progress:
                    print(_progress_line(completed_runs, total_runs, started_at, f"start seed={seed} {config.key()}"))
                train_result = train(_train_args(args, config, seed, checkpoint))
            except RuntimeError as exc:
                warnings.append(f"{config.key()} seed={seed} skipped: {exc}")
                completed_runs += 1
                if args.progress:
                    print(_progress_line(completed_runs, total_runs, started_at, f"skipped seed={seed} {exc}"))
                continue
            eval_payload = {
                "data_dir": args.data,
                "checkpoint": train_result["checkpoint"],
                "split": train_result["split"],
                "gnn": train_result["test_metrics"],
                "baseline": train_result["baseline_test_metrics"],
                "teacher": train_result.get("teacher_test_metrics", {}),
                "metric_scope": "held_out_time_test",
            }
            eval_path.write_text(json.dumps(eval_payload, indent=2, sort_keys=True), encoding="utf-8")
            runs.append({
                "seed": seed,
                "checkpoint": train_result["checkpoint"],
                "evaluation_path": str(eval_path),
                "selection_metric": train_result["selection_metric"],
                "best_selection_score": train_result["best_selection_score"],
                "best_selection_metrics": train_result["best_selection_metrics"],
                "best_validation_score": train_result["best_validation_score"],
                "best_validation_metrics": train_result["best_validation_metrics"],
                "gnn_metrics": train_result["test_metrics"],
                "baseline_metrics": train_result["baseline_test_metrics"],
                "teacher_metrics": train_result.get("teacher_test_metrics", {}),
                "split": train_result["split"],
            })
            completed_runs += 1
            if args.progress:
                score = float(train_result["test_metrics"].get(args.metric, 0.0) or 0.0)
                print(_progress_line(completed_runs, total_runs, started_at, f"done seed={seed} {args.metric}={score:.6f}"))
        if not runs:
            continue
        summary = _summarize(runs, args.metric)
        candidates[config.key()] = {
            "config": config.as_dict(),
            "summary": summary,
            "runs": runs,
        }
        if summary["mean"] > best_score:
            best_score = summary["mean"]
            best_summary = summary
            best_config_key = config.key()
        if args.progress:
            print(_progress_line(completed_runs, total_runs, started_at, f"config_done mean_{args.metric}={summary['mean']:.6f} best={best_score:.6f}"))
        _write_report(
            args.output,
            _build_report(args, configs, candidates, best_config_key, best_score, best_summary, tabular, warnings, "running"),
        )

    if best_summary is None:
        raise RuntimeError("No GNN tuning candidates completed successfully.")

    result = _build_report(args, configs, candidates, best_config_key, best_score, best_summary, tabular, warnings, "complete")
    _write_report(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune MuleGuard GNN configs and compare against tabular baselines.")
    parser.add_argument("--data", default="runtime/data/amlsim_1k_features")
    parser.add_argument("--output", default="runtime/reports/gnn_model_selection.json")
    parser.add_argument("--checkpoint-dir", default="models/gnn_tuning")
    parser.add_argument("--tabular-report", default="runtime/reports/model_selection.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="capture_at_5pct")
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--cutoffs", type=_parse_cutoffs, default=[0.01, 0.02, 0.05])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--architectures", type=_parse_csv_strings, default=None)
    parser.add_argument("--graph-views", type=_parse_csv_strings, default=None)
    parser.add_argument("--losses", type=_parse_csv_strings, default=None)
    parser.add_argument("--hidden-channels-grid", type=_parse_csv_ints, default=None)
    parser.add_argument("--layers-grid", type=_parse_csv_ints, default=None)
    parser.add_argument("--dropout-grid", type=_parse_csv_floats, default=None)
    parser.add_argument("--lr-grid", type=_parse_csv_floats, default=None)
    parser.add_argument("--input-skip", dest="input_skip", action="store_true", default=True)
    parser.add_argument("--no-input-skip", dest="input_skip", action="store_false")
    parser.add_argument("--head-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--focal-gamma-grid", type=_parse_csv_floats, default=None)
    parser.add_argument("--ranking-loss-weight-grid", type=_parse_csv_floats, default=None)
    parser.add_argument("--use-tabular-teacher", action="store_true", default=False)
    parser.add_argument("--teacher-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-model", default="numpy_logistic")
    parser.add_argument("--teacher-cv-folds", type=int, default=5)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.1)
    parser.add_argument("--ranking-max-pairs", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--promotion-margin", type=float, default=0.03)
    parser.add_argument("--target-capture-at-5pct", type=float, default=0.25)
    parser.add_argument("--target-lift-at-5pct", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = tune(args)
    print(f"gnn_tuning_written={args.output}")
    print(f"best_config={result['best_config']} {args.metric}={result['best_score']}")
    print(f"promotion_decision={result['promotion_decision']}")


if __name__ == "__main__":
    main()
