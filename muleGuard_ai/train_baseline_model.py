import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluate import _metrics, _parse_cutoffs
from .graph_dataset import build_graph_dataset


SKLEARN_MODELS = {"logistic", "random_forest", "gradient_boosting"}
ALL_MODELS = ["numpy_logistic", "logistic", "random_forest", "gradient_boosting"]


def _require_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for this model. Use `--model numpy_logistic` "
            "or install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return GradientBoostingClassifier, RandomForestClassifier, LogisticRegression, make_pipeline, StandardScaler


def _load_labeled_matrix(data_dir: str):
    data = Path(data_dir)
    dataset = build_graph_dataset(
        str(data / "muleguard_core_transactions.csv"),
        str(data / "muleguard_digital_telemetry.csv"),
        str(data / "muleguard_entity_map_full.csv"),
        str(data / "muleguard_node_features_full.csv"),
    )
    account_ids = dataset.labeled_accounts()
    if not account_ids:
        raise RuntimeError("No labeled accounts found. Add an `is_mule` column to node features.")
    x = np.array([dataset.features[("account", account_id)] for account_id in account_ids], dtype=np.float32)
    y = np.array([dataset.account_labels[account_id] for account_id in account_ids], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise RuntimeError("Training requires both positive and negative account labels.")
    return dataset, account_ids, x, y


def _random_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in sorted(set(y.tolist())):
        idxs = np.where(y == label)[0]
        rng.shuffle(idxs)
        if len(idxs) < 2:
            train_parts.append(idxs)
            continue
        test_count = max(1, int(round(len(idxs) * test_size)))
        test_count = min(test_count, len(idxs) - 1)
        test_parts.append(idxs[:test_count])
        train_parts.append(idxs[test_count:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts) if test_parts else train_idx.copy()
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _time_split_indices(
    account_ids: List[str],
    first_seen: Dict[str, int],
    y: np.ndarray,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ordered = sorted(range(len(account_ids)), key=lambda idx: (first_seen.get(account_ids[idx], 0), account_ids[idx]))
    if len(ordered) < 2:
        idx = np.array(ordered, dtype=np.int64)
        return idx, idx.copy()
    test_count = max(1, int(round(len(ordered) * test_size)))
    test_count = min(test_count, len(ordered) - 1)
    train_idx = np.array(ordered[:-test_count], dtype=np.int64)
    test_idx = np.array(ordered[-test_count:], dtype=np.int64)
    if len(set(y[train_idx].astype(int).tolist())) < 2:
        raise RuntimeError("Time split training slice has only one class. Use a larger dataset or earlier labels.")
    return train_idx, test_idx


def _split_indices(
    account_ids: List[str],
    first_seen: Dict[str, int],
    y: np.ndarray,
    test_size: float,
    seed: int,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if strategy == "time":
        return _time_split_indices(account_ids, first_seen, y, test_size)
    if strategy == "random":
        return _random_split_indices(y, test_size, seed)
    raise RuntimeError(f"Unsupported split strategy: {strategy}")


def _split_metadata(dataset, account_ids: List[str], train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray, strategy: str) -> Dict:
    def bounds(indices: np.ndarray) -> Dict:
        if len(indices) == 0:
            return {"accounts": 0, "positives": 0, "negatives": 0, "first_seen_min": None, "first_seen_max": None}
        times = [dataset.account_first_seen.get(account_ids[int(idx)], 0) for idx in indices]
        positives = int(y[indices].sum())
        return {
            "accounts": int(len(indices)),
            "positives": positives,
            "negatives": int(len(indices) - positives),
            "first_seen_min": int(min(times)),
            "first_seen_max": int(max(times)),
        }

    return {
        "split_strategy": strategy,
        "train": bounds(train_idx),
        "test": bounds(test_idx),
    }


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _class_weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y, dtype=np.float32)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = np.ones(2, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights[y]


def _fit_logistic_regression(x: np.ndarray, y: np.ndarray, class_weight: str, lr: float, max_iter: int, l2: float) -> Tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    sample_weight = _class_weights(y, class_weight)
    sample_weight = sample_weight / max(sample_weight.mean(), 1e-6)
    y_float = y.astype(np.float32)
    for _ in range(max_iter):
        probs = _sigmoid(x @ weights + bias)
        error = (probs - y_float) * sample_weight
        grad_w = (x.T @ error) / len(x) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def _train_numpy_logistic(args: argparse.Namespace, x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[Dict, List[float]]:
    train_x, test_x, mean, std = _standardize(x[train_idx], x[test_idx])
    weights, bias = _fit_logistic_regression(
        train_x,
        y[train_idx],
        args.class_weight,
        args.lr,
        args.max_iter,
        args.l2,
    )
    scores = _sigmoid(test_x @ weights + bias).tolist()
    payload = {
        "type": "numpy_logistic_regression",
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "std": std,
    }
    return payload, scores


def _train_sklearn_model(args: argparse.Namespace, model_name: str, x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> Tuple[Dict, List[float]]:
    GradientBoostingClassifier, RandomForestClassifier, LogisticRegression, make_pipeline, StandardScaler = _require_sklearn()
    class_weight = args.class_weight if args.class_weight != "none" else None
    if model_name == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight=class_weight, max_iter=args.max_iter, random_state=seed),
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight=class_weight,
            random_state=seed,
        )
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth or 3,
            random_state=seed,
        )
    else:
        raise RuntimeError(f"Unsupported sklearn model: {model_name}")
    estimator.fit(x[train_idx], y[train_idx])
    scores = estimator.predict_proba(x[test_idx])[:, 1].tolist()
    return {"type": model_name, "estimator": estimator}, scores


def _train_candidate(args: argparse.Namespace, model_name: str, seed: int) -> Dict:
    dataset, account_ids, x, y = _load_labeled_matrix(args.data)

    train_idx, test_idx = _split_indices(account_ids, dataset.account_first_seen, y, args.test_size, seed, args.split_strategy)
    if model_name == "numpy_logistic":
        model_payload, scores = _train_numpy_logistic(args, x, y, train_idx, test_idx)
    elif model_name in SKLEARN_MODELS:
        model_payload, scores = _train_sklearn_model(args, model_name, x, y, train_idx, test_idx, seed)
    else:
        raise RuntimeError(f"Unsupported model: {model_name}")
    labels = y[test_idx].astype(int).tolist()

    metrics = _metrics(labels, scores, threshold=args.threshold, top_k=args.top_k, cutoffs=args.cutoffs)
    payload = {
        "model": model_payload,
        "feature_names": dataset.feature_names,
        "metadata": {
            "data_dir": args.data,
            "model": model_name,
            "seed": seed,
            "test_size": args.test_size,
            "split_strategy": args.split_strategy,
            "split": _split_metadata(dataset, account_ids, train_idx, test_idx, y, args.split_strategy),
            "class_weight": args.class_weight,
            "lr": args.lr,
            "l2": args.l2,
            "max_iter": args.max_iter,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "train_accounts": int(len(train_idx)),
            "test_accounts": int(len(test_idx)),
        },
    }
    return {
        "payload": payload,
        "model_path": args.output,
        "feature_names": dataset.feature_names,
        "metrics": metrics,
        "metadata": payload["metadata"],
    }


def _write_model(path: str, payload: Dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(payload, f)

def _write_result(path: Optional[str], result: Dict) -> None:
    if not path:
        return
    metrics_out = Path(path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key != "payload"}
    metrics_out.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def _metric_value(result: Dict, metric: str) -> float:
    try:
        return float(result["metrics"][metric])
    except KeyError as exc:
        raise RuntimeError(f"Metric `{metric}` is not available in model results.") from exc


def _summarize_runs(runs: List[Dict], metric: str) -> Dict:
    values = [_metric_value(run, metric) for run in runs]
    best_run = max(runs, key=lambda run: _metric_value(run, metric))
    summary = {
        "runs": len(runs),
        "metric": metric,
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "best": round(max(values), 6),
        "best_seed": best_run["metadata"]["seed"],
        "best_metrics": best_run["metrics"],
    }
    return summary


def _select_best(args: argparse.Namespace) -> Dict:
    candidates = ALL_MODELS
    candidate_runs = {}
    warnings = []
    best_run = None
    best_model_name = None
    best_score = float("-inf")

    for model_name in candidates:
        runs = []
        for offset in range(args.runs):
            seed = args.seed + offset
            try:
                run = _train_candidate(args, model_name, seed)
            except RuntimeError as exc:
                warnings.append(f"{model_name} skipped: {exc}")
                runs = []
                break
            runs.append(run)
        if not runs:
            continue
        summary = _summarize_runs(runs, args.metric)
        candidate_runs[model_name] = {
            "summary": summary,
            "runs": [
                {
                    "model_path": run["model_path"],
                    "metrics": run["metrics"],
                    "metadata": run["metadata"],
                }
                for run in runs
            ],
        }
        run_best = max(runs, key=lambda run: _metric_value(run, args.metric))
        run_score = summary["mean"]
        if run_score > best_score:
            best_run = run_best
            best_model_name = model_name
            best_score = run_score

    if best_run is None:
        raise RuntimeError("No candidate models could be trained.")
    _write_model(args.output, best_run["payload"])
    result = {
        "model_path": args.output,
        "selection_metric": args.metric,
        "best_model": best_model_name,
        "best_score": round(best_score, 6),
        "best_score_basis": "mean_across_runs",
        "best_metrics": best_run["metrics"],
        "best_metadata": best_run["metadata"],
        "feature_names": best_run["feature_names"],
        "candidates": candidate_runs,
        "warnings": warnings,
    }
    _write_result(args.metrics_out, result)
    return result


def train(args: argparse.Namespace) -> dict:
    if args.select_best:
        return _select_best(args)
    result = _train_candidate(args, args.model, args.seed)
    _write_model(args.output, result["payload"])
    if args.metrics_out:
        _write_result(args.metrics_out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MuleGuard's tabular supervised baseline model.")
    parser.add_argument("--data", default="runtime/data/amlsim_sample")
    parser.add_argument("--output", default="models/tabular_baseline.pkl")
    parser.add_argument("--metrics-out", default="runtime/reports/tabular_baseline_metrics.json")
    parser.add_argument("--model", choices=ALL_MODELS, default="numpy_logistic")
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--metric", default="capture_at_5pct")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["time", "random"], default="time")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cutoffs", type=_parse_cutoffs, default=[0.01, 0.02, 0.05])
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = train(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"saved model: {result['model_path']}")
    if args.select_best:
        print(f"best_model={result['best_model']} {args.metric}={result['best_score']}")
    if args.metrics_out:
        print(f"metrics_written={args.metrics_out}")


if __name__ == "__main__":
    main()
