import argparse
import csv
import json
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .gnn_inference import score_accounts_with_checkpoint
from .graph_dataset import build_graph_dataset
from .risk_baseline import score_accounts


def _read_node_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {
            row["entity_id"]: row
            for row in rows
            if row.get("entity_type") == "account" and row.get("entity_id")
        }


def _parse_cutoffs(value: str) -> List[float]:
    cutoffs = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        cutoff = float(item)
        if not 0.0 < cutoff <= 1.0:
            raise argparse.ArgumentTypeError("cutoffs must be fractions in the range (0, 1].")
        cutoffs.append(cutoff)
    if not cutoffs:
        raise argparse.ArgumentTypeError("at least one cutoff is required.")
    return cutoffs


def _cutoff_label(cutoff: float) -> str:
    pct = cutoff * 100.0
    if pct.is_integer():
        return f"{int(pct)}pct"
    return f"{str(round(pct, 3)).replace('.', '_')}pct"


def _pr_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    fp = 0
    points = [(0.0, 1.0)]
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        points.append((recall, precision))
    area = 0.0
    prev_recall = 0.0
    prev_precision = points[0][1]
    for recall, precision in points[1:]:
        area += (recall - prev_recall) * precision
        prev_recall = recall
        prev_precision = precision
    return round(area, 6)


def _ks_statistic(labels: List[int], scores: List[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    fp = 0
    ks = 0.0
    for idx in ordered:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        ks = max(ks, abs((tp / positives) - (fp / negatives)))
    return round(ks, 6)


def _ranking_at_cutoffs(labels: List[int], scores: List[float], cutoffs: Iterable[float]) -> Dict:
    positives = sum(labels)
    prevalence = positives / len(labels) if labels else 0.0
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    metrics = {}
    for cutoff in cutoffs:
        k = max(1, min(len(labels), ceil(len(labels) * cutoff)))
        top = ordered[:k]
        hits = sum(labels[idx] for idx in top)
        precision = hits / k if k else 0.0
        label = _cutoff_label(cutoff)
        metrics[f"capture_at_{label}"] = round(hits / positives, 6) if positives else 0.0
        metrics[f"precision_at_{label}"] = round(precision, 6)
        metrics[f"lift_at_{label}"] = round(precision / prevalence, 6) if prevalence else 0.0
        metrics[f"review_count_at_{label}"] = k
    return metrics


def _metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
    top_k: int = 10,
    cutoffs: Optional[List[float]] = None,
) -> Dict:
    if not labels:
        raise RuntimeError("No labeled accounts found. Evaluation requires account labels.")
    if cutoffs is None:
        cutoffs = [0.01, 0.02, 0.05]
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    positives = sum(labels)
    top_hits = sum(labels[idx] for idx in ordered)
    pr_auc = _pr_auc(labels, scores)
    warnings = []
    if len(labels) < 100 or positives < 5 or (len(labels) - positives) < 5:
        warnings.append("Dataset is too small for serious model-quality claims; use this result as a smoke test only.")
    if pr_auc is None:
        warnings.append("PR-AUC skipped because labels do not contain both positive and negative classes.")
    result = {
        "threshold": threshold,
        "accounts_evaluated": len(labels),
        "positive_accounts": positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "top_k": min(top_k, len(labels)),
        "top_k_recall": round(top_hits / positives, 6) if positives else 0.0,
        "pr_auc": pr_auc,
        "ks": _ks_statistic(labels, scores),
        "warnings": warnings,
    }
    result.update(_ranking_at_cutoffs(labels, scores, cutoffs))
    return result


def _candidate_thresholds(scores: List[float]) -> List[float]:
    unique = sorted(set(scores))
    if not unique:
        return [0.5]
    candidates = [0.0, 1.0]
    candidates.extend(unique)
    for left, right in zip(unique, unique[1:]):
        candidates.append((left + right) / 2.0)
    return sorted(set(candidates))


def _f1_for_threshold(labels: List[int], scores: List[float], threshold: float) -> float:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _validation_indices(labels: List[int], account_ids: List[str], first_seen: Dict[str, int]) -> Tuple[List[int], List[str]]:
    if len(labels) < 20 or sum(labels) < 3:
        return list(range(len(labels))), ["Calibration used all labeled accounts because validation data was too small."]
    ordered = sorted(range(len(labels)), key=lambda idx: (first_seen.get(account_ids[idx], 0), account_ids[idx]))
    start = int(len(ordered) * 0.6)
    end = max(start + 1, int(len(ordered) * 0.8))
    validation = ordered[start:end]
    positives = sum(labels[idx] for idx in validation)
    negatives = len(validation) - positives
    if positives == 0 or negatives == 0:
        return ordered, ["Calibration used all labeled accounts because the validation slice had only one class."]
    return validation, []


def _calibrate_threshold(labels: List[int], scores: List[float], account_ids: List[str], first_seen: Dict[str, int]) -> Tuple[float, List[str]]:
    validation, warnings = _validation_indices(labels, account_ids, first_seen)
    best_threshold = 0.5
    best_score = -1.0
    val_labels = [labels[idx] for idx in validation]
    val_scores = [scores[idx] for idx in validation]
    for candidate in _candidate_thresholds(val_scores):
        score = _f1_for_threshold(val_labels, val_scores, candidate)
        if score > best_score or (score == best_score and candidate > best_threshold):
            best_threshold = candidate
            best_score = score
    return round(float(best_threshold), 6), warnings


def _decile_rows(scorer: str, labels: List[int], scores: List[float], account_ids: List[str]) -> List[Dict]:
    positives = sum(labels)
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    rows = []
    for decile in range(1, 11):
        start = int((decile - 1) * len(ordered) / 10)
        end = int(decile * len(ordered) / 10)
        idxs = ordered[start:end]
        if not idxs:
            continue
        hits = sum(labels[idx] for idx in idxs)
        rows.append({
            "scorer": scorer,
            "decile": decile,
            "count": len(idxs),
            "positives": hits,
            "fraud_rate": round(hits / len(idxs), 6),
            "capture": round(hits / positives, 6) if positives else 0.0,
            "score_min": round(min(scores[idx] for idx in idxs), 6),
            "score_max": round(max(scores[idx] for idx in idxs), 6),
            "sample_accounts": ",".join(account_ids[idx] for idx in idxs[:5]),
        })
    return rows


def _error_analysis(
    labels: List[int],
    scores: List[float],
    account_ids: List[str],
    threshold: float,
    metadata: Dict[str, Dict[str, str]],
    limit: int = 25,
) -> Dict:
    false_positives = []
    false_negatives = []
    for label, score, account_id in zip(labels, scores, account_ids):
        predicted = 1 if score >= threshold else 0
        row = {
            "account_id": account_id,
            "score": round(score, 6),
            "label": label,
            "typology": metadata.get(account_id, {}).get("amlsim_typology", ""),
        }
        if label == 0 and predicted == 1:
            false_positives.append(row)
        if label == 1 and predicted == 0:
            false_negatives.append(row)
    false_positives.sort(key=lambda row: row["score"], reverse=True)
    false_negatives.sort(key=lambda row: row["score"], reverse=True)
    return {
        "threshold": threshold,
        "false_positives": false_positives[:limit],
        "false_negatives": false_negatives[:limit],
    }


def _typology_metrics(labels: List[int], scores: List[float], account_ids: List[str], threshold: float, metadata: Dict[str, Dict[str, str]]) -> Dict:
    groups: Dict[str, List[int]] = {}
    for idx, account_id in enumerate(account_ids):
        typology = metadata.get(account_id, {}).get("amlsim_typology", "")
        if typology:
            groups.setdefault(typology, []).append(idx)
    report = {}
    for typology, idxs in sorted(groups.items()):
        group_labels = [labels[idx] for idx in idxs]
        group_scores = [scores[idx] for idx in idxs]
        report[typology] = _metrics(group_labels, group_scores, threshold, top_k=min(10, len(idxs)), cutoffs=[1.0])
    return report


def _write_deciles(path: str, rows: List[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scorer", "decile", "count", "positives", "fraud_rate", "capture", "score_min", "score_max", "sample_accounts"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload: Dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate(
    data_dir: str,
    checkpoint: Optional[str],
    output: str,
    threshold: float,
    top_k: int,
    cutoffs: Optional[List[float]] = None,
    calibrate_threshold: bool = False,
    error_analysis_out: Optional[str] = None,
    deciles_out: Optional[str] = None,
    typology_report_out: Optional[str] = None,
) -> Dict:
    data = Path(data_dir)
    node_features_path = data / "muleguard_node_features_full.csv"
    dataset = build_graph_dataset(
        str(data / "muleguard_core_transactions.csv"),
        str(data / "muleguard_digital_telemetry.csv"),
        str(data / "muleguard_entity_map_full.csv"),
        str(node_features_path),
    )
    account_ids = dataset.labeled_accounts()
    if not account_ids:
        raise RuntimeError("No labeled accounts found. Evaluation requires account labels.")

    metadata = _read_node_metadata(node_features_path)
    baseline_scores = {risk.account_id: risk.score for risk in score_accounts(dataset)}
    labels = [dataset.account_labels[account_id] for account_id in account_ids]
    score_sets = {
        "baseline": [baseline_scores[account_id] for account_id in account_ids],
    }
    if checkpoint:
        gnn_scores = {risk.account_id: risk.score for risk in score_accounts_with_checkpoint(dataset, checkpoint)}
        score_sets["gnn"] = [gnn_scores[account_id] for account_id in account_ids]

    results = {
        "data_dir": data_dir,
        "cutoffs": cutoffs or [0.01, 0.02, 0.05],
    }
    decile_rows = []
    error_payload = {}
    typology_payload = {}
    for scorer, scores in score_sets.items():
        scorer_threshold = threshold
        calibration_warnings = []
        if calibrate_threshold:
            scorer_threshold, calibration_warnings = _calibrate_threshold(labels, scores, account_ids, dataset.account_first_seen)
        metrics = _metrics(labels, scores, scorer_threshold, top_k, cutoffs)
        metrics["calibrated_threshold"] = bool(calibrate_threshold)
        metrics["warnings"].extend(calibration_warnings)
        results[scorer] = metrics
        decile_rows.extend(_decile_rows(scorer, labels, scores, account_ids))
        error_payload[scorer] = _error_analysis(labels, scores, account_ids, scorer_threshold, metadata)
        typology_payload[scorer] = _typology_metrics(labels, scores, account_ids, scorer_threshold, metadata)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if deciles_out:
        _write_deciles(deciles_out, decile_rows)
    if error_analysis_out:
        _write_json(error_analysis_out, error_payload)
    if typology_report_out:
        _write_json(typology_report_out, typology_payload)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MuleGuard scorers on labeled MuleGuard CSV data.")
    parser.add_argument("--data", default="runtime/data/amlsim_sample")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="runtime/reports/evaluation_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cutoffs", type=_parse_cutoffs, default=[0.01, 0.02, 0.05])
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--error-analysis-out", default=None)
    parser.add_argument("--deciles-out", default=None)
    parser.add_argument("--typology-report-out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate(
        args.data,
        args.checkpoint,
        args.output,
        args.threshold,
        args.top_k,
        args.cutoffs,
        args.calibrate_threshold,
        args.error_analysis_out,
        args.deciles_out,
        args.typology_report_out,
    )
    print(f"metrics_written={args.output}")
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "warnings" in metrics:
            for warning in metrics["warnings"]:
                print(f"warning[{name}]: {warning}")


if __name__ == "__main__":
    main()
