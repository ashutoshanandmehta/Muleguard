import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .graph_dataset import LEAKAGE_FEATURES


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for idx in order[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def _correlation(left: List[float], right: List[float]) -> float:
    if len(left) < 2:
        return 0.0
    a = np.array(left, dtype=np.float64)
    b = np.array(right, dtype=np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _feature_names(rows: List[dict]) -> List[str]:
    names = set()
    for row in rows:
        for key in row:
            if key in {"entity_type", "entity_id", "is_mule"} or key in LEAKAGE_FEATURES:
                continue
            names.add(key)
    return sorted(names)


def feature_quality(data_dir: str, output: str, top_n: int = 10) -> Dict:
    path = Path(data_dir) / "muleguard_node_features_full.csv"
    rows = [
        row
        for row in _read_csv(path)
        if row.get("entity_type") == "account" and _maybe_float(row.get("is_mule")) is not None
    ]
    if not rows:
        raise RuntimeError("No labeled account feature rows found.")

    labels = [int(_maybe_float(row.get("is_mule")) or 0) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("Feature quality requires both positive and negative labels.")

    by_feature = {}
    for name in _feature_names(rows):
        parsed = [_maybe_float(row.get(name)) for row in rows]
        present = [(value, label) for value, label in zip(parsed, labels) if value is not None]
        missing_rate = 1.0 - (len(present) / len(rows))
        if not present:
            by_feature[name] = {
                "missing_rate": 1.0,
                "positive_mean": 0.0,
                "negative_mean": 0.0,
                "separation_ratio": 0.0,
                "rank_correlation": 0.0,
                "present_count": 0,
            }
            continue
        values = [value for value, _ in present]
        present_labels = [label for _, label in present]
        pos_values = [value for value, label in present if label == 1]
        neg_values = [value for value, label in present if label == 0]
        pos_mean = sum(pos_values) / len(pos_values) if pos_values else 0.0
        neg_mean = sum(neg_values) / len(neg_values) if neg_values else 0.0
        separation = abs(pos_mean - neg_mean) / max(abs(neg_mean), 1e-6)
        rank_corr = _correlation(_rank(values), present_labels)
        by_feature[name] = {
            "missing_rate": round(missing_rate, 6),
            "positive_mean": round(pos_mean, 6),
            "negative_mean": round(neg_mean, 6),
            "separation_ratio": round(separation, 6),
            "rank_correlation": round(rank_corr, 6),
            "present_count": len(present),
        }

    strongest = sorted(
        by_feature.items(),
        key=lambda item: (abs(item[1]["rank_correlation"]), item[1]["separation_ratio"]),
        reverse=True,
    )[:top_n]
    report = {
        "data_dir": data_dir,
        "accounts": len(rows),
        "positive_accounts": positives,
        "negative_accounts": negatives,
        "feature_count": len(by_feature),
        "top_features": [
            {"feature": name, **metrics}
            for name, metrics in strongest
        ],
        "features": by_feature,
        "excluded_leakage_columns": sorted(LEAKAGE_FEATURES),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report non-leaky feature quality for MuleGuard account features.")
    parser.add_argument("--data", default="runtime/data/amlsim_1k_features")
    parser.add_argument("--output", default="runtime/reports/feature_quality.json")
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = feature_quality(args.data, args.output, args.top_n)
    print(f"feature_quality_written={args.output} features={report['feature_count']}")


if __name__ == "__main__":
    main()
