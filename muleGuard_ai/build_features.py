import argparse
import csv
import shutil
from collections import Counter, defaultdict
from math import log1p, sqrt
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Set


DATA_FILES = [
    "muleguard_core_transactions.csv",
    "muleguard_entity_map_full.csv",
    "muleguard_digital_telemetry.csv",
]

METADATA_COLUMNS = {"entity_type", "entity_id", "is_mule", "amlsim_typology"}
LEAKAGE_COLUMNS = {"is_mule", "amlsim_typology", "alert_name", "alert_type", "typology", "typology_label", "CHECK_NAME", "reason"}


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _day(ts: int) -> int:
    return ts // 86400 if ts else 0


def _numeric_original(row: dict) -> Dict[str, str]:
    keep = {}
    for key, value in row.items():
        if key in METADATA_COLUMNS or key in LEAKAGE_COLUMNS:
            continue
        if value in (None, ""):
            keep[key] = value
            continue
        try:
            float(value)
        except ValueError:
            continue
        keep[key] = value
    return keep


def _account_ids(entity_rows: List[dict], feature_rows: List[dict], tx_rows: List[dict]) -> Set[str]:
    ids = set()
    for row in entity_rows:
        if row.get("account_id"):
            ids.add(row["account_id"])
    for row in feature_rows:
        if row.get("entity_type") == "account" and row.get("entity_id"):
            ids.add(row["entity_id"])
    for row in tx_rows:
        if row.get("src_account"):
            ids.add(row["src_account"])
        if row.get("dst_account"):
            ids.add(row["dst_account"])
    return ids


def _rapid_in_out_count(in_events: List[tuple], out_events: List[tuple], window_seconds: int) -> int:
    count = 0
    out_idx = 0
    out_events = sorted(out_events)
    for in_ts, _ in sorted(in_events):
        while out_idx < len(out_events) and out_events[out_idx][0] < in_ts:
            out_idx += 1
        j = out_idx
        while j < len(out_events) and out_events[j][0] - in_ts <= window_seconds:
            count += 1
            j += 1
    return count


def _engineer_features(account_ids: Set[str], tx_rows: List[dict], rapid_window_seconds: int) -> Dict[str, Dict[str, float]]:
    in_counts = Counter()
    out_counts = Counter()
    in_counterparties = defaultdict(set)
    out_counterparties = defaultdict(set)
    in_amounts = defaultdict(list)
    out_amounts = defaultdict(list)
    all_amounts = defaultdict(list)
    timestamps = defaultdict(list)
    daily_counts = defaultdict(Counter)
    daily_amounts = defaultdict(lambda: defaultdict(float))
    in_events = defaultdict(list)
    out_events = defaultdict(list)
    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)

    for row in tx_rows:
        src = row.get("src_account", "")
        dst = row.get("dst_account", "")
        ts = _int(row.get("timestamp"))
        amount = _float(row.get("amount"))
        if src:
            out_counts[src] += 1
            out_counterparties[src].add(dst)
            out_amounts[src].append(amount)
            all_amounts[src].append(amount)
            timestamps[src].append(ts)
            daily_counts[src][_day(ts)] += 1
            daily_amounts[src][_day(ts)] += amount
            out_events[src].append((ts, dst))
            if dst:
                out_neighbors[src].add(dst)
        if dst:
            in_counts[dst] += 1
            in_counterparties[dst].add(src)
            in_amounts[dst].append(amount)
            all_amounts[dst].append(amount)
            timestamps[dst].append(ts)
            daily_counts[dst][_day(ts)] += 1
            daily_amounts[dst][_day(ts)] += amount
            in_events[dst].append((ts, src))
            if src:
                in_neighbors[dst].add(src)

    features = {}
    for account_id in sorted(account_ids):
        in_tx = float(in_counts[account_id])
        out_tx = float(out_counts[account_id])
        total_tx = in_tx + out_tx
        inbound = in_counterparties[account_id]
        outbound = out_counterparties[account_id]
        unique_total = len(inbound | outbound)
        amount_values = all_amounts[account_id]
        in_sum = sum(in_amounts[account_id])
        out_sum = sum(out_amounts[account_id])
        amount_sum = sum(amount_values)
        amount_mean = _safe_ratio(amount_sum, len(amount_values))
        amount_std = _std(amount_values)
        sorted_ts = sorted(ts for ts in timestamps[account_id] if ts)
        active_span_days = _safe_ratio((max(sorted_ts) - min(sorted_ts)), 86400.0) + 1.0 if sorted_ts else 0.0
        gaps = [right - left for left, right in zip(sorted_ts, sorted_ts[1:])]
        max_daily_count = max(daily_counts[account_id].values()) if daily_counts[account_id] else 0.0
        max_daily_amount = max(daily_amounts[account_id].values()) if daily_amounts[account_id] else 0.0
        overlap = inbound & outbound
        two_hop_out = set()
        for neighbor in outbound:
            two_hop_out.update(out_neighbors.get(neighbor, set()))
        two_hop_out.discard(account_id)
        two_hop_in = set()
        for neighbor in inbound:
            two_hop_in.update(in_neighbors.get(neighbor, set()))
        two_hop_in.discard(account_id)
        cycle_count = sum(1 for neighbor in outbound if account_id in out_neighbors.get(neighbor, set()))
        shared_counterparties = len(overlap)

        features[account_id] = {
            "in_tx_count": in_tx,
            "out_tx_count": out_tx,
            "in_out_ratio": _safe_ratio(in_tx, out_tx),
            "unique_in_counterparties": float(len(inbound)),
            "unique_out_counterparties": float(len(outbound)),
            "unique_total_counterparties": float(unique_total),
            "counterparty_reuse_ratio": 1.0 - _safe_ratio(unique_total, total_tx),
            "pass_through_ratio": _safe_ratio(min(in_tx, out_tx), total_tx),
            "fan_in_score": _safe_ratio(in_tx, total_tx),
            "fan_out_score": _safe_ratio(out_tx, total_tx),
            "amount_sum": amount_sum,
            "amount_mean": amount_mean,
            "amount_std": amount_std,
            "amount_max": max(amount_values) if amount_values else 0.0,
            "amount_cv": _safe_ratio(amount_std, amount_mean),
            "amount_in_sum": in_sum,
            "amount_out_sum": out_sum,
            "amount_in_out_ratio": _safe_ratio(in_sum, out_sum),
            "first_tx_ts": float(min(sorted_ts)) if sorted_ts else 0.0,
            "last_tx_ts": float(max(sorted_ts)) if sorted_ts else 0.0,
            "active_span_days": active_span_days,
            "tx_per_active_day": _safe_ratio(total_tx, active_span_days),
            "max_daily_tx_count": float(max_daily_count),
            "max_daily_amount": float(max_daily_amount),
            "burst_tx_ratio": _safe_ratio(float(max_daily_count), total_tx),
            "burst_amount_ratio": _safe_ratio(float(max_daily_amount), amount_sum),
            "median_inter_tx_gap_seconds": float(median(gaps)) if gaps else 0.0,
            "rapid_in_out_count": float(_rapid_in_out_count(in_events[account_id], out_events[account_id], rapid_window_seconds)),
            "two_hop_out_count": float(len(two_hop_out)),
            "two_hop_in_count": float(len(two_hop_in)),
            "cycle_count_2hop": float(cycle_count),
            "shared_counterparty_count": float(shared_counterparties),
            "chain_middle_score": _safe_ratio(float(len(inbound - outbound) * len(outbound - inbound)), max(unique_total, 1)),
            "sink_score": _safe_ratio(in_tx, total_tx) * (1.0 - _safe_ratio(out_tx, total_tx)),
            "source_score": _safe_ratio(out_tx, total_tx) * (1.0 - _safe_ratio(in_tx, total_tx)),
        }
    return features


def build_features(data_dir: str, output_dir: str, rapid_window_seconds: int = 86400) -> Dict[str, int]:
    source = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for filename in DATA_FILES:
        src = source / filename
        dst = output / filename
        if src.exists():
            shutil.copyfile(src, dst)

    tx_rows = _read_csv(source / "muleguard_core_transactions.csv")
    entity_rows = _read_csv(source / "muleguard_entity_map_full.csv")
    feature_path = source / "muleguard_node_features_full.csv"
    feature_rows = _read_csv(feature_path)
    account_ids = _account_ids(entity_rows, feature_rows, tx_rows)
    engineered = _engineer_features(account_ids, tx_rows, rapid_window_seconds)

    existing = {
        row["entity_id"]: row
        for row in feature_rows
        if row.get("entity_type") == "account" and row.get("entity_id")
    }
    output_rows = []
    for account_id in sorted(account_ids):
        original = existing.get(account_id, {})
        row = {
            "entity_type": "account",
            "entity_id": account_id,
            **_numeric_original(original),
            **{key: round(value, 6) for key, value in engineered.get(account_id, {}).items()},
            "amlsim_typology": original.get("amlsim_typology", ""),
            "is_mule": original.get("is_mule", ""),
        }
        output_rows.append(row)

    fields = ["entity_type", "entity_id"]
    numeric_fields = sorted({
        key
        for row in output_rows
        for key in row
        if key not in {"entity_type", "entity_id", "amlsim_typology", "is_mule"}
    })
    fields.extend(numeric_fields)
    fields.extend(["amlsim_typology", "is_mule"])
    _write_csv(output / "muleguard_node_features_full.csv", fields, output_rows)
    return {
        "accounts": len(output_rows),
        "features": len(numeric_fields),
        "transactions": len(tx_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build enhanced MuleGuard account features.")
    parser.add_argument("--data", default="runtime/data/amlsim_1k")
    parser.add_argument("--output", default="runtime/data/amlsim_1k_features")
    parser.add_argument("--rapid-window-seconds", type=int, default=86400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_features(args.data, args.output, args.rapid_window_seconds)
    print(f"features_built={summary} output={args.output}")


if __name__ == "__main__":
    main()
