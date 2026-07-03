import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import log1p
from pathlib import Path
from typing import Dict, Iterable, List, Set


BASE_DATE = datetime(2017, 1, 1)


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _timestamp_from_step(value: str) -> int:
    try:
        step = int(float(value))
    except (TypeError, ValueError):
        step = 0
    return int((BASE_DATE + timedelta(days=max(step - 1, 0))).timestamp())


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scaled_log(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return min(1.0, log1p(max(value, 0.0)) / log1p(reference))


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def convert_amlsim(input_dir: str, output_dir: str) -> Dict[str, int]:
    source = Path(input_dir)
    output = Path(output_dir)
    accounts = _read_csv(source / "accounts.csv")
    final_output_format = (source / "tx.csv").exists()
    transactions = _read_csv(source / "tx.csv" if final_output_format else source / "transactions.csv")
    alerts_path = source / "alerts.csv"
    alert_members_path = source / "alert_members.csv"
    alerts = _read_csv(alerts_path) if alerts_path.exists() else []
    alert_members = _read_csv(alert_members_path) if alert_members_path.exists() else []

    suspicious_accounts: Set[str] = {
        str(row.get("ACCOUNT_ID", "")).strip()
        for row in alerts
        if row.get("ACCOUNT_ID", "") != ""
    }
    suspicious_accounts.update(
        str(row.get("accountID", "")).strip()
        for row in alert_members
        if row.get("accountID", "") != ""
    )
    alert_typology = {
        str(row.get("ACCOUNT_ID", "")).strip(): row.get("CHECK_NAME", "")
        for row in alerts
        if row.get("ACCOUNT_ID", "") != ""
    }
    alert_typology.update({
        str(row.get("accountID", "")).strip(): row.get("reason", "")
        for row in alert_members
        if row.get("accountID", "") != ""
    })

    tx_counts = Counter()
    in_counts = Counter()
    out_counts = Counter()
    total_amount = defaultdict(float)
    first_seen: Dict[str, int] = {}
    core_rows = []

    for row in transactions:
        if final_output_format:
            src = str(row.get("ACCOUNT_ID", "")).strip()
            dst = str(row.get("COUNTER_PARTY_ACCOUNT_NUM", "")).strip()
            amount = float(row.get("TXN_AMOUNT_ORIG") or 0.0)
            timestamp = _timestamp_from_step(row.get("start") or row.get("end") or "0")
            event_id = f"AMLSIM-{row.get('TXN_ID', len(core_rows) + 1)}"
            channel = row.get("TXN_SOURCE_TYPE_CODE", "TRANSFER")
        else:
            src = str(row.get("src", "")).strip()
            dst = str(row.get("dst", "")).strip()
            tx_id = int(float(row.get("id") or len(core_rows) + 1))
            amount = 100.0 + float(tx_id % 900)
            timestamp = _timestamp_from_step(str((tx_id % 720) + 1))
            event_id = f"AMLSIM-{tx_id}"
            channel = row.get("ttype", "TRANSFER")
        core_rows.append({
            "event_id": event_id,
            "kind": "transaction",
            "timestamp": timestamp,
            "amount": amount,
            "currency": "USD",
            "src_account": src,
            "dst_account": dst,
            "merchant_id": "",
            "channel": channel,
            "auth_result": "SUCCESS",
            "txn_status": "POSTED",
        })
        for account_id in (src, dst):
            if account_id:
                tx_counts[account_id] += 1
                total_amount[account_id] += amount
                first_seen[account_id] = min(first_seen.get(account_id, timestamp), timestamp)
        if src:
            out_counts[src] += 1
        if dst:
            in_counts[dst] += 1

    account_ids = [
        str(row.get("ACCOUNT_ID") or row.get("accountID") or "").strip()
        for row in accounts
    ]
    account_ids = [account_id for account_id in account_ids if account_id]
    velocity_reference = max(_percentile([float(tx_counts[aid]) for aid in account_ids], 0.95), 1.0)
    amount_reference = max(_percentile([float(total_amount[aid]) for aid in account_ids], 0.95), 1.0)

    entity_rows = []
    feature_rows = []
    for row in accounts:
        account_id = str(row.get("ACCOUNT_ID") or row.get("accountID") or "").strip()
        if not account_id:
            continue
        customer_id = row.get("PRIMARY_CUSTOMER_ID") or row.get("CUSTOMER_ID") or f"C_{account_id}"
        is_positive = (
            account_id in suspicious_accounts
            or _truthy(row.get("suspicious", ""))
            or _truthy(row.get("isFraud", ""))
            or _truthy(row.get("IS_SAR", ""))
        )
        velocity = float(tx_counts[account_id])
        in_degree = float(in_counts[account_id])
        out_degree = float(out_counts[account_id])
        amount = float(total_amount[account_id])
        velocity_score = _scaled_log(velocity, velocity_reference)
        amount_score = _scaled_log(amount, amount_reference)
        total_degree = max(in_degree + out_degree, 1.0)
        pass_through_score = min(in_degree, out_degree) / total_degree
        fan_imbalance = abs(in_degree - out_degree) / total_degree
        ts_anomaly = min(1.0, 0.45 * velocity_score + 0.35 * amount_score + 0.20 * pass_through_score)
        rule_uplift = 0.0
        if velocity_score >= 0.75 and pass_through_score >= 0.25:
            rule_uplift += 0.12
        if amount_score >= 0.85 and fan_imbalance >= 0.60:
            rule_uplift += 0.10
        if in_degree >= 5.0 and out_degree >= 5.0:
            rule_uplift += 0.08
        rule_uplift = min(0.30, rule_uplift)

        entity_rows.append({
            "account_id": account_id,
            "customer_id": customer_id,
            "device_id": "",
            "ip_id": "",
            "merchant_id": "",
            "session_id": "",
        })
        feature_rows.append({
            "entity_type": "account",
            "entity_id": account_id,
            "txn_velocity_24h": round(velocity, 4),
            "recency_decay": 1.0,
            "ts_anomaly": round(ts_anomaly, 4),
            "rule_uplift": rule_uplift,
            "device_entropy": "",
            "ip_reputation": "",
            "initial_balance": row.get("init_balance") or row.get("INIT_BALANCE", ""),
            "account_country": row.get("country") or row.get("COUNTRY", ""),
            "business_type": row.get("business") or row.get("ACCOUNT_TYPE", ""),
            "amlsim_typology": alert_typology.get(account_id, ""),
            "is_mule": int(is_positive),
        })

    _write_csv(
        output / "muleguard_core_transactions.csv",
        ["event_id", "kind", "timestamp", "amount", "currency", "src_account", "dst_account", "merchant_id", "channel", "auth_result", "txn_status"],
        core_rows,
    )
    _write_csv(
        output / "muleguard_entity_map_full.csv",
        ["account_id", "customer_id", "device_id", "ip_id", "merchant_id", "session_id"],
        entity_rows,
    )
    _write_csv(
        output / "muleguard_node_features_full.csv",
        [
            "entity_type",
            "entity_id",
            "txn_velocity_24h",
            "recency_decay",
            "ts_anomaly",
            "rule_uplift",
            "device_entropy",
            "ip_reputation",
            "initial_balance",
            "account_country",
            "business_type",
            "amlsim_typology",
            "is_mule",
        ],
        feature_rows,
    )
    _write_csv(
        output / "muleguard_digital_telemetry.csv",
        ["kind", "timestamp", "account_id", "session_id", "device_id", "ip_id", "geo_lat", "geo_lon", "vpn_proxy_flag", "failed_logins", "auth_method"],
        [],
    )
    return {
        "accounts": len(entity_rows),
        "transactions": len(core_rows),
        "positive_accounts": sum(int(row["is_mule"]) for row in feature_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AMLSim outputs into MuleGuard-compatible CSVs.")
    parser.add_argument("--input", default="/Users/ashutoshanand/AMLSim/sample/outputs")
    parser.add_argument("--output", default="runtime/data/amlsim_sample")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_amlsim(args.input, args.output)
    print(f"converted={summary} output={args.output}")


if __name__ == "__main__":
    main()
