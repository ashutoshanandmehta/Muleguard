import csv
from dataclasses import dataclass
from math import log1p
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TypedNode:
    node_type: str
    node_id: str


@dataclass(frozen=True)
class TypedEdge:
    src_type: str
    src_id: str
    relation: str
    dst_type: str
    dst_id: str
    timestamp: int
    weight: float


@dataclass
class GraphDataset:
    node_types: Dict[str, List[str]]
    features: Dict[Tuple[str, str], np.ndarray]
    feature_names: List[str]
    edges: List[TypedEdge]
    account_labels: Dict[str, int]
    account_first_seen: Dict[str, int]

    def account_ids(self) -> List[str]:
        return self.node_types.get("account", [])

    def labeled_accounts(self) -> List[str]:
        return [account_id for account_id in self.account_ids() if account_id in self.account_labels]


BASE_FEATURES = [
    "txn_velocity_24h",
    "recency_decay",
    "ts_anomaly",
    "rule_uplift",
    "device_entropy",
    "ip_reputation",
    "in_degree",
    "out_degree",
    "total_amount_log",
]

LEAKAGE_FEATURES = {
    "is_mule",
    "amlsim_typology",
    "alert_name",
    "alert_type",
    "typology",
    "typology_label",
    "CHECK_NAME",
    "reason",
}


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _add_node(nodes: Dict[str, set], node_type: str, node_id: Optional[str]) -> None:
    if node_id:
        nodes.setdefault(node_type, set()).add(node_id)


def _feature_map(rows: Iterable[dict]) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, int], List[str]]:
    features: Dict[Tuple[str, str], Dict[str, float]] = {}
    labels: Dict[str, int] = {}
    feature_names = set(BASE_FEATURES)
    for row in rows:
        node_type = row["entity_type"]
        node_id = row["entity_id"]
        vals: Dict[str, float] = {}
        for key, value in row.items():
            if key in ("entity_type", "entity_id"):
                continue
            if key == "is_mule":
                parsed = _maybe_float(value)
                if node_type == "account" and parsed is not None:
                    labels[node_id] = int(parsed)
                continue
            if key in LEAKAGE_FEATURES:
                continue
            parsed = _maybe_float(value)
            if parsed is not None:
                vals[key] = parsed
                feature_names.add(key)
        features[(node_type, node_id)] = vals
    ordered = list(BASE_FEATURES)
    ordered.extend(sorted(name for name in feature_names if name not in set(BASE_FEATURES)))
    return features, labels, ordered


def build_graph_dataset(
    tx_path: str,
    telemetry_path: str,
    entity_map_path: str,
    node_features_path: str,
) -> GraphDataset:
    tx_rows = _read_csv(Path(tx_path))
    telemetry_rows = _read_csv(Path(telemetry_path))
    entity_rows = _read_csv(Path(entity_map_path))
    raw_features, labels, feature_names = _feature_map(_read_csv(Path(node_features_path)))

    nodes: Dict[str, set] = {}
    edges: List[TypedEdge] = []
    in_degree: Dict[Tuple[str, str], float] = {}
    out_degree: Dict[Tuple[str, str], float] = {}
    amount_log: Dict[Tuple[str, str], float] = {}
    first_seen: Dict[str, int] = {}

    def add_edge(edge: TypedEdge) -> None:
        edges.append(edge)
        _add_node(nodes, edge.src_type, edge.src_id)
        _add_node(nodes, edge.dst_type, edge.dst_id)
        out_key = (edge.src_type, edge.src_id)
        in_key = (edge.dst_type, edge.dst_id)
        out_degree[out_key] = out_degree.get(out_key, 0.0) + 1.0
        in_degree[in_key] = in_degree.get(in_key, 0.0) + 1.0

    for row in entity_rows:
        account_id = row.get("account_id")
        customer_id = row.get("customer_id")
        device_id = row.get("device_id")
        ip_id = row.get("ip_id")
        merchant_id = row.get("merchant_id")
        session_id = row.get("session_id")
        _add_node(nodes, "account", account_id)
        _add_node(nodes, "customer", customer_id)
        _add_node(nodes, "device", device_id)
        _add_node(nodes, "ip", ip_id)
        _add_node(nodes, "merchant", merchant_id)
        _add_node(nodes, "session", session_id)
        if account_id and customer_id:
            add_edge(TypedEdge("customer", customer_id, "owns", "account", account_id, 0, 1.0))
        if account_id and device_id:
            add_edge(TypedEdge("account", account_id, "uses_device", "device", device_id, 0, 1.0))
        if device_id and ip_id:
            add_edge(TypedEdge("device", device_id, "uses_ip", "ip", ip_id, 0, 1.0))
        if session_id and account_id:
            add_edge(TypedEdge("session", session_id, "session_to_account", "account", account_id, 0, 1.0))
        if account_id and merchant_id:
            add_edge(TypedEdge("account", account_id, "pays_merchant", "merchant", merchant_id, 0, 1.0))

    for row in tx_rows:
        ts = int(row["timestamp"])
        src = row.get("src_account")
        dst = row.get("dst_account")
        merchant = row.get("merchant_id")
        amount = float(row.get("amount") or 0.0)
        weight = log1p(max(amount, 0.0))
        _add_node(nodes, "account", src)
        _add_node(nodes, "account", dst)
        if src:
            first_seen[src] = min(first_seen.get(src, ts), ts)
            amount_log[("account", src)] = amount_log.get(("account", src), 0.0) + weight
        if dst:
            first_seen[dst] = min(first_seen.get(dst, ts), ts)
            amount_log[("account", dst)] = amount_log.get(("account", dst), 0.0) + weight
        if src and dst:
            add_edge(TypedEdge("account", src, "transfers_to", "account", dst, ts, weight))
        if src and merchant:
            _add_node(nodes, "merchant", merchant)
            add_edge(TypedEdge("account", src, "pays_merchant", "merchant", merchant, ts, weight))

    for row in telemetry_rows:
        ts = int(row["timestamp"])
        account = row.get("account_id")
        device = row.get("device_id")
        ip_id = row.get("ip_id")
        session = row.get("session_id")
        _add_node(nodes, "account", account)
        _add_node(nodes, "device", device)
        _add_node(nodes, "ip", ip_id)
        _add_node(nodes, "session", session)
        if account:
            first_seen[account] = min(first_seen.get(account, ts), ts)
        if account and device:
            add_edge(TypedEdge("account", account, "login_device", "device", device, ts, 1.0))
        if device and ip_id:
            add_edge(TypedEdge("device", device, "login_ip", "ip", ip_id, ts, 1.0))
        if session and account:
            add_edge(TypedEdge("session", session, "session_to_account", "account", account, ts, 1.0))

    node_types = {node_type: sorted(ids) for node_type, ids in nodes.items()}
    features: Dict[Tuple[str, str], np.ndarray] = {}
    for node_type, ids in node_types.items():
        for node_id in ids:
            key = (node_type, node_id)
            vals = dict(raw_features.get(key, {}))
            vals["in_degree"] = in_degree.get(key, 0.0)
            vals["out_degree"] = out_degree.get(key, 0.0)
            vals["total_amount_log"] = amount_log.get(key, 0.0)
            features[key] = np.array([vals.get(name, 0.0) for name in feature_names], dtype=np.float32)

    return GraphDataset(
        node_types=node_types,
        features=features,
        feature_names=feature_names,
        edges=edges,
        account_labels=labels,
        account_first_seen=first_seen,
    )
