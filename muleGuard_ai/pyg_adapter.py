from math import ceil
from typing import Dict, Tuple

import numpy as np

from .graph_dataset import GraphDataset


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    clean = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = clean.mean(axis=0, keepdims=True)
    std = clean.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((clean - mean) / std).astype(np.float32)


def _normalize_edge_attrs(attrs: list) -> np.ndarray:
    return _normalize_matrix(np.array(attrs, dtype=np.float32))


def _transaction_node_features(transfer_edges, min_timestamp: int) -> np.ndarray:
    """Per-transaction features: amount, recency, and in/out burst gaps.

    ``gap_src`` is the time since the same source account's previous outgoing
    transfer; ``gap_dst`` the time since the destination's previous incoming
    transfer. Rapid in-out mule behavior shows up as small gaps on both sides.
    First transfers get -1.0 sentinels (distinct from a zero-second gap).
    """
    by_src: Dict[str, list] = {}
    by_dst: Dict[str, list] = {}
    for idx, edge in enumerate(transfer_edges):
        by_src.setdefault(edge.src_id, []).append(idx)
        by_dst.setdefault(edge.dst_id, []).append(idx)
    gap_src = [-1.0] * len(transfer_edges)
    gap_dst = [-1.0] * len(transfer_edges)
    for gaps, groups in ((gap_src, by_src), (gap_dst, by_dst)):
        for idxs in groups.values():
            idxs.sort(key=lambda i: (transfer_edges[i].timestamp, i))
            previous = None
            for i in idxs:
                ts = transfer_edges[i].timestamp
                if previous is not None:
                    gaps[i] = float(np.log1p(max(ts - previous, 0)))
                previous = ts
    features = []
    for idx, edge in enumerate(transfer_edges):
        day_offset = max(edge.timestamp - min_timestamp, 0) / 86400.0 if edge.timestamp else 0.0
        features.append([edge.weight, float(np.log1p(day_offset)), gap_src[idx], gap_dst[idx]])
    return np.array(features, dtype=np.float32)


def to_pyg_heterodata(dataset: GraphDataset, graph_view: str = "full"):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and PyTorch Geometric are required for GNN training. "
            "Install the packages from requirements.txt, then rerun training."
        ) from exc

    data = HeteroData()
    node_index: Dict[Tuple[str, str], int] = {}

    if graph_view not in {"full", "account_only", "transaction"}:
        raise RuntimeError(f"Unsupported graph view: {graph_view}")
    included_node_types = ["account"] if graph_view == "account_only" else list(dataset.node_types.keys())

    for node_type in included_node_types:
        node_ids = dataset.node_types.get(node_type, [])
        for idx, node_id in enumerate(node_ids):
            node_index[(node_type, node_id)] = idx
        matrix = [dataset.features[(node_type, node_id)] for node_id in node_ids]
        data[node_type].x = torch.tensor(_normalize_matrix(np.array(matrix)), dtype=torch.float)

    edge_timestamps = [edge.timestamp for edge in dataset.edges if edge.timestamp > 0]
    min_timestamp = min(edge_timestamps) if edge_timestamps else 0
    edge_groups = {}
    transfer_edges = []
    for edge in dataset.edges:
        if edge.src_type not in included_node_types or edge.dst_type not in included_node_types:
            continue
        if graph_view == "account_only" and (edge.src_type, edge.relation, edge.dst_type) != ("account", "transfers_to", "account"):
            continue
        if graph_view == "transaction" and (edge.src_type, edge.relation, edge.dst_type) == ("account", "transfers_to", "account"):
            # Reified as account -> transaction -> account below.
            transfer_edges.append(edge)
            continue
        key = (edge.src_type, edge.relation, edge.dst_type)
        edge_groups.setdefault(key, ([], [], [], []))
        src_list, dst_list, weights, attrs = edge_groups[key]
        src_idx = node_index[(edge.src_type, edge.src_id)]
        dst_idx = node_index[(edge.dst_type, edge.dst_id)]
        src_list.append(src_idx)
        dst_list.append(dst_idx)
        weights.append(edge.weight)
        day_offset = max(edge.timestamp - min_timestamp, 0) / 86400.0 if edge.timestamp else 0.0
        attrs.append([edge.weight, np.log1p(day_offset), 1.0])
        rev_key = (edge.dst_type, f"rev_{edge.relation}", edge.src_type)
        edge_groups.setdefault(rev_key, ([], [], [], []))
        rev_src_list, rev_dst_list, rev_weights, rev_attrs = edge_groups[rev_key]
        rev_src_list.append(dst_idx)
        rev_dst_list.append(src_idx)
        rev_weights.append(edge.weight)
        rev_attrs.append([edge.weight, np.log1p(day_offset), -1.0])

    if graph_view == "transaction" and transfer_edges:
        data["transaction"].x = torch.tensor(
            _normalize_matrix(_transaction_node_features(transfer_edges, min_timestamp)),
            dtype=torch.float,
        )
        for txn_idx, edge in enumerate(transfer_edges):
            src_idx = node_index[("account", edge.src_id)]
            dst_idx = node_index[("account", edge.dst_id)]
            day_offset = max(edge.timestamp - min_timestamp, 0) / 86400.0 if edge.timestamp else 0.0
            for key, s, d, direction in (
                (("account", "sends", "transaction"), src_idx, txn_idx, 1.0),
                (("transaction", "delivers", "account"), txn_idx, dst_idx, 1.0),
                (("transaction", "rev_sends", "account"), txn_idx, src_idx, -1.0),
                (("account", "rev_delivers", "transaction"), dst_idx, txn_idx, -1.0),
            ):
                edge_groups.setdefault(key, ([], [], [], []))
                src_list, dst_list, weights, attrs = edge_groups[key]
                src_list.append(s)
                dst_list.append(d)
                weights.append(edge.weight)
                attrs.append([edge.weight, np.log1p(day_offset), direction])

    for key, (src_list, dst_list, weights, attrs) in edge_groups.items():
        data[key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[key].edge_weight = torch.tensor(weights, dtype=torch.float)
        data[key].edge_attr = torch.tensor(_normalize_edge_attrs(attrs), dtype=torch.float)

    account_ids = dataset.account_ids()
    labels = [dataset.account_labels.get(account_id, -1) for account_id in account_ids]
    data["account"].y = torch.tensor(labels, dtype=torch.long)
    labeled = [idx for idx, label in enumerate(labels) if label >= 0]
    train_mask = torch.zeros(len(account_ids), dtype=torch.bool)
    val_mask = torch.zeros(len(account_ids), dtype=torch.bool)
    test_mask = torch.zeros(len(account_ids), dtype=torch.bool)
    if len(labeled) == 1:
        train_mask[labeled] = True
        test_mask[labeled] = True
    else:
        ordered = sorted(labeled, key=lambda i: (dataset.account_first_seen.get(account_ids[i], 0), account_ids[i]))
        test_count = max(1, int(round(len(ordered) * 0.2)))
        test_count = min(test_count, len(ordered) - 1)
        val_end = len(ordered) - test_count
        train_end = max(1, int(round(len(ordered) * 0.6)))
        train_end = min(train_end, max(1, val_end - 1))
        train_mask[ordered[:train_end]] = True
        val_mask[ordered[train_end:val_end]] = True
        test_mask[ordered[val_end:]] = True
    data["account"].train_mask = train_mask
    data["account"].val_mask = val_mask
    data["account"].test_mask = test_mask
    data["account"].split_strategy = "time"
    data["account"].graph_view = graph_view
    data["account"].node_ids = account_ids
    return data
