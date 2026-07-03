import csv
from typing import Dict, List, Tuple
from .models import EventData, EntityMap, FeatureVector
from .config import Config, Thresholds, FusionWeights, EdgeWeightParams

def load_core_transactions(path: str) -> List[EventData]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(EventData(
                kind="transaction",
                timestamp=int(r["timestamp"]),
                amount=float(r["amount"]),
                src_account=r["src_account"] or None,
                dst_account=r["dst_account"] or None,
                merchant_id=r.get("merchant_id") or None,
                device_id=None,
                ip_id=None,
                session_id=None,
                channel=r.get("channel") or None
            ))
    return out

def load_digital_telemetry(path: str) -> List[EventData]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(EventData(
                kind=r["kind"],
                timestamp=int(r["timestamp"]),
                amount=None,
                src_account=r.get("account_id") or None,
                dst_account=None,
                merchant_id=None,
                device_id=r.get("device_id") or None,
                ip_id=r.get("ip_id") or None,
                session_id=r.get("session_id") or None,
                channel=None
            ))
    return out

def load_entity_map(path: str) -> Dict[str, EntityMap]:
    out: Dict[str, EntityMap] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            em = EntityMap(
                account_id=r["account_id"],
                customer_id=r.get("customer_id") or "",
                device_id=r.get("device_id") or None,
                ip_id=r.get("ip_id") or None,
                merchant_id=r.get("merchant_id") or None,
                session_id=r.get("session_id") or None
            )
            out[em.account_id] = em
    return out

def load_node_features(path: str) -> Dict[str, FeatureVector]:
    by_id: Dict[str, FeatureVector] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            eid = r["entity_id"]
            vals = {k: float(v) for k, v in r.items()
                    if k not in ("entity_type", "entity_id") and v not in (None, "", "NaN")}
            by_id[eid] = FeatureVector(vals)
    return by_id

def topological_inputs(
    tx_path: str,
    tel_path: str,
    emap_path: str,
    node_feats_path: str
) -> Tuple[List[EventData], List[EventData], Dict[str, EntityMap], Dict[str, FeatureVector]]:
    tx = load_core_transactions(tx_path)
    tel = load_digital_telemetry(tel_path)
    emap = load_entity_map(emap_path)
    nfe = load_node_features(node_feats_path)
    return tx, tel, emap, nfe



def load_config_csv(path: str) -> Config:
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = r["parameter"].strip()
            v = r["value"].strip()
            try:
                rows[k] = float(v)
            except ValueError:
                rows[k] = v

    thr = Thresholds(
        ALLOW_T=rows["ALLOW_T"],      # set in CSV
        STEP_UP_T=rows["STEP_UP_T"],
        HOLD_T=rows["HOLD_T"],
        BLOCK_T=rows["BLOCK_T"],
    )
    fus = FusionWeights(
        w_gnn=rows["w_gnn"],
        w_ts=rows["w_ts"],
        w_rule=rows["w_rule"],
    )
    ew = EdgeWeightParams(
        alpha=rows["alpha"],
        beta=rows["beta"],
        gamma=rows["gamma"],
        delta=rows["delta"],
    )
    cfg = Config(
        thresholds=thr,
        fusion=fus,
        edge_weight=ew,
        time_horizon_days=int(rows.get("time_horizon_days", 30)),
        model_parameters={},   # optionally add pointers here via CSV if you like
        policies={}            # optionally add rule knobs via CSV if you like
    )
    return cfg