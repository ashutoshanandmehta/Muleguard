import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from .alerts import Alert


def build_run_report(alerts: Iterable[Alert], model_version: str, accounts_scored: int) -> dict:
    items: List[Alert] = list(alerts)
    action_counts = Counter(alert.action for alert in items)
    evidence_counts = Counter()
    for alert in items:
        for item in alert.evidence:
            evidence_counts[item] += 1

    return {
        "run_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "accounts_scored": accounts_scored,
        "alerts_written": len(items),
        "action_counts": dict(sorted(action_counts.items())),
        "top_risky_accounts": [
            {
                "account_id": alert.account_id,
                "score": alert.score,
                "action": alert.action,
                "priority": alert.priority,
            }
            for alert in sorted(items, key=lambda alert: alert.score, reverse=True)[:10]
        ],
        "top_evidence_edges": [
            {"evidence": edge, "count": count}
            for edge, count in evidence_counts.most_common(10)
        ],
    }


def export_run_report(report: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
