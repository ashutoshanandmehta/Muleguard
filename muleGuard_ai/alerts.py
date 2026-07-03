import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class Alert:
    account_id: str
    score: float
    action: str
    model_version: str
    contributors: Dict[str, float]
    evidence: List[str]
    status: str = "OPEN"

    @property
    def priority(self) -> str:
        if self.action == "BLOCK":
            return "P1"
        if self.action == "HOLD":
            return "P2"
        if self.action == "STEP_UP":
            return "P3"
        return "P4"


def export_alerts(alerts: Iterable[Alert], path: str) -> None:
    items = list(alerts)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".jsonl":
        with output.open("w", encoding="utf-8") as f:
            for alert in items:
                f.write(json.dumps(asdict(alert), sort_keys=True) + "\n")
        return

    fieldnames = [
        "account_id",
        "score",
        "action",
        "priority",
        "status",
        "model_version",
        "contributors",
        "evidence",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alert in items:
            writer.writerow({
                "account_id": alert.account_id,
                "score": alert.score,
                "action": alert.action,
                "priority": alert.priority,
                "status": alert.status,
                "model_version": alert.model_version,
                "contributors": json.dumps(alert.contributors, sort_keys=True),
                "evidence": json.dumps(alert.evidence),
            })
