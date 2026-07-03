import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class GovernanceConfig:
    kill_switch_enabled: bool = False
    model_version: str = "baseline-v1"
    operator: str = "system"
    manual_overrides: Dict[str, str] = field(default_factory=dict)


def load_manual_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    out: Dict[str, str] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            account_id = row.get("account_id")
            action = row.get("action")
            if account_id and action:
                out[account_id] = action
    return out


def apply_governance(account_id: str, proposed_action: str, cfg: GovernanceConfig) -> str:
    if cfg.kill_switch_enabled:
        return "HOLD"
    return cfg.manual_overrides.get(account_id, proposed_action)


class AuditLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_events(self, events: Iterable[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for event in events:
                payload = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    **event,
                }
                f.write(json.dumps(payload, sort_keys=True) + "\n")
