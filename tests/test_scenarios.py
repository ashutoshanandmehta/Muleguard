import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.operational import build_parser, run


SCENARIOS = {
    "digital_arrest": "MULE_DA_01",
    "phishing_upi": "MULE_PU_01",
    "loan_app": "MULE_LA_01",
    "betting_crypto": "MULE_BC_01",
}


class ScenarioSmokeTest(unittest.TestCase):
    def test_scenario_top_risk_accounts(self):
        for scenario, expected_top in SCENARIOS.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                base = Path("data/scenarios") / scenario
                report_path = Path(tmp) / "report.json"
                args = build_parser().parse_args([
                    "--transactions", str(base / "muleguard_core_transactions.csv"),
                    "--telemetry", str(base / "muleguard_digital_telemetry.csv"),
                    "--entity-map", str(base / "muleguard_entity_map_full.csv"),
                    "--node-features", str(base / "muleguard_node_features_full.csv"),
                    "--alerts-out", str(Path(tmp) / "alerts.csv"),
                    "--audit-log", str(Path(tmp) / "audit.jsonl"),
                    "--report-out", str(report_path),
                ])
                alerts = run(args)
                report = json.loads(report_path.read_text(encoding="utf-8"))

                self.assertGreaterEqual(len(alerts), 1)
                self.assertEqual(report["top_risky_accounts"][0]["account_id"], expected_top)


if __name__ == "__main__":
    unittest.main()
