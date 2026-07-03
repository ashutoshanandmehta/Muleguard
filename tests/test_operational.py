import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.governance import GovernanceConfig, apply_governance
from muleGuard_ai.integrations import FederatedLearningIntegration, I4CIntegration, RBIHIntegration
from muleGuard_ai.operational import build_parser, run


class OperationalWorkflowTest(unittest.TestCase):
    def test_kill_switch_holds_non_allow_actions(self):
        cfg = GovernanceConfig(kill_switch_enabled=True)
        self.assertEqual(apply_governance("ACC123", "BLOCK", cfg), "HOLD")

    def test_manual_override_wins_without_kill_switch(self):
        cfg = GovernanceConfig(manual_overrides={"ACC123": "BLOCK"})
        self.assertEqual(apply_governance("ACC123", "STEP_UP", cfg), "BLOCK")

    def test_operational_run_writes_alerts_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            alerts_path = Path(tmp) / "alerts.csv"
            audit_path = Path(tmp) / "audit.jsonl"
            args = build_parser().parse_args([
                "--alerts-out", str(alerts_path),
                "--audit-log", str(audit_path),
            ])
            alerts = run(args)

            self.assertTrue(alerts_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertGreaterEqual(len(alerts), 1)
            first_event = json.loads(audit_path.read_text().splitlines()[0])
            self.assertEqual(first_event["event_type"], "account_scored")

    def test_future_integrations_are_explicit_placeholders(self):
        with self.assertRaises(NotImplementedError):
            RBIHIntegration().submit_alerts([])
        with self.assertRaises(NotImplementedError):
            I4CIntegration().submit_suspicious_accounts([])
        with self.assertRaises(NotImplementedError):
            FederatedLearningIntegration().train_round()


if __name__ == "__main__":
    unittest.main()
