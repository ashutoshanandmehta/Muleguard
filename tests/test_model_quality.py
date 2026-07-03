import csv
import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.evaluate import _metrics, evaluate


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class ModelQualityMetricsTest(unittest.TestCase):
    def test_cutoff_metrics_and_ks(self):
        labels = [1, 0, 1, 0, 0]
        scores = [0.95, 0.80, 0.70, 0.20, 0.10]
        metrics = _metrics(labels, scores, threshold=0.5, top_k=2, cutoffs=[0.4])

        self.assertEqual(metrics["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 2, "fn": 0})
        self.assertEqual(metrics["capture_at_40pct"], 0.5)
        self.assertEqual(metrics["precision_at_40pct"], 0.5)
        self.assertEqual(metrics["lift_at_40pct"], 1.25)
        self.assertEqual(metrics["ks"], 0.666667)

    def test_evaluate_writes_sidecar_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            _write(
                data_dir / "muleguard_core_transactions.csv",
                "event_id,kind,timestamp,amount,currency,src_account,dst_account,merchant_id,channel,auth_result,txn_status\n"
                "E1,transaction,1,100,INR,A,B,,UPI,SUCCESS,POSTED\n"
                "E2,transaction,2,100,INR,C,A,,UPI,SUCCESS,POSTED\n"
                "E3,transaction,3,100,INR,D,B,,UPI,SUCCESS,POSTED\n",
            )
            _write(
                data_dir / "muleguard_entity_map_full.csv",
                "account_id,customer_id,device_id,ip_id,merchant_id,session_id\n"
                "A,C_A,,,,\nB,C_B,,,,\nC,C_C,,,,\nD,C_D,,,,\n",
            )
            _write(
                data_dir / "muleguard_digital_telemetry.csv",
                "kind,timestamp,account_id,session_id,device_id,ip_id,geo_lat,geo_lon,vpn_proxy_flag,failed_logins,auth_method\n",
            )
            _write(
                data_dir / "muleguard_node_features_full.csv",
                "entity_type,entity_id,txn_velocity_24h,recency_decay,ts_anomaly,rule_uplift,amlsim_typology,is_mule\n"
                "account,A,3,1,0.9,0.3,fan_in,1\n"
                "account,B,3,1,0.8,0.2,fan_out,1\n"
                "account,C,1,1,0.2,0.0,,0\n"
                "account,D,1,1,0.1,0.0,,0\n",
            )
            metrics_path = Path(tmp) / "metrics.json"
            deciles_path = Path(tmp) / "deciles.csv"
            errors_path = Path(tmp) / "errors.json"
            typology_path = Path(tmp) / "typology.json"

            results = evaluate(
                str(data_dir),
                None,
                str(metrics_path),
                threshold=0.5,
                top_k=2,
                cutoffs=[0.5],
                error_analysis_out=str(errors_path),
                deciles_out=str(deciles_path),
                typology_report_out=str(typology_path),
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(deciles_path.exists())
            self.assertTrue(errors_path.exists())
            self.assertTrue(typology_path.exists())
            self.assertIn("capture_at_50pct", results["baseline"])
            with deciles_path.open(newline="", encoding="utf-8") as f:
                self.assertGreater(len(list(csv.DictReader(f))), 0)
            errors = json.loads(errors_path.read_text(encoding="utf-8"))
            self.assertIn("false_positives", errors["baseline"])
            typologies = json.loads(typology_path.read_text(encoding="utf-8"))
            self.assertIn("fan_in", typologies["baseline"])


if __name__ == "__main__":
    unittest.main()
