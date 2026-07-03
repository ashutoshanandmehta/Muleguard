import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muleGuard_ai.train_baseline_model import _split_indices, build_parser, train


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TrainBaselineModelTest(unittest.TestCase):
    def test_default_split_is_chronological(self):
        account_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        first_seen = {account_id: idx for idx, account_id in enumerate(account_ids)}
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int64)

        train_idx, test_idx = _split_indices(account_ids, first_seen, y, test_size=0.3, seed=42, strategy="time")

        train_latest = max(first_seen[account_ids[int(idx)]] for idx in train_idx)
        test_earliest = min(first_seen[account_ids[int(idx)]] for idx in test_idx)
        self.assertLessEqual(train_latest, test_earliest)
        self.assertEqual([account_ids[int(idx)] for idx in test_idx], ["H", "I", "J"])

    def test_numpy_tabular_baseline_writes_model_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            _write(
                data_dir / "muleguard_core_transactions.csv",
                "event_id,kind,timestamp,amount,currency,src_account,dst_account,merchant_id,channel,auth_result,txn_status\n"
                "E1,transaction,1,100,INR,A,B,,UPI,SUCCESS,POSTED\n"
                "E2,transaction,2,100,INR,C,A,,UPI,SUCCESS,POSTED\n"
                "E3,transaction,3,100,INR,D,B,,UPI,SUCCESS,POSTED\n"
                "E4,transaction,4,100,INR,E,F,,UPI,SUCCESS,POSTED\n",
            )
            _write(
                data_dir / "muleguard_entity_map_full.csv",
                "account_id,customer_id,device_id,ip_id,merchant_id,session_id\n"
                "A,C_A,,,,\nB,C_B,,,,\nC,C_C,,,,\nD,C_D,,,,\nE,C_E,,,,\nF,C_F,,,,\n",
            )
            _write(
                data_dir / "muleguard_digital_telemetry.csv",
                "kind,timestamp,account_id,session_id,device_id,ip_id,geo_lat,geo_lon,vpn_proxy_flag,failed_logins,auth_method\n",
            )
            _write(
                data_dir / "muleguard_node_features_full.csv",
                "entity_type,entity_id,txn_velocity_24h,recency_decay,ts_anomaly,rule_uplift,is_mule\n"
                "account,A,3,1,0.9,0.3,1\n"
                "account,B,2,1,0.8,0.2,1\n"
                "account,C,1,1,0.2,0.0,0\n"
                "account,D,1,1,0.1,0.0,0\n"
                "account,E,1,1,0.2,0.0,0\n"
                "account,F,1,1,0.1,0.0,0\n",
            )
            model_path = Path(tmp) / "model.pkl"
            metrics_path = Path(tmp) / "metrics.json"
            args = build_parser().parse_args([
                "--data", str(data_dir),
                "--output", str(model_path),
                "--metrics-out", str(metrics_path),
                "--max-iter", "20",
                "--cutoffs", "0.5",
            ])

            result = train(args)

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertIn("metrics", result)
            self.assertEqual(result["metadata"]["split_strategy"], "time")
            self.assertLessEqual(
                result["metadata"]["split"]["train"]["first_seen_max"],
                result["metadata"]["split"]["test"]["first_seen_min"],
            )
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("capture_at_50pct", written["metrics"])


if __name__ == "__main__":
    unittest.main()
