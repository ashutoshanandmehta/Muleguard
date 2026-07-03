import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.build_features import build_features
from muleGuard_ai.feature_quality import feature_quality
from muleGuard_ai.graph_dataset import build_graph_dataset
from muleGuard_ai.train_baseline_model import build_parser, train


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_dataset(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir()
    _write(
        data_dir / "muleguard_core_transactions.csv",
        "event_id,kind,timestamp,amount,currency,src_account,dst_account,merchant_id,channel,auth_result,txn_status\n"
        "E1,transaction,100,100,INR,A,B,,UPI,SUCCESS,POSTED\n"
        "E2,transaction,150,120,INR,C,A,,UPI,SUCCESS,POSTED\n"
        "E3,transaction,180,130,INR,A,D,,UPI,SUCCESS,POSTED\n"
        "E4,transaction,86400,50,INR,E,F,,UPI,SUCCESS,POSTED\n"
        "E5,transaction,172800,60,INR,F,E,,UPI,SUCCESS,POSTED\n"
        "E6,transaction,259200,70,INR,G,H,,UPI,SUCCESS,POSTED\n",
    )
    _write(
        data_dir / "muleguard_entity_map_full.csv",
        "account_id,customer_id,device_id,ip_id,merchant_id,session_id\n"
        "A,C_A,,,,\nB,C_B,,,,\nC,C_C,,,,\nD,C_D,,,,\nE,C_E,,,,\nF,C_F,,,,\nG,C_G,,,,\nH,C_H,,,,\n",
    )
    _write(
        data_dir / "muleguard_digital_telemetry.csv",
        "kind,timestamp,account_id,session_id,device_id,ip_id,geo_lat,geo_lon,vpn_proxy_flag,failed_logins,auth_method\n",
    )
    _write(
        data_dir / "muleguard_node_features_full.csv",
        "entity_type,entity_id,txn_velocity_24h,recency_decay,ts_anomaly,rule_uplift,amlsim_typology,is_mule\n"
        "account,A,3,1,0.9,0.3,fan_out,1\n"
        "account,B,1,1,0.2,0.0,,0\n"
        "account,C,1,1,0.2,0.0,,0\n"
        "account,D,1,1,0.2,0.0,,0\n"
        "account,E,2,1,0.8,0.2,cycle,1\n"
        "account,F,2,1,0.7,0.1,cycle,1\n"
        "account,G,1,1,0.1,0.0,,0\n"
        "account,H,1,1,0.1,0.0,,0\n",
    )
    return data_dir


class FeatureEngineeringTest(unittest.TestCase):
    def test_build_features_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _make_dataset(Path(tmp))
            enhanced = Path(tmp) / "enhanced"
            summary = build_features(str(source), str(enhanced), rapid_window_seconds=100)

            self.assertEqual(summary["accounts"], 8)
            dataset = build_graph_dataset(
                str(enhanced / "muleguard_core_transactions.csv"),
                str(enhanced / "muleguard_digital_telemetry.csv"),
                str(enhanced / "muleguard_entity_map_full.csv"),
                str(enhanced / "muleguard_node_features_full.csv"),
            )
            self.assertIn("pass_through_ratio", dataset.feature_names)
            self.assertIn("rapid_in_out_count", dataset.feature_names)
            self.assertIn("two_hop_out_count", dataset.feature_names)
            self.assertNotIn("is_mule", dataset.feature_names)
            self.assertNotIn("amlsim_typology", dataset.feature_names)

    def test_feature_quality_and_model_selection_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _make_dataset(Path(tmp))
            enhanced = Path(tmp) / "enhanced"
            build_features(str(source), str(enhanced), rapid_window_seconds=100)
            quality_path = Path(tmp) / "quality.json"
            report = feature_quality(str(enhanced), str(quality_path), top_n=5)

            self.assertTrue(quality_path.exists())
            self.assertGreaterEqual(report["feature_count"], 5)
            self.assertIn("top_features", report)
            self.assertIn("pass_through_ratio", report["features"])

            model_path = Path(tmp) / "model.pkl"
            metrics_path = Path(tmp) / "selection.json"
            args = build_parser().parse_args([
                "--data", str(enhanced),
                "--output", str(model_path),
                "--metrics-out", str(metrics_path),
                "--select-best",
                "--runs", "2",
                "--metric", "capture_at_50pct",
                "--cutoffs", "0.5",
                "--max-iter", "20",
            ])
            result = train(args)

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertIn("best_model", result)
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("candidates", written)


if __name__ == "__main__":
    unittest.main()
