import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.build_features import build_features
from muleGuard_ai.tune_gnn import expand_graph_views, expand_grid, filter_grid, promotion_decision, build_parser, tune, valid_metric_names


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_dataset(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir()
    _write(
        data_dir / "muleguard_core_transactions.csv",
        "event_id,kind,timestamp,amount,currency,src_account,dst_account,merchant_id,channel,auth_result,txn_status\n"
        "E1,transaction,100,100,INR,A,B,,UPI,SUCCESS,POSTED\n"
        "E2,transaction,200,120,INR,C,A,,UPI,SUCCESS,POSTED\n"
        "E3,transaction,300,130,INR,A,D,,UPI,SUCCESS,POSTED\n"
        "E4,transaction,400,50,INR,E,F,,UPI,SUCCESS,POSTED\n"
        "E5,transaction,500,60,INR,F,E,,UPI,SUCCESS,POSTED\n"
        "E6,transaction,600,70,INR,G,H,,UPI,SUCCESS,POSTED\n",
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
        "entity_type,entity_id,txn_velocity_24h,recency_decay,ts_anomaly,rule_uplift,is_mule\n"
        "account,A,3,1,0.9,0.3,1\n"
        "account,B,1,1,0.2,0.0,0\n"
        "account,C,1,1,0.2,0.0,0\n"
        "account,D,1,1,0.2,0.0,0\n"
        "account,E,2,1,0.8,0.2,1\n"
        "account,F,2,1,0.7,0.1,1\n"
        "account,G,1,1,0.1,0.0,0\n"
        "account,H,1,1,0.1,0.0,0\n",
    )
    return data_dir


class TuneGNNTest(unittest.TestCase):
    def test_grid_sizes(self):
        self.assertEqual(len(expand_grid(smoke=True)), 2)
        self.assertEqual(len(expand_grid(smoke=False)), 216)

    def test_promotion_decision(self):
        self.assertEqual(promotion_decision(0.31, 0.20, 0.03, 0.25, 4.0, {"lift_at_5pct": 4.2}), "PROMOTE_GNN")
        self.assertEqual(promotion_decision(0.23, 0.18, 0.03, 0.25, 4.0, {"lift_at_5pct": 4.2}), "NEEDS_MORE_DATA")
        self.assertEqual(promotion_decision(0.10, 0.18, 0.03, 0.25, 4.0, {"lift_at_5pct": 4.2}), "KEEP_TABULAR")
        self.assertEqual(promotion_decision(0.31, None, 0.03, 0.25, 4.0, {"lift_at_5pct": 4.2}), "NEEDS_MORE_DATA")

    def test_valid_metric_names_include_cutoffs(self):
        names = valid_metric_names([0.01, 0.025, 0.05])
        self.assertIn("capture_at_1pct", names)
        self.assertIn("lift_at_2_5pct", names)
        self.assertIn("precision_at_5pct", names)
        self.assertIn("pr_auc", names)

    def test_filter_grid(self):
        configs = filter_grid(
            expand_grid(smoke=False),
            architectures=["edge_transformer"],
            losses=["focal"],
            hidden_channels=[32],
            layers=[3],
            dropouts=[0.4],
            learning_rates=[0.005],
        )
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].architecture, "edge_transformer")
        self.assertEqual(configs[0].loss, "focal")
        self.assertEqual(configs[0].hidden_channels, 32)

    def test_expand_requested_graph_view(self):
        configs = expand_graph_views(expand_grid(smoke=True), ["account_only"])
        self.assertTrue(configs)
        self.assertTrue(all(config.graph_view == "account_only" for config in configs))

    def test_invalid_metric_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _make_dataset(Path(tmp))
            args = build_parser().parse_args([
                "--data", str(data),
                "--output", str(Path(tmp) / "out.json"),
                "--tabular-report", str(Path(tmp) / "missing.json"),
                "--smoke",
                "--runs", "1",
                "--epochs", "1",
                "--metric", "capture_at_5",
            ])
            with self.assertRaisesRegex(RuntimeError, "Invalid metric"):
                tune(args)

    def test_invalid_selection_metric_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _make_dataset(Path(tmp))
            args = build_parser().parse_args([
                "--data", str(data),
                "--output", str(Path(tmp) / "out.json"),
                "--tabular-report", str(Path(tmp) / "missing.json"),
                "--smoke",
                "--runs", "1",
                "--epochs", "1",
                "--metric", "capture_at_50pct",
                "--selection-metric", "queue_magic",
                "--cutoffs", "0.5",
            ])
            with self.assertRaisesRegex(RuntimeError, "Invalid selection metric"):
                tune(args)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"),
        "PyTorch and PyTorch Geometric are not installed",
    )
    def test_tune_gnn_smoke_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _make_dataset(Path(tmp))
            enhanced = Path(tmp) / "enhanced"
            build_features(str(source), str(enhanced), rapid_window_seconds=100)
            output = Path(tmp) / "gnn_smoke.json"
            checkpoint_dir = Path(tmp) / "models"
            args = build_parser().parse_args([
                "--data", str(enhanced),
                "--output", str(output),
                "--checkpoint-dir", str(checkpoint_dir),
                "--tabular-report", str(Path(tmp) / "missing_tabular.json"),
                "--smoke",
                "--runs", "1",
                "--epochs", "1",
                "--patience", "1",
                "--metric", "capture_at_50pct",
                "--selection-metric", "pr_auc",
                "--cutoffs", "0.5",
            ])

            result = tune(args)

            self.assertTrue(output.exists())
            self.assertEqual(result["config_count"], 2)
            self.assertEqual(result["selection_metric"], "pr_auc")
            self.assertIn(result["promotion_decision"], {"KEEP_TABULAR", "NEEDS_MORE_DATA", "PROMOTE_GNN"})
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["selection_metric"], "pr_auc")
            self.assertIn("candidates", written)


if __name__ == "__main__":
    unittest.main()
