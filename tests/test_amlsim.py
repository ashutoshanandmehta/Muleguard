import csv
import json
import tempfile
import unittest
from pathlib import Path

from muleGuard_ai.convert_amlsim import convert_amlsim
from muleGuard_ai.evaluate import evaluate
from muleGuard_ai.graph_dataset import build_graph_dataset
from muleGuard_ai.operational import build_parser, run
from muleGuard_ai.prepare_amlsim import _repair_account_count_to_degree_multiple, patch_networkx_generator


AMLSIM_SAMPLE = Path("/Users/ashutoshanand/AMLSim/sample/outputs")


class AMLSimPrepTest(unittest.TestCase):
    def test_patch_networkx_generator_fixes_nodeview_and_attribute_api(self):
        source = (
            "import networkx as nx\n"
            "        nodes = self.g.nodes()\n"
            "        for src_i, dst_i in self.g.edges():\n"
            "            src = nodes[src_i]\n"
            "            dst = nodes[dst_i]\n"
            "            sub_g.add_node(_acct, attr_dict)\n"
            "        nx.set_edge_attributes(self.g, 'active', False)\n"
            "            nx.set_edge_attributes(subgraph, 'active', True)\n"
            "                tid = attr['edge_id']\n"
            "                if attr['active']:\n"
        )

        patched = patch_networkx_generator(source)

        self.assertIn("nx.Graph.node = property", patched)
        self.assertIn("nx.Graph.edge = property", patched)
        self.assertIn("nodes = list(self.g.nodes())", patched)
        self.assertIn("sub_g.add_node(_acct, **attr_dict)", patched)
        self.assertIn("nx.set_edge_attributes(self.g, False, 'active')", patched)
        self.assertIn("nx.set_edge_attributes(subgraph, True, 'active')", patched)
        self.assertIn("tid = attr.get('edge_id', self.edge_id)", patched)
        self.assertIn("if attr.get('active', False):", patched)

    def test_repair_account_count_to_degree_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp)
            (params / "accounts.csv").write_text(
                "count,min_balance,max_balance,country,business_type,model,bank_id\n"
                "50,1,2,US,I,1,bank\n"
                "50,1,2,US,I,2,bank\n",
                encoding="utf-8",
            )
            (params / "degree.csv").write_text(
                "Count,In-degree,Out-degree\n"
                "30,1,1\n"
                "32,2,2\n",
                encoding="utf-8",
            )

            repair = _repair_account_count_to_degree_multiple(params)

            self.assertEqual(repair["account_total_before"], 100)
            self.assertEqual(repair["degree_total"], 62)
            self.assertEqual(repair["account_total_after"], 124)
            with (params / "accounts.csv").open(newline="") as handle:
                total = sum(int(row["count"]) for row in csv.DictReader(handle))
            self.assertEqual(total, 124)


@unittest.skipUnless(AMLSIM_SAMPLE.exists(), "AMLSim sample outputs are not available")
class AMLSimConversionTest(unittest.TestCase):
    def test_convert_amlsim_sample_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = convert_amlsim(str(AMLSIM_SAMPLE), tmp)
            self.assertGreater(summary["accounts"], 0)
            self.assertGreater(summary["transactions"], 0)
            self.assertGreater(summary["positive_accounts"], 0)
            for name in [
                "muleguard_core_transactions.csv",
                "muleguard_entity_map_full.csv",
                "muleguard_node_features_full.csv",
                "muleguard_digital_telemetry.csv",
            ]:
                self.assertTrue((Path(tmp) / name).exists())

    def test_converted_sample_can_be_scored_and_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            alerts_path = Path(tmp) / "alerts.csv"
            audit_path = Path(tmp) / "audit.jsonl"
            report_path = Path(tmp) / "report.json"
            metrics_path = Path(tmp) / "metrics.json"
            convert_amlsim(str(AMLSIM_SAMPLE), str(data_dir))

            args = build_parser().parse_args([
                "--transactions", str(data_dir / "muleguard_core_transactions.csv"),
                "--telemetry", str(data_dir / "muleguard_digital_telemetry.csv"),
                "--entity-map", str(data_dir / "muleguard_entity_map_full.csv"),
                "--node-features", str(data_dir / "muleguard_node_features_full.csv"),
                "--alerts-out", str(alerts_path),
                "--audit-log", str(audit_path),
                "--report-out", str(report_path),
                "--include-allow",
            ])
            alerts = run(args)
            results = evaluate(str(data_dir), None, str(metrics_path), threshold=0.5, top_k=10)

            self.assertTrue(alerts_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertGreater(len(alerts), 0)
            self.assertIn("baseline", results)
            self.assertTrue(results["baseline"]["warnings"])

    def test_missing_labels_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for filename in [
                "muleguard_core_transactions.csv",
                "muleguard_entity_map_full.csv",
                "muleguard_digital_telemetry.csv",
            ]:
                (data_dir / filename).write_text("", encoding="utf-8")
            (data_dir / "muleguard_core_transactions.csv").write_text(
                "event_id,kind,timestamp,amount,currency,src_account,dst_account,merchant_id,channel,auth_result,txn_status\n",
                encoding="utf-8",
            )
            (data_dir / "muleguard_entity_map_full.csv").write_text(
                "account_id,customer_id,device_id,ip_id,merchant_id,session_id\nA,C,,,,\n",
                encoding="utf-8",
            )
            (data_dir / "muleguard_digital_telemetry.csv").write_text(
                "kind,timestamp,account_id,session_id,device_id,ip_id,geo_lat,geo_lon,vpn_proxy_flag,failed_logins,auth_method\n",
                encoding="utf-8",
            )
            (data_dir / "muleguard_node_features_full.csv").write_text(
                "entity_type,entity_id,txn_velocity_24h,recency_decay,ts_anomaly,rule_uplift\naccount,A,1,1,0.1,0.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "No labeled accounts"):
                evaluate(str(data_dir), None, str(data_dir / "metrics.json"), 0.5, 10)


if __name__ == "__main__":
    unittest.main()
