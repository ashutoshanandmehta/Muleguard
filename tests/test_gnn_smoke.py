import importlib.util
import unittest

from muleGuard_ai.graph_dataset import build_graph_dataset


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"),
    "PyTorch and PyTorch Geometric are not installed",
)
class GNNPathSmokeTest(unittest.TestCase):
    def test_pyg_adapter_and_model_forward(self):
        from muleGuard_ai.gnn_model import build_account_gnn, build_account_graphsage
        from muleGuard_ai.pyg_adapter import to_pyg_heterodata

        dataset = build_graph_dataset(
            "muleguard_core_transactions.csv",
            "muleguard_digital_telemetry.csv",
            "muleguard_entity_map_full.csv",
            "muleguard_node_features_full.csv",
        )
        data = to_pyg_heterodata(dataset)
        self.assertTrue(bool(data["account"].x.isfinite().all()))
        self.assertAlmostEqual(float(data["account"].x.mean()), 0.0, places=5)
        account_ids = data["account"].node_ids
        train_times = [
            dataset.account_first_seen.get(account_ids[idx], 0)
            for idx, keep in enumerate(data["account"].train_mask.tolist())
            if keep
        ]
        later_times = [
            dataset.account_first_seen.get(account_ids[idx], 0)
            for mask_name in ("val_mask", "test_mask")
            for idx, keep in enumerate(getattr(data["account"], mask_name).tolist())
            if keep
        ]
        if train_times and later_times:
            self.assertLessEqual(max(train_times), min(later_times))
        model = build_account_graphsage(data.metadata(), hidden_channels=8, out_channels=2)
        out = model(data.x_dict, data.edge_index_dict)["account"]

        self.assertEqual(out.shape[0], len(dataset.account_ids()))
        self.assertEqual(out.shape[1], 2)
        self.assertTrue(bool(data["account"].train_mask.any()))
        self.assertTrue(any(hasattr(data[edge_type], "edge_attr") for edge_type in data.edge_types))
        for edge_type in data.edge_types:
            self.assertTrue(bool(data[edge_type].edge_attr.isfinite().all()))

        gatv2 = build_account_gnn(data.metadata(), hidden_channels=8, out_channels=2, architecture="gatv2")
        gatv2_out = gatv2(data.x_dict, data.edge_index_dict, data.edge_attr_dict)["account"]
        self.assertEqual(gatv2_out.shape[0], len(dataset.account_ids()))
        self.assertEqual(gatv2_out.shape[1], 2)

        transformer = build_account_gnn(data.metadata(), hidden_channels=8, out_channels=2, architecture="edge_transformer")
        transformer_out = transformer(data.x_dict, data.edge_index_dict, data.edge_attr_dict)["account"]
        self.assertEqual(transformer_out.shape[0], len(dataset.account_ids()))
        self.assertEqual(transformer_out.shape[1], 2)

    def test_account_only_graph_view_keeps_transfer_edges(self):
        from muleGuard_ai.pyg_adapter import to_pyg_heterodata

        dataset = build_graph_dataset(
            "muleguard_core_transactions.csv",
            "muleguard_digital_telemetry.csv",
            "muleguard_entity_map_full.csv",
            "muleguard_node_features_full.csv",
        )
        data = to_pyg_heterodata(dataset, graph_view="account_only")

        self.assertEqual(data.node_types, ["account"])
        self.assertIn(("account", "transfers_to", "account"), data.edge_types)
        self.assertIn(("account", "rev_transfers_to", "account"), data.edge_types)
        self.assertNotIn(("customer", "owns", "account"), data.edge_types)
        self.assertEqual(data["account"].graph_view, "account_only")

    def test_pairwise_ranking_loss_samples_large_pair_sets(self):
        import torch
        from muleGuard_ai.train_gnn import _pairwise_ranking_loss

        logits = torch.randn(1200, 2)
        labels = torch.cat([torch.ones(100, dtype=torch.long), torch.zeros(1100, dtype=torch.long)])
        loss = _pairwise_ranking_loss(torch, logits, labels, max_pairs=32)

        self.assertTrue(bool(loss.isfinite()))


if __name__ == "__main__":
    unittest.main()
