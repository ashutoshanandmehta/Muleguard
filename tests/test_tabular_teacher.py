import importlib.util
import unittest

import numpy as np

from muleGuard_ai.tabular_teacher import apply_teacher_logits, compute_teacher_logits


def _synthetic_matrix(seed: int = 7, count: int = 240, dims: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, dims)).astype(np.float32)
    logits = 2.5 * x[:, 0] - 1.5 * x[:, 1]
    y = (logits + rng.normal(scale=0.5, size=count) > 0).astype(np.int64)
    return x, y


class TabularTeacherTest(unittest.TestCase):
    def test_logits_shape_and_holdout_ranking(self):
        x, y = _synthetic_matrix()
        train_idx = np.arange(150)
        teacher_logit, payload = compute_teacher_logits(x, y, train_idx, seed=11)

        self.assertEqual(teacher_logit.shape, (len(x),))
        self.assertTrue(np.isfinite(teacher_logit).all())
        holdout = np.arange(150, len(x))
        pos = teacher_logit[holdout][y[holdout] == 1]
        neg = teacher_logit[holdout][y[holdout] == 0]
        self.assertGreater(pos.mean(), neg.mean())
        self.assertEqual(payload["type"], "numpy_logistic")

    def test_train_rows_use_out_of_fold_predictions(self):
        x, y = _synthetic_matrix()
        train_idx = np.arange(150)
        teacher_logit, payload = compute_teacher_logits(x, y, train_idx, folds=5, seed=11)
        full_train_logit = apply_teacher_logits(payload, x)

        holdout = np.arange(150, len(x))
        np.testing.assert_allclose(teacher_logit[holdout], full_train_logit[holdout], rtol=1e-6)
        self.assertFalse(np.allclose(teacher_logit[train_idx], full_train_logit[train_idx]))

    def test_single_class_train_slice_raises(self):
        x, y = _synthetic_matrix()
        train_idx = np.where(y == 1)[0][:20]
        with self.assertRaises(RuntimeError):
            compute_teacher_logits(x, y, train_idx)

    def test_unknown_teacher_model_raises(self):
        x, y = _synthetic_matrix()
        with self.assertRaises(RuntimeError):
            compute_teacher_logits(x, y, np.arange(150), model_type="mystery_model")


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"),
    "PyTorch and PyTorch Geometric are not installed",
)
class TeacherOffsetForwardTest(unittest.TestCase):
    def test_forward_scores_adds_class1_logit_offset(self):
        import torch
        from muleGuard_ai.gnn_model import build_account_graphsage
        from muleGuard_ai.graph_dataset import build_graph_dataset
        from muleGuard_ai.pyg_adapter import to_pyg_heterodata
        from muleGuard_ai.train_gnn import _forward, _forward_scores

        dataset = build_graph_dataset(
            "muleguard_core_transactions.csv",
            "muleguard_digital_telemetry.csv",
            "muleguard_entity_map_full.csv",
            "muleguard_node_features_full.csv",
        )
        data = to_pyg_heterodata(dataset)
        model = build_account_graphsage(data.metadata(), hidden_channels=8, out_channels=2)
        model.eval()
        with torch.no_grad():
            _forward(model, data)

        teacher_logit = torch.linspace(-2.0, 2.0, data["account"].x.shape[0])
        data["account"].teacher_logit = teacher_logit
        with torch.no_grad():
            plain = _forward(model, data)
            hybrid = _forward_scores(model, data, teacher_alpha=2.0)

        torch.testing.assert_close(hybrid[:, 0], plain[:, 0])
        torch.testing.assert_close(hybrid[:, 1], plain[:, 1] + 2.0 * teacher_logit)

    def test_zero_init_head_starts_at_teacher_anchor(self):
        import torch
        from muleGuard_ai.gnn_model import build_account_graphsage
        from muleGuard_ai.graph_dataset import build_graph_dataset
        from muleGuard_ai.pyg_adapter import to_pyg_heterodata
        from muleGuard_ai.train_gnn import _forward, _forward_scores

        dataset = build_graph_dataset(
            "muleguard_core_transactions.csv",
            "muleguard_digital_telemetry.csv",
            "muleguard_entity_map_full.csv",
            "muleguard_node_features_full.csv",
        )
        data = to_pyg_heterodata(dataset)
        model = build_account_graphsage(data.metadata(), hidden_channels=8, out_channels=2)
        model.eval()
        with torch.no_grad():
            _forward(model, data)
        model.zero_init_head()

        teacher_logit = torch.linspace(-2.0, 2.0, data["account"].x.shape[0])
        data["account"].teacher_logit = teacher_logit
        with torch.no_grad():
            plain = _forward(model, data)
            hybrid = _forward_scores(model, data, teacher_alpha=1.0)

        torch.testing.assert_close(plain, torch.zeros_like(plain))
        torch.testing.assert_close(hybrid[:, 1] - hybrid[:, 0], teacher_logit)

    def test_transaction_graph_view_reifies_transfers(self):
        import torch
        from muleGuard_ai.gnn_model import build_account_graphsage
        from muleGuard_ai.graph_dataset import build_graph_dataset
        from muleGuard_ai.pyg_adapter import to_pyg_heterodata

        dataset = build_graph_dataset(
            "muleguard_core_transactions.csv",
            "muleguard_digital_telemetry.csv",
            "muleguard_entity_map_full.csv",
            "muleguard_node_features_full.csv",
        )
        data = to_pyg_heterodata(dataset, graph_view="transaction")

        transfer_count = sum(
            1 for e in dataset.edges if (e.src_type, e.relation, e.dst_type) == ("account", "transfers_to", "account")
        )
        self.assertGreater(transfer_count, 0)
        self.assertIn("transaction", data.node_types)
        self.assertEqual(data["transaction"].x.shape, (transfer_count, 4))
        self.assertTrue(bool(data["transaction"].x.isfinite().all()))
        self.assertNotIn(("account", "transfers_to", "account"), data.edge_types)
        self.assertIn(("account", "sends", "transaction"), data.edge_types)
        self.assertIn(("transaction", "delivers", "account"), data.edge_types)
        self.assertEqual(data[("account", "sends", "transaction")].edge_index.shape[1], transfer_count)

        model = build_account_graphsage(data.metadata(), hidden_channels=8, out_channels=2)
        out = model(data.x_dict, data.edge_index_dict)["account"]
        self.assertEqual(out.shape, (len(dataset.account_ids()), 2))


if __name__ == "__main__":
    unittest.main()
