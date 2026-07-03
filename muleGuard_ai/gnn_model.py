def require_gnn_dependencies():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATv2Conv, HeteroConv, Linear, SAGEConv, TransformerConv
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and PyTorch Geometric are required for this module. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return torch, F, GATv2Conv, HeteroConv, Linear, SAGEConv, TransformerConv


def build_account_gnn(
    metadata,
    hidden_channels: int = 32,
    out_channels: int = 2,
    architecture: str = "hetero_sage",
    edge_dim: int = 3,
    dropout: float = 0.3,
    num_layers: int = 2,
    residual: bool = True,
    input_skip: bool = True,
    head_layers: int = 1,
):
    torch, F, GATv2Conv, HeteroConv, Linear, SAGEConv, TransformerConv = require_gnn_dependencies()
    node_types, edge_types = metadata
    if architecture not in {"hetero_sage", "gatv2", "edge_transformer"}:
        raise RuntimeError(f"Unsupported GNN architecture: {architecture}")
    if num_layers < 1:
        raise RuntimeError("num_layers must be >= 1")
    if head_layers not in {1, 2}:
        raise RuntimeError("head_layers must be 1 or 2")

    def make_conv():
        if architecture == "gatv2":
            return HeteroConv({
                edge_type: GATv2Conv(
                    (-1, -1),
                    hidden_channels,
                    heads=1,
                    concat=False,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
                for edge_type in edge_types
            }, aggr="sum")
        if architecture == "edge_transformer":
            return HeteroConv({
                edge_type: TransformerConv(
                    (-1, -1),
                    hidden_channels,
                    heads=1,
                    concat=False,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for edge_type in edge_types
            }, aggr="sum")
        return HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in edge_types
        }, aggr="sum")

    class AccountHeteroGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = architecture
            self.uses_edge_attr = architecture in {"gatv2", "edge_transformer"}
            self.residual = residual
            self.input_skip = input_skip
            self.node_lins = torch.nn.ModuleDict({
                node_type: Linear(-1, hidden_channels)
                for node_type in node_types
            })
            self.convs = torch.nn.ModuleList([make_conv() for _ in range(num_layers)])
            self.norms = torch.nn.ModuleDict({
                node_type: torch.nn.LayerNorm(hidden_channels)
                for node_type in node_types
            })
            self.dropout = torch.nn.Dropout(dropout)
            account_out_channels = hidden_channels * 2 if input_skip else hidden_channels
            if head_layers == 2:
                self.account_out = torch.nn.Sequential(
                    Linear(account_out_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    Linear(hidden_channels, out_channels),
                )
            else:
                self.account_out = Linear(account_out_channels, out_channels)

        def zero_init_head(self):
            """Zero the final head layer so initial logits are exactly zero.

            Used by the hybrid tabular-teacher mode: with a zeroed head the model
            starts at ``sigmoid(alpha * teacher_logit)`` and only learns the graph
            residual. Call after the first forward pass so lazy Linear layers are
            materialized.
            """
            last = self.account_out[-1] if isinstance(self.account_out, torch.nn.Sequential) else self.account_out
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)

        def _merge(self, previous, updated):
            out = dict(previous)
            for node_type, value in updated.items():
                if self.residual:
                    value = value + previous[node_type]
                value = self.norms[node_type](value)
                out[node_type] = self.dropout(value.relu())
            return out

        def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
            h = {
                node_type: self.dropout(self.node_lins[node_type](x).relu())
                for node_type, x in x_dict.items()
            }
            initial_account = h["account"]
            for conv in self.convs:
                if self.uses_edge_attr:
                    updated = conv(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
                else:
                    updated = conv(h, edge_index_dict)
                h = self._merge(h, updated)
            account_h = h["account"]
            if self.input_skip:
                account_h = torch.cat([account_h, initial_account], dim=-1)
            return {"account": self.account_out(account_h)}

    return AccountHeteroGNN()


def build_account_graphsage(metadata, hidden_channels: int = 32, out_channels: int = 2):
    return build_account_gnn(metadata, hidden_channels, out_channels, architecture="hetero_sage")
