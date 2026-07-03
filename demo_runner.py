from muleGuard_ai.csv_io import load_config_csv
from muleGuard_ai.decisioning import DecisionEngine
from muleGuard_ai.graph_dataset import build_graph_dataset
from muleGuard_ai.risk_baseline import score_accounts


def main() -> None:
    cfg = load_config_csv("muleguard_config.csv")
    decider = DecisionEngine(cfg)
    dataset = build_graph_dataset(
        "muleguard_core_transactions.csv",
        "muleguard_digital_telemetry.csv",
        "muleguard_entity_map_full.csv",
        "muleguard_node_features_full.csv",
    )

    print("MuleGuard AI - Plan B MVP demo")
    print(f"node_types={ {k: len(v) for k, v in dataset.node_types.items()} }")
    print(f"edges={len(dataset.edges)} labeled_accounts={len(dataset.labeled_accounts())}")
    print()

    for risk in score_accounts(dataset):
        action = decider.decide(risk.score)
        print(f"{risk.account_id}: score={risk.score:.4f} action={action}")
        print(f"  contributors={risk.contributors}")
        for item in risk.evidence:
            print(f"  evidence: {item}")
        print()

    print("GNN training path:")
    print("  pip install -r requirements.txt")
    print("  python -m muleGuard_ai.train_gnn --epochs 50")


if __name__ == "__main__":
    main()
