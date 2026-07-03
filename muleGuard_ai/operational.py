import argparse
from typing import List

from .alerts import Alert, export_alerts
from .csv_io import load_config_csv
from .decisioning import DecisionEngine
from .gnn_inference import score_accounts_with_checkpoint
from .governance import AuditLogger, GovernanceConfig, apply_governance, load_manual_overrides
from .graph_dataset import build_graph_dataset
from .reporting import build_run_report, export_run_report
from .risk_baseline import AccountRisk, score_accounts


def build_alerts(risks: List[AccountRisk], args: argparse.Namespace) -> List[Alert]:
    cfg = load_config_csv(args.config)
    decider = DecisionEngine(cfg)
    governance = GovernanceConfig(
        kill_switch_enabled=args.kill_switch,
        model_version=args.model_version,
        operator=args.operator,
        manual_overrides=load_manual_overrides(args.manual_overrides),
    )

    alerts: List[Alert] = []
    for risk in risks:
        proposed_action = decider.decide(risk.score)
        final_action = apply_governance(risk.account_id, proposed_action, governance)
        if final_action == "ALLOW" and not args.include_allow:
            continue
        alerts.append(
            Alert(
                account_id=risk.account_id,
                score=risk.score,
                action=final_action,
                model_version=governance.model_version,
                contributors=risk.contributors,
                evidence=risk.evidence,
            )
        )
    return alerts


def run(args: argparse.Namespace) -> List[Alert]:
    dataset = build_graph_dataset(
        args.transactions,
        args.telemetry,
        args.entity_map,
        args.node_features,
    )
    if args.checkpoint:
        risks = score_accounts_with_checkpoint(dataset, args.checkpoint)
    else:
        risks = score_accounts(dataset)

    alerts = build_alerts(risks, args)
    export_alerts(alerts, args.alerts_out)
    export_run_report(
        build_run_report(alerts, args.model_version, len(dataset.account_ids())),
        args.report_out,
    )
    AuditLogger(args.audit_log).write_events([
        {
            "event_type": "account_scored",
            "account_id": alert.account_id,
            "score": alert.score,
            "action": alert.action,
            "model_version": alert.model_version,
            "operator": args.operator,
            "evidence_count": len(alert.evidence),
        }
        for alert in alerts
    ])
    return alerts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the operational MuleGuard account scoring workflow.")
    parser.add_argument("--transactions", default="muleguard_core_transactions.csv")
    parser.add_argument("--telemetry", default="muleguard_digital_telemetry.csv")
    parser.add_argument("--entity-map", default="muleguard_entity_map_full.csv")
    parser.add_argument("--node-features", default="muleguard_node_features_full.csv")
    parser.add_argument("--config", default="muleguard_config.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--alerts-out", default="runtime/alerts/account_alerts.csv")
    parser.add_argument("--audit-log", default="runtime/audit/audit.jsonl")
    parser.add_argument("--report-out", default="runtime/reports/run_report.json")
    parser.add_argument("--manual-overrides", default=None)
    parser.add_argument("--model-version", default="baseline-v1")
    parser.add_argument("--operator", default="system")
    parser.add_argument("--kill-switch", action="store_true")
    parser.add_argument("--include-allow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alerts = run(args)
    print(f"alerts_written={len(alerts)} path={args.alerts_out}")
    print(f"audit_log={args.audit_log}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
