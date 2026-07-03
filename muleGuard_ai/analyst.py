from .models import Subgraph

class AnalystInterface:
    def create_case(self, focal: str, score: float, action: str, explanation: dict, subgraph: Subgraph) -> str:
        return f"CASE-{abs(hash((focal, score, action)))%10_000_000}"

    def build_explanation(self, gnn: float, ts: float, rule: float, subgraph: Subgraph) -> dict:
        return {
            "contributors": {"gnn": gnn, "timeseries": ts, "rule": rule},
            "edges_sample": [(e.src, e.kind, e.dst, round(e.weight, 4)) for e in subgraph.edges[:10]]
        }
