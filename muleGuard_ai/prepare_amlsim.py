import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict


def patch_networkx_generator(generator_text: str) -> str:
    """Patch AMLSim's NetworkX 1.x assumptions for modern NetworkX runtimes."""
    generator_text = generator_text.replace(
        "import networkx as nx\n",
        "import networkx as nx\n"
        "if not hasattr(nx.Graph, 'node'):\n"
        "    nx.Graph.node = property(lambda self: self._node)\n"
        "    nx.DiGraph.node = property(lambda self: self._node)\n"
        "if not hasattr(nx.Graph, 'edge'):\n"
        "    nx.Graph.edge = property(lambda self: self._adj)\n"
        "    nx.DiGraph.edge = property(lambda self: self._adj)\n",
    )
    generator_text = generator_text.replace(
        "        nodes = self.g.nodes()\n        for src_i, dst_i in self.g.edges():\n            src = nodes[src_i]\n            dst = nodes[dst_i]\n",
        "        nodes = list(self.g.nodes())\n        for src_i, dst_i in self.g.edges():\n            src = nodes[src_i]\n            dst = nodes[dst_i]\n",
    )
    generator_text = generator_text.replace("            sub_g.add_node(_acct, attr_dict)\n", "            sub_g.add_node(_acct, **attr_dict)\n")
    generator_text = generator_text.replace("                tid = attr['edge_id']\n", "                tid = attr.get('edge_id', self.edge_id)\n")
    generator_text = generator_text.replace("                if attr['active']:\n", "                if attr.get('active', False):\n")
    generator_text = generator_text.replace("        nx.set_edge_attributes(self.g, 'active', False)\n", "        nx.set_edge_attributes(self.g, False, 'active')\n")
    generator_text = generator_text.replace("            nx.set_edge_attributes(subgraph, 'active', True)\n", "            nx.set_edge_attributes(subgraph, True, 'active')\n")
    return generator_text


def _csv_count(path: Path, count_column: str) -> int:
    with path.open(newline="") as handle:
        return sum(int(row[count_column]) for row in csv.DictReader(handle))


def _repair_account_count_to_degree_multiple(params_dir: Path) -> Dict[str, int]:
    accounts_path = params_dir / "accounts.csv"
    degree_path = params_dir / "degree.csv"
    account_total = _csv_count(accounts_path, "count")
    degree_total = _csv_count(degree_path, "Count")
    if degree_total <= 0:
        raise RuntimeError(f"Invalid AMLSim degree count in {degree_path}")
    if account_total % degree_total == 0:
        return {
            "account_total_before": account_total,
            "degree_total": degree_total,
            "account_total_after": account_total,
        }

    target_total = degree_total * math.ceil(account_total / degree_total)
    with accounts_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows or "count" not in fieldnames:
        raise RuntimeError(f"Invalid AMLSim accounts file: {accounts_path}")

    scaled_counts = []
    running = 0
    for index, row in enumerate(rows):
        original = int(row["count"])
        if index == len(rows) - 1:
            adjusted = target_total - running
        else:
            adjusted = int(round(original * target_total / account_total))
            running += adjusted
        scaled_counts.append(max(0, adjusted))
    if sum(scaled_counts) != target_total:
        scaled_counts[-1] += target_total - sum(scaled_counts)

    for row, adjusted in zip(rows, scaled_counts):
        row["count"] = str(adjusted)
    with accounts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "account_total_before": account_total,
        "degree_total": degree_total,
        "account_total_after": target_total,
    }


def prepare_amlsim_run(amlsim_root: str, size: str, output_root: str, repair_degree_multiple: bool = False) -> Dict[str, str]:
    source_root = Path(amlsim_root)
    source_params = source_root / "paramFiles" / size
    if not source_params.exists():
        raise RuntimeError(f"AMLSim param directory not found: {source_params}")
    source_conf = source_params / "conf.json"
    if not source_conf.exists():
        raise RuntimeError(f"AMLSim conf.json not found: {source_conf}")

    run_root = Path(output_root) / size
    params_dir = run_root / "paramFiles"
    temporal_root = run_root / "tmp"
    final_output = run_root / "outputs"
    if params_dir.exists():
        shutil.rmtree(params_dir)
    shutil.copytree(source_params, params_dir)
    degree_repair = None
    if repair_degree_multiple:
        degree_repair = _repair_account_count_to_degree_multiple(params_dir)
    temporal_root.mkdir(parents=True, exist_ok=True)
    final_output.mkdir(parents=True, exist_ok=True)

    conf = json.loads((params_dir / "conf.json").read_text(encoding="utf-8"))
    conf.setdefault("general", {})["simulation_name"] = size
    conf.setdefault("input", {})["directory"] = str(params_dir.resolve())
    conf["input"]["normal_models"] = conf["input"].get("normal_models", "normalModels.csv")
    conf.setdefault("temporal", {})["directory"] = str(temporal_root.resolve())
    conf.setdefault("output", {})["directory"] = str(final_output.resolve())

    prepared_conf = run_root / "conf.json"
    prepared_conf.write_text(json.dumps(conf, indent=2, sort_keys=True), encoding="utf-8")
    source_generator = source_root / "scripts" / "transaction_graph_generator.py"
    patched_generator = run_root / "transaction_graph_generator_patched.py"
    patched_generator.write_text(patch_networkx_generator(source_generator.read_text(encoding="utf-8")), encoding="utf-8")
    pythonpath = str(source_root / "scripts")
    manifest = {
        "size": size,
        "amlsim_root": str(source_root),
        "source_param_dir": str(source_params),
        "run_root": str(run_root),
        "prepared_conf": str(prepared_conf),
        "param_dir": str(params_dir),
        "temporal_output": str(temporal_root / size),
        "final_output": str(final_output),
        "generator_command": f"PYTHONPATH={pythonpath} python {patched_generator} {prepared_conf}",
        "generator_pythonpath": pythonpath,
        "patched_generator": str(patched_generator),
        "convert_command": f"python -m muleGuard_ai.convert_amlsim --input {temporal_root / size} --output runtime/data/amlsim_{size.lower()}",
    }
    if degree_repair is not None:
        manifest["degree_multiple_repair"] = degree_repair
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a workspace-local AMLSim run config without mutating AMLSim.")
    parser.add_argument("--amlsim-root", default="/Users/ashutoshanand/AMLSim")
    parser.add_argument("--size", default="10K")
    parser.add_argument("--output-root", default="runtime/amlsim_runs")
    parser.add_argument("--repair-degree-multiple", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_amlsim_run(args.amlsim_root, args.size, args.output_root, args.repair_degree_multiple)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
