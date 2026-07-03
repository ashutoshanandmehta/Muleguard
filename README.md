# MuleGuard AI

MuleGuard AI is a local CLI package for bank-side mule-account detection. It builds a heterogeneous account graph, scores accounts with a baseline graph-risk model or a trained GNN checkpoint, exports analyst alerts, writes audit logs, and evaluates labeled AMLSim data.

This version is intentionally scoped to Phase 0-2:

- operational batch scoring
- scenario demos
- AMLSim conversion
- metrics-based evaluation
- local CLI deployment

Dashboard, REST API, RBIH/I4C submission, federated learning, and live bank integrations are placeholders for future versions.

## Quickstart (reproducible)

Requires Python 3.9-3.13 (this repo is exactly tested on 3.9.6; `requirements-lock.txt`
pins that environment).

```bash
python3 -m venv .venv-ml
.venv-ml/bin/pip install -r requirements.txt   # or requirements-lock.txt for exact pins

# 1. Demo + test suite (36 tests)
.venv-ml/bin/python demo_runner.py
.venv-ml/bin/python -m unittest discover -s tests

# 2. Rebuild engineered features from the committed AMLSim 1K conversion
.venv-ml/bin/python -m muleGuard_ai.build_features \
  --data data/amlsim_1k --output data/amlsim_1k_features

# 3. Train the tabular baseline and the hybrid tabular-teacher + graph-residual GNN
.venv-ml/bin/python -m muleGuard_ai.train_baseline_model \
  --data data/amlsim_1k_features \
  --metrics-out runtime/reports/tabular_1k_metrics.json
.venv-ml/bin/python -m muleGuard_ai.train_gnn \
  --transactions data/amlsim_1k_features/muleguard_core_transactions.csv \
  --telemetry data/amlsim_1k_features/muleguard_digital_telemetry.csv \
  --entity-map data/amlsim_1k_features/muleguard_entity_map_full.csv \
  --node-features data/amlsim_1k_features/muleguard_node_features_full.csv \
  --use-tabular-teacher \
  --output models/hybrid_1k.pt

# 4. Evaluate the checkpoint with analyst-queue metrics
.venv-ml/bin/python -m muleGuard_ai.evaluate \
  --data data/amlsim_1k_features \
  --checkpoint models/hybrid_1k.pt \
  --output runtime/reports/hybrid_1k_metrics.json
```

`data/amlsim_1k/` is a committed AMLSim 1K conversion so the full pipeline runs out of the
box. It is smoke-scale: its time-based test slice holds only 6 positive accounts, so treat
1K metrics as plumbing checks. The 10K/100K experiments in
`HYBRID_TEACHER_FINDINGS.md` require generating AMLSim data externally
(https://github.com/IBM/AMLSim) and converting it with `muleGuard_ai.convert_amlsim` (see
below).

## Commands

Run the small built-in demo:

```bash
python demo_runner.py
```

Install locally for development:

```bash
pip install -e .
```

Run operational baseline scoring:

```bash
python -m muleGuard_ai.operational
```

Outputs:

- `runtime/alerts/account_alerts.csv`
- `runtime/audit/audit.jsonl`
- `runtime/reports/run_report.json`

Run operational scoring with all accounts included:

```bash
python -m muleGuard_ai.operational --include-allow
```

Use the kill switch:

```bash
python -m muleGuard_ai.operational --kill-switch
```

Train a GNN checkpoint:

```bash
pip install -r requirements.txt
python -m muleGuard_ai.train_gnn --epochs 50 --output models/account_graphsage.pt
```

GNN training uses normalized node tensors, stratified labeled splits, residual heterogeneous message passing, balanced class weighting, a small pairwise ranking loss, best-validation checkpointing, early stopping, and gradient clipping by default. Available architectures are `hetero_sage` and edge-aware `gatv2`.

Graph views (`--graph-view`): `full` (all entity types, direct `transfers_to` account edges), `account_only`, and `transaction`, which reifies each transfer as a transaction node (`account -> transaction -> account`) carrying `[log_amount, log_day_offset, log_gap_since_src_prev_out, log_gap_since_dst_prev_in]` — the burst-timing structure that direct account edges cannot express. Checkpoints record their graph view and inference rebuilds the same view automatically.

Train an edge-aware GATv2 checkpoint:

```bash
python -m muleGuard_ai.train_gnn \
  --transactions runtime/data/amlsim_1k_features/muleguard_core_transactions.csv \
  --telemetry runtime/data/amlsim_1k_features/muleguard_digital_telemetry.csv \
  --entity-map runtime/data/amlsim_1k_features/muleguard_entity_map_full.csv \
  --node-features runtime/data/amlsim_1k_features/muleguard_node_features_full.csv \
  --architecture gatv2 \
  --loss focal \
  --ranking-loss-weight 0.1 \
  --validation-metric capture_at_5pct \
  --output models/account_gatv2.pt
```

Train a hybrid tabular-teacher + graph-residual checkpoint:

```bash
python -m muleGuard_ai.train_gnn \
  --transactions runtime/data/amlsim_1k_features/muleguard_core_transactions.csv \
  --telemetry runtime/data/amlsim_1k_features/muleguard_digital_telemetry.csv \
  --entity-map runtime/data/amlsim_1k_features/muleguard_entity_map_full.csv \
  --node-features runtime/data/amlsim_1k_features/muleguard_node_features_full.csv \
  --use-tabular-teacher \
  --teacher-alpha 1.0 \
  --teacher-model numpy_logistic \
  --ranking-loss-weight 0.2 \
  --output models/account_hybrid.pt
```

The hybrid mode trains a tabular teacher on the GNN train split only (train rows receive
K-fold out-of-fold scores, so the teacher feature is non-leaky), then adds
`alpha * teacher_logit` to the GNN's class-1 logit during training and inference. The GNN
therefore starts at the tabular baseline and only has to learn residual graph lift.
`--teacher-model` accepts `numpy_logistic`, `logistic`, `random_forest`, or
`gradient_boosting` (sklearn models require scikit-learn). Checkpoints store the fitted
teacher, and reports include `teacher_test_metrics` — the honest baseline for the
`+0.03` promotion comparison. See `HYBRID_TEACHER_PLAN.md` for the full design and
`HYBRID_TEACHER_FINDINGS.md` for the AMLSim 100K evaluation: residual lift is positive on
every seed but ~+0.003, structure-independent, and absent over a stronger teacher — on
AMLSim 100K the engineered features already encode the graph, so the honest promotion
outcome there is `KEEP_TABULAR` and the hybrid harness is the tool to re-run on richer
data (1M / real bank data).

Run structural GNN tuning:

```bash
python -m muleGuard_ai.tune_gnn \
  --data runtime/data/amlsim_1k_features \
  --runs 3 \
  --metric capture_at_5pct \
  --output runtime/reports/gnn_model_selection.json
```

Run a small GNN tuning smoke:

```bash
python -m muleGuard_ai.tune_gnn \
  --data runtime/data/amlsim_sample_features \
  --smoke \
  --output runtime/reports/gnn_smoke_selection.json
```

The tuner tests `hetero_sage` and `gatv2`, summarizes mean/std metrics across seeds, compares against the tabular baseline report when available, and writes a decision of `PROMOTE_GNN`, `KEEP_TABULAR`, or `NEEDS_MORE_DATA`. It rejects unknown metric names before training so tuning does not silently optimize an empty metric. Passing `--use-tabular-teacher` (with `--teacher-alpha`, `--teacher-model`, `--teacher-cv-folds`) applies the hybrid teacher to every config in the sweep; sweep `alpha` by running the tuner once per value with a distinct `--output` and `--checkpoint-dir`.

Run a targeted edge-aware sweep after the smoke passes:

```bash
python -m muleGuard_ai.tune_gnn \
  --data runtime/data/amlsim_1k_features \
  --runs 3 \
  --epochs 8 \
  --patience 3 \
  --architectures gatv2 \
  --losses cross_entropy,focal \
  --hidden-channels-grid 16,32 \
  --layers-grid 2 \
  --dropout-grid 0.2 \
  --lr-grid 0.005,0.01 \
  --metric capture_at_5pct \
  --output runtime/reports/gnn_model_selection_gatv2_best_3seed.json
```

To disable class weighting:

```bash
python -m muleGuard_ai.train_gnn --class-weighting none
```

Score with a GNN checkpoint:

```bash
python -m muleGuard_ai.operational \
  --checkpoint models/account_graphsage.pt \
  --model-version gnn-v1
```

## AMLSim Conversion

AMLSim is expected outside this repo at:

```text
/Users/ashutoshanand/AMLSim
```

Convert the included AMLSim sample outputs:

```bash
python -m muleGuard_ai.convert_amlsim \
  --input /Users/ashutoshanand/AMLSim/sample/outputs \
  --output runtime/data/amlsim_sample
```

Converted files:

- `runtime/data/amlsim_sample/muleguard_core_transactions.csv`
- `runtime/data/amlsim_sample/muleguard_entity_map_full.csv`
- `runtime/data/amlsim_sample/muleguard_node_features_full.csv`
- `runtime/data/amlsim_sample/muleguard_digital_telemetry.csv`

Score converted AMLSim data:

```bash
python -m muleGuard_ai.operational \
  --transactions runtime/data/amlsim_sample/muleguard_core_transactions.csv \
  --telemetry runtime/data/amlsim_sample/muleguard_digital_telemetry.csv \
  --entity-map runtime/data/amlsim_sample/muleguard_entity_map_full.csv \
  --node-features runtime/data/amlsim_sample/muleguard_node_features_full.csv
```

## Evaluation

Evaluate the baseline scorer:

```bash
python -m muleGuard_ai.evaluate \
  --data runtime/data/amlsim_sample \
  --output runtime/reports/evaluation_metrics.json
```

Evaluate a GNN checkpoint:

```bash
python -m muleGuard_ai.evaluate \
  --data runtime/data/amlsim_sample \
  --checkpoint models/account_graphsage.pt \
  --output runtime/reports/evaluation_metrics.json
```

Metrics include:

- precision
- recall
- F1
- confusion matrix
- top-k recall
- capture at configured review cutoffs
- precision at configured review cutoffs
- lift at configured review cutoffs
- KS statistic
- PR-AUC when labels contain both positive and negative classes

Generate full model-quality reports:

```bash
python -m muleGuard_ai.evaluate \
  --data runtime/data/amlsim_1k \
  --checkpoint models/amlsim-1k-smoke.pt \
  --output runtime/reports/amlsim_1k_quality_metrics.json \
  --cutoffs 0.01,0.02,0.05 \
  --calibrate-threshold \
  --deciles-out runtime/reports/amlsim_1k_deciles.csv \
  --error-analysis-out runtime/reports/amlsim_1k_error_analysis.json \
  --typology-report-out runtime/reports/amlsim_1k_typology_metrics.json
```

Train the dependency-light tabular benchmark:

```bash
python -m muleGuard_ai.train_baseline_model \
  --data runtime/data/amlsim_1k \
  --output models/tabular_baseline.pkl \
  --metrics-out runtime/reports/tabular_baseline_metrics.json \
  --cutoffs 0.01,0.02,0.05
```

Build enhanced account features:

```bash
python -m muleGuard_ai.build_features \
  --data runtime/data/amlsim_1k \
  --output runtime/data/amlsim_1k_features
```

Report feature quality:

```bash
python -m muleGuard_ai.feature_quality \
  --data runtime/data/amlsim_1k_features \
  --output runtime/reports/feature_quality.json
```

Run multi-model tabular selection:

```bash
python -m muleGuard_ai.train_baseline_model \
  --data runtime/data/amlsim_1k_features \
  --select-best \
  --runs 3 \
  --metric capture_at_5pct \
  --metrics-out runtime/reports/model_selection.json \
  --cutoffs 0.01,0.02,0.05
```

Candidate models are `numpy_logistic`, `logistic`, `random_forest`, and `gradient_boosting`. Selection uses the mean configured metric across runs, then saves the best run for the winning model. If scikit-learn is unavailable, the CLI keeps the NumPy logistic fallback and records skipped candidates.

The main operating metric for model quality is ranking performance in the analyst queue: `capture_at_5pct`, `precision_at_5pct`, and `lift_at_5pct`. F1, recall, and PR-AUC remain useful diagnostics, but they are not the release gate for this fraud workflow.

The AMLSim sample is tiny and should be used only for smoke testing. For credible Phase 2 metrics, generate AMLSim `1K` data from:

```text
/Users/ashutoshanand/AMLSim/paramFiles/1K/conf.json
```

The converter also accepts AMLSim generator temp outputs, which is useful when the Python graph generator completes but the Java simulator stage is unavailable:

```bash
python -m muleGuard_ai.convert_amlsim \
  --input /Users/ashutoshanand/AMLSim/tmp/1K \
  --output runtime/data/amlsim_1k

python -m muleGuard_ai.evaluate \
  --data runtime/data/amlsim_1k \
  --output runtime/reports/amlsim_1k_metrics.json
```

On the current generated `1K` run, the corrected non-leaky baseline produced weak but reproducible first-pass metrics. With calibrated thresholding, the baseline reached `capture_at_5pct=0.109589`, `precision_at_5pct=0.109589`, and `lift_at_5pct=2.170764` across `1,446` accounts and `73` positives. After normalized GNN inputs, residual heterogeneous message passing, ranking loss, and targeted GATv2/focal tuning, the best targeted 3-seed GNN report reached mean `capture_at_5pct=0.132420`; its best seed reached `capture_at_5pct=0.136986` and `lift_at_5pct=2.713455`. That is meaningful progress, but the report still returns `NEEDS_MORE_DATA` because the project promotion gate remains `capture_at_5pct >= 0.25` and `lift_at_5pct >= 4.0`.

## Scenario Fixtures

Tracked scenario fixtures live under `data/scenarios/`:

- `digital_arrest`
- `phishing_upi`
- `loan_app`
- `betting_crypto`

Each scenario can be scored by pointing `muleGuard_ai.operational` at that folder's four MuleGuard CSVs.

## Governance And Outputs

Implemented:

- alert CSV/JSONL export
- JSONL audit logging
- run report JSON
- kill switch
- manual override CSV
- model version in alerts and audit records
- placeholder adapters for RBIH/MuleHunter, I4C/NCRP, and federated learning

Alert fields:

- `account_id`
- `score`
- `action`
- `priority`
- `status`
- `model_version`
- `contributors`
- `evidence`

## Tests

Run lightweight tests (requires an interpreter with NumPy — the package imports it at module level):

```bash
.venv-ml/bin/python -m unittest discover -s tests
```

When PyTorch and PyTorch Geometric are installed (as in `.venv-ml`), the GNN smoke tests and hybrid-teacher tests also run; on a NumPy-only interpreter they are skipped automatically.

## Important Caveat

The checked-in CSVs and AMLSim sample outputs are smoke-test data only. Do not claim real detection performance from them. Real model-quality claims should start only after AMLSim `1K` conversion and evaluation pass with reproducible metrics.
