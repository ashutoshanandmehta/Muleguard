# Hybrid Tabular-Teacher + Graph-Residual: Findings (AMLSim 100K)

Date: 2026-07-03. All numbers on the identical held-out time-test slice:
**24,903 accounts / 1,236 positives** (60/20/20 time split by account first-seen).
Primary metric: `capture_at_5pct` (fraction of mules in the riskiest 5% of the queue).

## Evidence table

| Model | capture@5% (test) | lift@5% |
|---|---|---|
| Rule baseline | 0.0502 | 1.00 |
| GB-200 teacher floor (60% train budget, non-leaky) | 0.0647 | 1.29 |
| Hybrid (GB-200 + residual), 3-seed mean | 0.0674 ± 0.0010 | ~1.35 |
| Hybrid best seed (42), full view | 0.0688 | 1.37 |
| Hybrid, transaction view (reified txn nodes) | 0.0688 | 1.37 |
| GB-500 teacher floor alone | 0.0680 | 1.36 |
| GB-500 + residual (lr .01 and lr .003) | = anchor (best epoch 1) | — |
| Reference: 80%-budget select-best tabular | 0.084 | — |
| Short-term milestone | 0.11 | — |
| Promotion gate | ≥ 0.25 and teacher+0.03 | ≥ 4.0 |

## What was established

1. **The hybrid mechanism works.** Teacher logit-offset + zero-init head starts the model
   exactly at the tabular floor; the graph residual then lifts it. Residual lift was
   **positive on every seed** (+0.0016 / +0.0025 / +0.0041) — the first GNN configuration
   in this project to beat its tabular comparator head-to-head. No vanilla GNN ever did
   (best vanilla ≈ 0.05–0.06 vs tabular 0.084).

2. **The lift is capped at ~+0.003 and is structure-independent.** Reifying transfers into
   transaction nodes with burst-gap timing features (`log_gap_since_src_prev_out`,
   `log_gap_since_dst_prev_in`) produced *exactly* the same number (0.0688). The graph
   restructure that was supposed to unlock rapid-in-out / chain-middle patterns changed
   nothing.

3. **The residual only lifts weak anchors.** On the stronger GB-500 teacher, the residual
   never improved validation (best epoch = 1 at two learning rates) — the GNN partially
   recovers teacher shortfall rather than extracting signal the features lack.

4. **Teacher strength is the productive axis on this dataset.** GB 200→500 trees moved the
   floor +0.0033 (0.0647→0.0680) — more than the residual's mean contribution. The
   remaining gap to the 0.084 reference is mostly **data budget**: the non-leaky teacher
   trains on 60% of accounts, the standalone tabular number on 80%.

## Interpretation

AMLSim 100K's engineered account features (`rapid_in_out_count`, `cycle_count_2hop`,
`chain_middle_score`, rule uplift, degree/amount aggregates) already summarize essentially
all graph and timing signal the simulator generates. Weak homophily (19.4% mule-neighbor
rate vs 6% base) is real but is *already priced into* the features. On this dataset there
is little graph-only residual left for any GNN to find — the promotion rule's honest
outcome is `KEEP_TABULAR`.

## Recommendations

1. **Production posture now:** tuned tabular (GB, full data budget, select-best protocol)
   as the scorer. The hybrid harness (this repo) stays ready — it is strictly ≥ teacher by
   construction and costs one flag (`--use-tabular-teacher`).
2. **Where the hybrid should win:** data whose engineered features do NOT already encode
   the graph — real bank data (feature pipelines never match a simulator's typology
   generator), or AMLSim 1M with more/overlapping typologies and lower feature coverage.
   Re-run this exact harness there before any further architecture work.
3. **Not recommended on 100K:** more GNN tuning (alpha/LR/heads/layers), v2 transaction
   features, hard-negative mining — three independent negative results say the ceiling is
   the data, not the model.
4. **Protocol note for the +0.03 gate:** compare hybrid against `teacher_test_metrics`
   (same slice, same budget — stored in every checkpoint), not against the 80%-budget
   standalone report; or move tabular selection to the same 60/20/20 protocol.

## Artifacts

- Code: `muleGuard_ai/tabular_teacher.py` (OOF cross-fit teacher),
  `train_gnn.py --use-tabular-teacher --teacher-*`, `pyg_adapter.py --graph-view
  transaction`, teacher-aware `gnn_inference.py`, tuner pass-through in `tune_gnn.py`.
- Tests: `tests/test_tabular_teacher.py` (7 tests: OOF invariants, anchor exactness,
  zero-init exactness, transaction-view reification).
- Checkpoints: `models/account_hybrid_100k_a10_gb.pt` (best hybrid),
  `models/hybrid_100k_gb500*.pt` (teacher-strength runs),
  `models/hybrid_100k_txview.pt` (transaction view), per-seed variants.
- Design/history: `HYBRID_TEACHER_PLAN.md`.
