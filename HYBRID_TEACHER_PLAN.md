# Hybrid Tabular-Teacher + Graph-Residual Plan

## Goal
Make the GNN add **residual graph lift** on top of the already-strong engineered tabular
model, instead of trying to rediscover tabular signal through message passing.

Target metric: `capture_at_5pct` (fraction of mules captured in the riskiest 5% of accounts).

| Milestone | Target |
|-----------|--------|
| Short-term | `capture_at_5pct >= 0.11` |
| Stronger | `capture_at_5pct >= 0.25`, `lift_at_5pct >= 4.0` |
| Promotion | hybrid beats tabular mean by `+0.03` across seeds |

## Core idea: logit offset (anchored residual)
Every downstream loss/metric in `train_gnn.py` reads the 2-column `out` logits tensor
(`_forward` -> focal/CE -> pairwise ranking -> `_score_metric`). So we anchor the model at
the tabular teacher by adding the teacher logit to the class-1 logit, right after the
forward pass:

```
out[:, 1] = out[:, 1] + alpha * teacher_logit    # class-0 logit untouched
```

The final head layer is zero-initialized in teacher mode (`--teacher-zero-init-head`,
default on; implemented as `zero_init_head()` called after lazy-param materialization), so
the initial prediction is exactly `sigmoid(alpha * teacher_logit)` — the model **starts at
the tabular baseline** and learns the graph delta on top. This makes tabular's
`capture_at_5pct` (~0.084 on 100K) the **floor**, not the target.

`alpha` ("hybrid score alpha") is a hyperparameter knob (grid it; optionally make it a
learnable scalar later).

## Non-leakage (teacher as a non-leaky feature)
The time-based split is deterministic and identical in `pyg_adapter.py` and
`train_baseline_model.py` (`strategy="time"`), so it is reproducible from the GNN masks.

- Teacher is fit **only on the GNN `train_mask`** accounts.
- Val + test accounts get **full-train** teacher predictions (out-of-fold by construction).
- Train accounts get **K-fold cross-fit (OOF)** teacher predictions, so the GNN never sees
  optimistic in-sample teacher scores on train and over-trusts the feature.
- Fold-training subsets that end up single-class fall back to the full-train prediction.

## Changes by file
1. **`muleGuard_ai/tabular_teacher.py`** (new) — fit teacher (numpy-logistic default, sklearn
   GB/RF optional), cross-fit OOF logits on train, full-train logits elsewhere, and a
   reusable `apply` for inference. `teacher_logit = logit(clip(prob))`.
2. **`muleGuard_ai/train_gnn.py`** — add `--use-tabular-teacher`, `--teacher-alpha`,
   `--teacher-model`, `--teacher-cv-folds`. Compute `teacher_logit` aligned to
   `data["account"].node_ids`, attach to `data`, apply the offset at every forward site,
   persist the teacher payload + alpha in the checkpoint, and record the **tabular teacher's**
   test metrics in the report (the honest `+0.03` comparison baseline).
3. **`muleGuard_ai/gnn_inference.py`** — reload the teacher payload + alpha and apply the same
   offset before softmax so scoring matches training.
4. **`tests/`** — hybrid smoke test on `amlsim_1k_features`.

Metadata guard in `gnn_inference.py` is unaffected (it checks node/edge *types*, not feature
dims; Linear layers are lazy).

## Findings so far
- AMLSim 1K end-to-end smoke (2026-07-03): pipeline works (teacher fit, offset, checkpoint,
  reload), but the 1K time-test slice has only **6 positives / 289 accounts** — the teacher
  itself scores `capture_at_5pct = 0.0` there (KS 0.30). 1K is a plumbing smoke only;
  metric judgment happens on 100K (~1.5K test positives expected).
- First 100K hybrid run (2026-07-03, GB teacher, alpha=1.0, zero-init head, lr=0.01,
  rank 0.2; test slice = 24,903 accounts / 1,236 positives):

  | model (same test slice) | capture@5% | lift@5% |
  |---|---|---|
  | rule baseline | 0.0502 | 1.00 |
  | GB teacher floor | 0.0647 | 1.29 |
  | hybrid | **0.0688** | **1.37** |

  The residual mechanism works — first GNN config to beat its tabular comparator
  head-to-head. Two follow-ups: (a) the non-leaky teacher trains on the 60% train split,
  while the standalone tabular 0.084 number trains on 80% — the floor has headroom via a
  stronger teacher family/params, not via the GNN; (b) residual lift peaks at ~epoch 5 with
  lr=0.01 then decays — the residual wants a lower LR / longer patience.

## Experiments (100K data is ready: `runtime/data/amlsim_100k_features/`, 124,517 nodes)
Grid: `teacher_alpha in {0.5,1.0,1.5}` x `ranking_loss_weight in {0.1,0.2,0.3}`,
teacher in {numpy_logistic, gradient_boosting}, 3 seeds, select on `capture_at_5pct`.

- Gate 1: mean `capture_at_5pct >= 0.11` (expected — floor is ~0.084).
- Gate 2: `>= 0.25` and `lift@5pct >= 4.0`. If not reached, that is the signal to move to the
  transaction-centric graph (transaction nodes) and temporal windowing — sequenced *after*
  this, since they are much larger graph-construction changes and should not stack on an
  unproven residual framing.

## Phase 2: transaction-centric graph (activated 2026-07-03)
The residual-lift condition triggered: sweep showed residual adds only ~+0.003 test
capture (seed noise scale), so structural change is required for the 0.11 milestone.

- `--graph-view transaction` (implemented in `pyg_adapter.py`): each transfer becomes a
  transaction node with features `[log_amount, log_day_offset, log_gap_since_src_prev_out,
  log_gap_since_dst_prev_in]` (-1 sentinel for first transfers); edges `account -sends->
  transaction -delivers-> account` plus reverses. Direct `transfers_to` edges removed in
  this view; entity edges kept. Inference rebuilds the checkpoint's stored view.
- Sweep results (100K, teacher floor 0.0647 on test): GB lr .01 hybrid 0.0688; GB lr .003
  0.0672; RF teacher worse floor (0.0583, overconfident probs); no-class-weighting residual
  never leaves the anchor. Residual deltas are 3-5 accounts of 1,236 positives.
- v2 candidates if v1 helps: channel/status features on transaction nodes (requires
  graph_dataset change), temporal windowed subgraphs, hard-negative mining.
- 3-seed consolidation (GB200 teacher, alpha 1.0, lr .01, rank 0.2; teacher floor 0.0647):
  hybrid test capture@5% = 0.0688 / 0.0663 / 0.0672 (seeds 42/123/456) — mean 0.0674,
  **residual lift positive on every seed** but only +0.0027 mean.
- Teacher strength: GB 500 trees depth 4 lifts the floor alone to **0.0680** (test) /
  0.0841 (val anchor) — as good as the GB200 hybrids — though in that run the residual
  never improved on the stronger anchor (early stop at epoch 1). Teacher strength currently
  buys more than the graph residual; the 0.084 (80%-budget select-best) tabular remains
  above all 60%-budget floors.

## Outcome (2026-07-03)
GB-500 + gentle residual (lr .003, patience 15, seed 42): best epoch = 1 — residual never
lifted the stronger anchor, confirming the GB-500 lr .01 result. Transaction view matched
the full view exactly (0.0688). Conclusion and recommendations moved to
`HYBRID_TEACHER_FINDINGS.md`: mechanism validated, residual lift real but ~+0.003 and
capped by the dataset (features already encode the graph); teacher-first for production,
re-run the harness on 1M / real data.

## Still deferred
- Explicit temporal train/val/test windows (early/middle/future) beyond the current
  first-seen ordering.
