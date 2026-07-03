# GNN Mule Detection Improvement Plan
## Phase-Wise Roadmap to Reach Promotion Gates (capture@5% ≥ 0.25, lift@5% ≥ 4.0)

---

### Current Baseline (AMLSim 10K)
| Metric | Baseline (RF) | Best GNN (GATv2) | Target | Gap |
|--------|--------------|------------------|--------|-----|
| capture@5% | 0.119 | **0.154** | 0.25 | +62% |
| lift@5% | 2.38 | **3.07** | 4.0 | +30% |
| precision@5% | 0.132 | **0.189** | — | — |
| recall | **0.504** | 0.342 | >0.5 | -32% |
| PR-AUC | 0.082 | **0.109** | >0.15 | — |

---

## Phase 0: Foundation & Infrastructure (Week 1)
**Goal**: Stabilize training, enable rapid experimentation

### Tasks
- [ ] **0.1** Fix PyTorch Geometric import timeout issue (pandas compat)
  - Pin `pandas<2.2` or upgrade PyG to ≥2.6.1
- [ ] **0.2** Create experiment tracking wrapper
  - Log: config, seed, metrics, checkpoint path, git commit
  - Output: `runtime/reports/experiments/<timestamp>_<config>.json`
- [ ] **0.3** Build AMLSim 100K dataset
  - Run `/Users/ashutoshanand/AMLSim/paramFiles/100K/conf.json`
  - Convert via `convert_amlsim.py` → `runtime/data/amlsim_100k`
  - Run `build_features.py` → `runtime/data/amlsim_100k_features`
  - Target: ~10K positive accounts (vs 742 in 10K)
- [ ] **0.4** Add automated split verification script
  - Confirm temporal split integrity (no future leakage)
  - Report: train/val/test date ranges, positive counts per split

### Deliverables
- Stable training env (< 30s startup)
- 100K feature dataset ready
- Experiment logging infrastructure

---

## Phase 1: Architecture Search (Week 2)
**Goal**: Find best architecture for 10K, validate on 100K

### 1.1 Targeted Hyperparameter Search (10K)
```bash
# Expanded search space
python -m muleGuard_ai.tune_gnn \
  --data runtime/data/amlsim_10k_features \
  --runs 3 \
  --epochs 80 \
  --patience 12 \
  --architectures hetero_sage,gatv2,edge_transformer \
  --losses focal,cross_entropy \
  --hidden-channels-grid 32,64,128 \
  --layers-grid 2,3,4 \
  --dropout-grid 0.2,0.3,0.4 \
  --lr-grid 0.003,0.005,0.01,0.02 \
  --ranking-loss-weight-grid 0.1,0.2,0.3 \
  --focal-gamma-grid 1.0,1.5,2.0 \
  --metric capture_at_2pct \
  --output runtime/reports/gnn_arch_search_10k.json
```

### 1.2 Key Hypotheses to Test
| Hypothesis | Config | Expected Effect |
|------------|--------|-----------------|
| Edge-aware attention beats message passing | `edge_transformer` vs `gatv2` | Better cycle detection |
| 3+ layers needed for 3-hop patterns | `layers=3,4` | Capture cycle_3hop |
| Lower focal γ improves recall | `focal_gamma=1.0` | Less suppression of hard positives |
| Higher ranking weight optimizes capture@k | `ranking_loss_weight=0.2-0.3` | Direct metric optimization |

### 1.3 Validation Criteria
- Primary: `capture_at_2pct` mean across 3 seeds
- Secondary: `lift_at_2pct`, `recall` > 0.35
- Gate: Best config must beat GATv2 baseline by ≥5% on capture@2%

### Deliverables
- `gnn_arch_search_10k.json` with full results
- Best checkpoint: `models/best_arch_10k.pt`
- Decision doc: chosen architecture + rationale

---

## Phase 2: Feature Engineering (Week 3)
**Goal**: Enrich node/edge features for structural + temporal signals

### 2.1 Node Feature Enhancements (`build_features.py`)
```python
# Add to _engineer_features():
- txn_time_entropy: entropy of transaction hour-of-day
- night_tx_ratio: txns 22:00-06:00 / total
- active_hours_span: max_hour - min_hour of activity
- burstiness: (max_daily_tx - mean_daily_tx) / std_daily_tx
- counterparty_concentration: Herfindahl index of counterparty distribution
- amount_entropy: entropy of transaction amounts
```

### 2.2 Graph-Level Features (PyG Transforms)
```python
# In pyg_adapter.py, after to_pyg_heterodata():
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
data = AddLaplacianEigenvectorPE(k=16)(data)  # positional encoding
data = AddRandomWalkPE(walk_length=16)(data)   # structural encoding
```

### 2.3 Edge Feature Enhancements
- Add: `time_since_last_tx`, `tx_sequence_position`, `amount_zscore_per_account`
- Current: `[weight, log1p(day_offset), direction]` → expand to 6-8 dims

### 2.4 Feature Quality Audit
```bash
python -m muleGuard_ai.feature_quality \
  --data runtime/data/amlsim_10k_features \
  --output runtime/reports/feature_quality_10k_v2.json
```
Target: Remove features with >95% zero variance, add mutual info scores vs label

### Deliverables
- Updated `build_features.py` with new features
- Rebuilt 10K and 100K feature sets
- Feature quality report

---

## Phase 3: Training Strategy Optimization (Week 4)
**Goal**: Maximize validation capture@k with chosen architecture

### 3.1 Loss Function Ablation
| Config | Description |
|--------|-------------|
| CE + ranking (0.2) | Baseline |
| Focal(γ=1.0) + ranking (0.2) | Less aggressive |
| Focal(γ=1.5) + ranking (0.3) | Stronger ranking |
| Focal(γ=1.0) + ranking (0.3) + **contrastive** | Add node-level contrastive loss |
| **Asymmetric loss** (higher FN cost) | `weight_fn=3.0, weight_fp=1.0` |

### 3.2 Learning Rate & Schedule
- Warmup: 5 epochs linear warmup
- Cosine annealing with restarts (T_0=20, T_mult=2)
- Min LR: 1e-5

### 3.3 Regularization
- Label smoothing: 0.1
- DropEdge: 0.1 (randomly drop 10% edges per epoch)
- Feature dropout: 0.15 (in addition to model dropout)

### 3.4 Class Balancing Strategy
- Current: `balanced` class weights
- Try: `effective_num` weighting (Cui et al. 2019)
- Try: Focal loss handles imbalance → set `class_weighting=none`

### Deliverables
- Training config YAML for best setup
- `models/best_train_strategy_100k.pt`
- Ablation study report

---

## Phase 4: Scale to 100K (Week 5)
**Goal**: Validate on production-scale data

### 4.1 Full Training on 100K
```bash
python -m muleGuard_ai.train_gnn \
  --transactions runtime/data/amlsim_100k_features/muleguard_core_transactions.csv \
  --telemetry runtime/data/amlsim_100k_features/muleguard_digital_telemetry.csv \
  --entity-map runtime/data/amlsim_100k_features/muleguard_entity_map_full.csv \
  --node-features runtime/data/amlsim_100k_features/muleguard_node_features_full.csv \
  --architecture <best_from_phase1> \
  --hidden-channels <best> \
  --layers <best> \
  --epochs 150 \
  --patience 20 \
  --lr <best> \
  --ranking-loss-weight <best> \
  --focal-gamma <best> \
  --validation-metric capture_at_2pct \
  --output models/account_gnn_100k_best.pt
```

### 4.2 Multi-Seed Validation (5 seeds)
```bash
for seed in 42 123 456 789 999; do
  python -m muleGuard_ai.train_gnn ... --seed $seed --output models/100k_seed${seed}.pt
done
python -m muleGuard_ai.evaluate --data runtime/data/amlsim_100k_features --checkpoint models/100k_seed42.pt ...
```

### 4.3 Stress Test: Temporal Generalization
- Train on first 80% time, test on last 20%
- Compare: random split vs temporal split performance gap
- Target: <10% degradation on temporal split

### Deliverables
- 5-seed mean/std metrics on 100K
- Temporal generalization report
- Production checkpoint: `models/account_gnn_100k_production.pt`

---

## Phase 5: Ensemble & Calibration (Week 6)
**Goal**: Combine GNN + Tabular for best recall/precision tradeoff

### 5.1 Tabular Baseline Retraining on 100K
```bash
python -m muleGuard_ai.train_baseline_model \
  --data runtime/data/amlsim_100k_features \
  --select-best \
  --runs 5 \
  --metric capture_at_5pct \
  --output models/tabular_100k_best.pkl
```

### 5.2 Ensemble Strategies
| Method | Formula |
|--------|---------|
| Weighted average | `0.6*gnn + 0.4*rf` |
| Stacking | Logistic regression on [gnn_score, rf_score, baseline_score] |
| Rank averaging | Average of percentile ranks |
| **Per-typology thresholds** | Calibrate threshold per `amlsim_typology` |

### 5.3 Threshold Calibration
- Use validation set (time-based) for threshold selection
- Optimize for: `capture_at_5pct` subject to `precision_at_5pct >= 0.15`
- Export per-typology thresholds for production

### Deliverables
- Ensemble model + calibration artifacts
- Production scoring script with thresholds
- Final metrics on 100K held-out test

---

## Phase 6: Production Readiness (Week 7)
**Goal**: Operationalize best model

### 6.1 Inference Optimization
- Export to TorchScript/ONNX
- Batch inference benchmark (target: <100ms per 10K accounts)
- Memory profiling

### 6.2 Monitoring & Drift Detection
- Feature drift: KS test on node features (weekly)
- Score distribution drift: PSI on score histograms
- Label drift: Track positive rate in alerts

### 6.3 Documentation
- Model card (architecture, data, metrics, limitations)
- Runbook: retraining trigger, rollback procedure
- API spec for `muleGuard_ai.operational --checkpoint`

---

## Success Criteria Summary

| Phase | Gate Metric | Target |
|-------|-------------|--------|
| 1 (Architecture) | capture@2% (10K, 3-seed mean) | >0.12 |
| 2 (Features) | Feature MI improvement | >15% avg MI gain |
| 3 (Training) | capture@2% (10K, best seed) | >0.15 |
| 4 (100K Scale) | capture@5% (100K, 5-seed mean) | >0.20 |
| 5 (Ensemble) | capture@5% + precision@5% | >0.25 / >0.15 |
| **Final** | **Promotion gates** | **≥0.25 / ≥4.0** |

---

## Resource Estimates

| Phase | Compute | Time | Storage |
|-------|---------|------|---------|
| 0 | Low | 1 day | 5 GB (100K data) |
| 1 | 3 archs × 3 seeds × 80 ep = 720 runs | 2 days (GPU) | 2 GB checkpoints |
| 2 | Feature eng only | 0.5 days | 1 GB |
| 3 | 20 configs × 3 seeds = 60 runs | 1 day (GPU) | 1 GB |
| 4 | 5 seeds × 150 ep | 3 days (GPU) | 5 GB |
| 5 | Ensemble eval | 0.5 days | 0.5 GB |
| 6 | Export + test | 0.5 days | 0.5 GB |
| **Total** | — | **~8 days** | **~15 GB** |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 100K AMLSim generation fails | Medium | High | Fallback: use 10K + heavy augmentation (edge dropout, feature noise) |
| GNN still can't beat tabular on 100K | Low | High | Pivot: use GNN embeddings as tabular features (GraphSAGE → RF) |
| Temporal split gap >20% | Medium | Medium | Add time-aware positional encoding; try T-GNN variants |
| PyG/pandas compatibility breaks | Low | High | Pin versions in `requirements.txt`; test in CI |

---

## Next Immediate Actions

```bash
# 1. Fix environment
pip install "pandas<2.2" -q  # or upgrade pyg

# 2. Generate 100K AMLSim data
cd /Users/ashutoshanand/AMLSim && ./run.sh paramFiles/100K/conf.json

# 3. Convert & build features
python -m muleGuard_ai.convert_amlsim --input /Users/ashutoshanand/AMLSim/outputs/100K --output runtime/data/amlsim_100k
python -m muleGuard_ai.build_features --data runtime/data/amlsim_100k --output runtime/data/amlsim_100k_features

# 4. Run Phase 1 architecture search
python -m muleGuard_ai.tune_gnn --data runtime/data/amlsim_10k_features --runs 3 --epochs 80 --architectures hetero_sage,gatv2,edge_transformer --hidden-channels-grid 32,64 --layers-grid 2,3 --metric capture_at_2pct --output runtime/reports/gnn_arch_search_10k.json
```

---

*Plan created: 2026-06-28 | Target completion: 2026-07-19*