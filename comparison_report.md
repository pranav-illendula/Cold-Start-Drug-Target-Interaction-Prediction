# Project vs. GraESM-FuseDTA: Detailed Comparison & Improvement Roadmap

> **Paper:** *GraESM-FuseDTA: Adaptive gated multimodal fusion of graph neural networks and protein language models for robust drug-target affinity prediction* — Jun-wen He, Naunyn-Schmiedeberg's Archives of Pharmacology, 2026.

---

## 1. At-a-Glance: Results Comparison

### Your Project (V4 — best checkpoint, epoch 5, BindingDB pIC50)

| Split | Samples | MAE | RMSE | Pearson r | Within ±1.0 |
|---|---:|---:|---:|---:|---:|
| Cold-Drug | 138,556 | **0.8896** | 1.1295 | **0.6021** | 63.83% |
| Cold-Pair | 31,839 | 0.9992 | 1.2491 | 0.3879 | 56.74% |
| Cold-Target | 123,438 | 1.0410 | 1.3041 | 0.3210 | 55.54% |

> Scale: pIC50 (3–12 range). Your Pearson r on Cold-Target (0.32) and Cold-Pair (0.39) is the core weak point.

---

### GraESM-FuseDTA (Davis dataset, 5-fold CV)

| Split | MSE | RMSE | Pearson r | CI | R² |
|---|---:|---:|---:|---:|---:|
| Warm Start | 0.238 | 0.488 | **0.886** | 0.881 | 0.761 |
| Cold-Drug | 0.512 | 0.716 | 0.590 | 0.754 | 0.261 |
| Cold-Target | 0.421 | 0.649 | **0.760** | 0.847 | 0.568 |
| Strict Pair Cold-Start | 0.496 | 0.704 | **0.866** | 0.912 | — |

> Scale: Davis Kd (different absolute range — not directly comparable with pIC50 MAE).
> The paper's **cold-target Pearson r = 0.760** vs. your **0.321** is the most diagnostic gap.

---

## 2. Component-Level Comparison

### 2a. Drug Encoder

| Aspect | Your Project | GraESM-FuseDTA |
|---|---|---|
| Architecture | 3× `GCNConv` (vanilla) | 4× Residual **GINE** layers |
| Edge features | ❌ Node-only (GCN ignores bond type) | ✅ Bond type, conjugation, ring, stereochemistry |
| Atom features | 18 (good baseline) | 10+ atom + 4 bond descriptors |
| Pooling | Global **mean** pool only | **Mean + Max + Add** (3× multi-pooling) |
| Residual connections | ❌ None | ✅ Between every GINE layer |
| Hidden dim | 128 | 256 |

> **🔴 Critical gap:** GCN cannot distinguish bond types (single vs. double vs. aromatic). GINE extends GIN to use edge features, making it strictly more expressive (Weisfeiler-Lehman hierarchy). This directly hurts cold-drug generalization because two atoms connected by different bond types look identical to GCN.

---

### 2b. Target / Protein Encoder

| Aspect | Your Project | GraESM-FuseDTA |
|---|---|---|
| ESM model | `esm2_t30_150M` (150M, 1280-d) | `esm2_t30_150M` (same backbone) |
| ESM fine-tuning | ❌ Frozen | ❌ Frozen (same — good) |
| Pooling | CLS + mean → 1280-d concat | CLS + mean → projected 256-d |
| Projection | 1280 → 256 → 128 (aggressive squeeze) | [CLS‖mean] → 256-d single projection |

> **🟡 Moderate gap:** Very similar ESM usage. The key difference: the paper projects into a **shared 256-d latent space** aligned with the drug embedding before fusion. Your 1280→256→128 two-step squeeze potentially discards information at the 128-d bottleneck.

---

### 2c. Fusion Module — The Most Important Gap

| Aspect | Your Project | GraESM-FuseDTA |
|---|---|---|
| Strategy | Simple **concatenation** [drug‖target] | **Gated interaction-aware fusion** |
| Compatibility signal | ❌ Implicit only | ✅ `m = d ⊙ p` (element-wise product) |
| Discrepancy signal | ❌ None | ✅ `r = |d − p|` (absolute difference) |
| Adaptive gate | ❌ None | ✅ `g = σ(Wq[d‖p‖r] + bq)`, pair-specific |
| Final representation | 256-d | 1280-d = [d‖p‖m‖r‖a] (5 components) |
| Regression head | MLP 256→128→64→1 (ReLU) | FC 1280→512→256→1 (LayerNorm + GELU + Dropout) |

> **🔴 Critical gap:** This is the paper's core contribution. Simple concatenation forces the regression MLP to *implicitly* discover compatibility from 256 numbers. The gated fusion gives the head **explicit** compatibility, mismatch, AND adaptive balance signals simultaneously. This is why cold-target and cold-pair Pearson r and CI are dramatically higher.

---

### 2d. Training Protocol

| Aspect | Your Project | GraESM-FuseDTA |
|---|---|---|
| Loss | Smooth L1 (β=1.0) ✅ | Smooth L1 ✅ (same) |
| Label normalization | z-score, train stats only ✅ | Fold-specific z-score ✅ (same) |
| Optimizer | AdamW, lr=3e-4, wd=1e-4 | AdamW, lr=5e-5, wd=1e-4 |
| LR schedule | ❌ Constant | ✅ Cosine decay + 10% warmup |
| Gradient accumulation | ❌ | ✅ Step=2 |
| Grad clipping | ✅ max_norm=1.0 | ✅ max_norm=1.0 |
| Batch size | 128 | 32 (effective 64 w/ accumulation) |
| Early stopping | ✅ patience=8 on val MAE | ✅ patience=30 on val MSE |
| Cross-validation | ❌ Single split (15% target holdout) | ✅ 5-fold outer CV |
| Statistical tests | ❌ | ✅ Fold-wise paired t-test |

> **🟡 Notable gap:** No LR scheduler + LR 6× higher than the paper's. The cosine warmup is important because ESM-adjacent projection layers need stable early updates. Your 3e-4 constant LR likely overshoots early in training.

---

### 2e. Dataset & Splits

| Aspect | Your Project | GraESM-FuseDTA |
|---|---|---|
| Dataset | **BindingDB** (~998K pairs, pIC50) | **Davis** (~30K) + **KIBA** (~118K) |
| Drug split | Murcko scaffold-balanced ✅ | Compound-level entity split |
| Target split | By name (exact match) | By sequence identity |
| Sequence clustering | ❌ Homologs can span train/test | ❌ Same limitation noted |
| Leakage check | ✅ Verified (all 0) | ✅ Enforced by construction |

> BindingDB is **harder**: 33× larger than Davis, much more label noise (multi-assay, multi-lab IC50s), and extreme pIC50 class imbalance. Your 0.60 cold-drug Pearson r on BindingDB vs. the paper's 0.59 on Davis is actually comparable.

---

## 3. Root Causes of Weak Cold-Target / Cold-Pair Performance

### 3.1 Cold-Target Pearson r: 0.321 (yours) vs. 0.760 (paper)

Two compounding causes:

1. **No explicit fusion signal** — When a target is new, the model has only the drug embedding and the unseen target's ESM embedding to work with. With simple concatenation, the MLP must learn *from scratch* what chemical-biological compatibility looks like for this pair. The compatibility term `m = d ⊙ p` provides a direct signal: dimensions where both drug and protein are simultaneously activated hint at binding potential.

2. **GCN drug encoder is less informative** — When the target is unseen, the model leans harder on the drug encoding to anchor predictions. GCN's inability to distinguish bond types means the drug embeddings carry less chemical nuance, making the fusion signal weaker.

### 3.2 High-pIC50 Failure (per-bin analysis)

Your cold-target MAE in the **9.0–10.5 bin = 2.697** vs. 0.595 in 6.0–7.5. This is severe mean regression. Root causes:

- Only ~5K training examples in pIC50 9–10.5 (vs. ~300K in 4.5–7.5)
- The model has no way to represent "this compound is unusually potent" — it regresses to the mean
- Multi-pooling (especially max-pool) would help by capturing the single most activated substructural motif, which tends to drive extreme potency

---

## 4. Actionable Improvements (Ranked by Impact)

### 🔴 Priority 1: Replace GCN with Residual GINE + Bond Features

```python
from torch_geometric.nn import GINEConv
import torch.nn as nn

class ResidualGINEBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index, edge_attr):
        return self.norm(x + self.dropout(self.conv(x, edge_index, edge_attr)))
```

Also add bond features (4-d) to your RDKit graph construction:
- Bond type (1-hot: single, double, triple, aromatic)
- Conjugated (bool)
- In ring (bool)
- Stereo (1-hot)

Stack 4 ResidualGINEBlocks. **Expected MAE drop: ~0.05–0.10 on cold-drug.**

---

### 🔴 Priority 2: Replace Concatenation with Gated Fusion

```python
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    z = [d | p | m | r | a]
    m = d ⊙ p  (compatibility)
    r = |d - p| (discrepancy)
    g = sigmoid(Wq [d|p|r] + bq)
    a = g ⊙ d + (1-g) ⊙ p  (adaptive mixture)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out_dim = hidden_dim * 5

    def forward(self, d, p):
        m = d * p
        r = (d - p).abs()
        g = torch.sigmoid(self.gate_proj(torch.cat([d, p, r], dim=-1)))
        a = g * d + (1 - g) * p
        return torch.cat([d, p, m, r, a], dim=-1)
```

Then update the regression head to accept 5×hidden_dim input:

```python
self.regression = nn.Sequential(
    nn.Linear(hidden_dim * 5, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.25),
    nn.Linear(256, 1),
)
```

**This is the single most impactful change for cold-target Pearson r. Expected gain: +0.10–0.20.**

---

### 🟡 Priority 3: Add Multi-Pooling for Drug

```python
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# In forward():
mean_vec = global_mean_pool(x, batch)   # h-d
max_vec  = global_max_pool(x, batch)    # h-d
add_vec  = global_add_pool(x, batch)    # h-d
drug_vector = self.drug_proj(torch.cat([mean_vec, max_vec, add_vec], dim=-1))
# drug_proj: Linear(3*h, h)
```

**Expected gain: +0.02–0.05 cold-drug Pearson r, helps high-pIC50 bins.**

---

### 🟡 Priority 4: Add Cosine LR Schedule with Warmup

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

INITIAL_LR = 5e-5  # lower than current 3e-4
total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.10 * total_steps)

optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

# In training loop, call scheduler.step() after each batch
```

**Expected gain: +0.02–0.05 across all splits, better convergence.**

---

### 🟡 Priority 5: Widen Hidden Dim to 256

Change `hidden_dim=128` → `hidden_dim=256` in model init. The fusion output is 5×256=1280-d, matching the paper's architecture exactly. This increases parameters from ~2M to ~8M — still very trainable on your NVIDIA RTX 2000 Ada.

---

### 🟢 Priority 6: Add Concordance Index (CI) Metric

CI is the standard ranking metric in DTA papers and more drug-discovery-relevant than MAE:

```python
def concordance_index(y_true, y_pred):
    """Efficient vectorized Harrell's CI using lifelines or manual."""
    from lifelines.utils import concordance_index as ci
    return ci(y_true, y_pred)

# Add to your score() function in evaluate_v4.py:
"ci": float(concordance_index(targets, predictions)),
```



## 5. Summary Priority Table

| Change | Effort | Pearson r gain (cold-target) | MAE gain |
|---|---|---|---|
| GCN → Residual GINE + bond features | **Medium** | +0.05–0.10 | −0.03–0.06 |
| Concat → Gated Fusion module | **Medium** | **+0.10–0.20** | −0.05–0.10 |
| Single pool → Mean+Max+Add | **Low** | +0.02–0.05 | −0.02 |
| Cosine warmup LR schedule | **Low** | +0.02–0.05 | −0.02–0.04 |
| Hidden dim 128 → 256 | **Low** | +0.01–0.03 | −0.01 |
| Concordance Index metric | **Trivial** | N/A (monitoring) | N/A |
| Sequence-identity target clustering | **High** | +0.0 (more honest) | 0 |

> **Bottom line:** Implementing **Priority 1 + Priority 2** (GINE + Gated Fusion) is the minimal intervention to close the biggest gaps. Together they should push cold-target Pearson r from **0.32 → ~0.50–0.55** on BindingDB — a large gain given your dataset is harder and larger than Davis.

---

## 6. What Your Project Already Does Well

- ✅ **Murcko scaffold split** — More rigorous drug split than simple random or even the paper's compound-level split
- ✅ **CLS + mean ESM pooling** — Matches the paper's approach exactly
- ✅ **Label z-score normalization using train stats only** — No leakage, same as paper
- ✅ **Smooth L1 loss** — Same as paper, good choice for noisy pIC50 labels
- ✅ **Per-bin evaluation** — More diagnostic than overall metrics
- ✅ **Leakage-verified splits** — All checks pass at 0
- ✅ **Grad clipping + AdamW** — Stable training practice
- ✅ **BindingDB scale** — 998K pairs is much larger and harder than Davis/KIBA benchmarks
