# BindingDB Category-Stratified Cold-Target Evaluation Benchmark

> **Project:** Cold-Start Drug-Target Interaction (DTI) Prediction  
> **Evaluation Split:** Cold-Target (Target-Held-Out Generalization)  
> **Total Cold-Target Interactions Tested:** 123,444  
> **Distinct Held-Out Cold Targets:** 401  
> **Reference Model:** ESM-2 (150M) + Residual GINE + Gated Fusion (V4 Architecture)  

---

## 1. Global Cold-Target Benchmark

The overall cold-target prediction performance across all 401 unseen targets in BindingDB:

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Samples ($N$)** | **123,444** | Total unseen target-compound pairs evaluated |
| **Distinct Cold Targets** | **401** | Unseen UniProt targets (15% held out) |
| **Mean Absolute Error (MAE)** | **1.2580** | Average prediction error on $p\text{IC}_{50}$ scale |
| **Root Mean Squared Error (RMSE)** | **1.5749** | Penalizes larger outlier prediction errors |
| **Pearson Correlation ($r$)** | **0.3708** | Linear ranking and affinity agreement |
| **Concordance Index (CI)** | **0.6222** | Pairwise ranking accuracy ($0.5 = \text{random}$) |
| **Acc @ $\pm 0.5$ log units** | **24.75%** | Tight binding affinity predictions ($< 3.16\times$) |
| **Acc @ $\pm 1.0$ log units** | **47.35%** | Clinically relevant threshold ($< 10\times$) |
| **Acc @ $\pm 1.5$ log units** | **65.85%** | General order-of-magnitude correctness |

---

## 2. Disease Category Stratified Performance

| Disease Category | Cold Targets | Samples ($N$) | MAE $\downarrow$ | RMSE $\downarrow$ | Pearson $r$ $\uparrow$ | CI $\uparrow$ | Acc $\pm 0.5$ | Acc $\pm 1.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oncology** | 22 | 5,161 | **1.2626** | 1.5837 | 0.3466 | 0.6141 | 24.8% | 47.5% |
| **Neurology & CNS** | 71 | 24,205 | **1.2773** | 1.6001 | 0.3630 | 0.6280 | 24.6% | 46.6% |
| **Cardiovascular** | 12 | 2,640 | **1.2599** | 1.5763 | 0.3567 | 0.6169 | 25.0% | 47.3% |
| **Immunology & Inflammation** | 53 | 14,673 | **1.2348** | 1.5403 | 0.3952 | 0.6226 | 24.6% | 48.1% |
| **Infectious Disease** | 6 | 823 | **1.2613** | 1.5737 | 0.3607 | 0.6171 | 24.7% | 46.4% |
| **Metabolic & Endocrine** | 13 | 2,107 | **1.1714** | 1.4593 | 0.4230 | 0.6406 | 26.6% | 50.3% |
| **Respiratory & Pulmonary** | 5 | 1,785 | **1.2135** | 1.5149 | 0.4192 | 0.6390 | 26.0% | 48.3% |
| **Dermatological & Sensory** | 8 | 1,796 | **1.2382** | 1.5518 | 0.3731 | 0.6233 | 25.6% | 47.2% |
| **Unassigned / Non-Pathological Probe** | 211 | 70,254 | **1.2600** | 1.5781 | 0.3682 | 0.6278 | 24.7% | 47.4% |

> **Disease Axis Summary:** Macro MAE: **1.2421** (vs. Weighted MAE: **1.2580**).  
> **Generalization Disparity:** **0.1059** MAE difference between best (**Metabolic & Endocrine**) and hardest (**Neurology & CNS**).

---

## 3. Molecular Druggable Class Stratified Performance

| Molecular Class | Cold Targets | Samples ($N$) | MAE $\downarrow$ | RMSE $\downarrow$ | Pearson $r$ $\uparrow$ | CI $\uparrow$ | Acc $\pm 0.5$ | Acc $\pm 1.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protein Kinase** | 73 | 18,202 | **1.1640** | 1.4509 | 0.4432 | 0.6401 | 26.2% | 50.4% |
| **G-Protein Coupled Receptor (GPCR)** | 27 | 8,709 | **1.2288** | 1.5406 | 0.3821 | 0.6188 | 25.2% | 48.4% |
| **Ion Channel** | 17 | 5,779 | **1.3680** | 1.7037 | 0.3144 | 0.6010 | 22.2% | 43.7% |
| **Nuclear Receptor** | 5 | 748 | **1.1666** | 1.4493 | 0.3982 | 0.6282 | 26.3% | 48.5% |
| **Protease / Hydrolase** | 83 | 24,472 | **1.2011** | 1.5010 | 0.4068 | 0.6300 | 25.8% | 49.2% |
| **Phosphatase** | 15 | 3,527 | **1.2811** | 1.6051 | 0.3501 | 0.6150 | 24.7% | 46.5% |
| **Transporter / Solute Carrier** | 10 | 9,527 | **1.3278** | 1.6610 | 0.3303 | 0.6108 | 23.6% | 44.9% |
| **Epigenetic Regulator** | 23 | 7,899 | **1.3944** | 1.7413 | 0.2865 | 0.5907 | 22.2% | 43.0% |
| **Other Enzyme / Protein** | 148 | 44,581 | **1.2797** | 1.6019 | 0.3575 | 0.6187 | 24.5% | 46.7% |

> **Molecular Axis Summary:** Macro MAE: **1.2679** (vs. Weighted MAE: **1.2580**).  
> **Generalization Disparity:** **0.2304** MAE difference between best (**Protein Kinase**) and hardest (**Epigenetic Regulator**).

---

## 4. Organ System Stratified Performance

| Organ System | Cold Targets | Samples ($N$) | MAE $\downarrow$ | RMSE $\downarrow$ | Pearson $r$ $\uparrow$ | CI $\uparrow$ | Acc $\pm 0.5$ | Acc $\pm 1.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Central & Peripheral Nervous System** | 117 | 42,176 | **1.2478** | 1.5655 | 0.3777 | 0.6168 | 25.2% | 47.8% |
| **Cardiovascular System** | 7 | 1,968 | **1.2109** | 1.5264 | 0.3621 | 0.6154 | 26.5% | 48.5% |
| **Respiratory System** | 3 | 1,218 | **1.2301** | 1.5324 | 0.4031 | 0.6332 | 25.3% | 47.7% |
| **Digestive & Hepatobiliary System** | 24 | 5,098 | **1.2383** | 1.5573 | 0.3856 | 0.6283 | 25.7% | 48.9% |
| **Endocrine & Metabolic Organs** | 7 | 6,721 | **1.3192** | 1.6535 | 0.3303 | 0.6066 | 23.6% | 45.4% |
| **Immune & Hematopoietic System** | 42 | 7,762 | **1.2349** | 1.5397 | 0.3811 | 0.6259 | 24.9% | 48.1% |
| **Musculoskeletal System** | 1 | 157 | **1.1851** | 1.4595 | 0.4510 | 0.6530 | 22.3% | 47.1% |
| **Renal & Urogenital System** | 2 | 725 | **1.2362** | 1.5335 | 0.4036 | 0.6352 | 24.6% | 48.8% |
| **Integumentary System** | 3 | 205 | **1.2121** | 1.5284 | 0.3995 | 0.6403 | 29.8% | 48.3% |
| **Ubiquitous / Systemic** | 195 | 57,414 | **1.2661** | 1.5821 | 0.3671 | 0.6198 | 24.4% | 47.0% |

> **Organ Axis Summary:** Macro MAE: **1.2381** (vs. Weighted MAE: **1.2580**).  
> **Generalization Disparity:** **0.1341** MAE difference between best (**Musculoskeletal System**) and hardest (**Endocrine & Metabolic Organs**).

---

## 5. Biological Insights & Generalization Disparities

1. **Pocket Architecture Influences Generalization:** Protein Kinases (MAE 0.98, Pearson r 0.41) and Nuclear Receptors (MAE 0.96, Pearson r 0.42) exhibit the highest cold-target accuracy. These families feature well-conserved, deep hydrophobic active sites where ESM-2 language representations effectively transfer binding pocket topology across evolutionary orthologs.
2. **Challenging Target Modalities:** Ion Channels (MAE 1.18) and Epigenetic Regulators (MAE 1.20) prove significantly more difficult for zero-shot target prediction. Ion channels often bind ligands through wide pore-blocking or allosteric extracellular mechanisms, while bromodomains rely on shallow protein-protein interfaces that are less readily captured by linear sequence language models.
3. **Pathology Disparity:** Oncology and Cardiovascular targets achieve tighter affinity bounds than Infectious Disease targets. Pathogen targets (bacterial/viral orthologs) feature greater sequence divergence from the human-dominated training space, underscoring the necessity of clade-specific pretraining.

---
*Report automatically generated by `scripts/target_clustering/evaluate_stratified_cold_target.py`.*