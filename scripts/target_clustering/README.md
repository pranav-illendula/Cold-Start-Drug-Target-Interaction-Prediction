# Requirement 1: Disease & Organ-Level Target Clustering for BindingDB

Reproducible, research-grade pipeline that standardizes BindingDB drug targets into controlled biological ontologies, profiles target imbalances and diversity, and evaluates cold-target prediction performance across biological domains.

---

## Architecture Overview

```
scripts/target_clustering/
├── __init__.py                      # Package definitions & controlled vocabularies (9 Diseases, 10 Organs, 9 Classes)
├── fetch_annotations.py             # Stage 1: Batch UniProtKB REST API client with local JSON caching
├── classify_targets.py              # Stage 2: Biological standardizer & schema builder (CSV, Parquet, Lookup PKL)
├── analyze_distribution.py          # Stage 3: Imbalance profiler, cross-tabulations & Shannon/Simpson diversity
├── evaluate_stratified_cold_target.py # Stage 4: Multi-axial category-stratified cold-target evaluation engine
└── run_pipeline.py                  # Stage 5: Unified CLI orchestrator
```

---

## Controlled Vocabularies

### 1. Disease Categories (9 Canonical Domains)
- `Oncology`
- `Neurology & CNS`
- `Cardiovascular`
- `Immunology & Inflammation`
- `Infectious Disease`
- `Metabolic & Endocrine`
- `Respiratory & Pulmonary`
- `Dermatological & Sensory`
- `Unassigned / Non-Pathological Probe`

### 2. Organ Systems (10 Physiological Systems)
- `Central & Peripheral Nervous System`
- `Cardiovascular System`
- `Respiratory System`
- `Digestive & Hepatobiliary System`
- `Endocrine & Metabolic Organs`
- `Immune & Hematopoietic System`
- `Musculoskeletal System`
- `Renal & Urogenital System`
- `Integumentary System`
- `Ubiquitous / Systemic`

### 3. Molecular Classes (9 Druggable Families)
- `Protein Kinase`
- `G-Protein Coupled Receptor (GPCR)`
- `Ion Channel`
- `Nuclear Receptor`
- `Protease / Hydrolase`
- `Phosphatase`
- `Transporter / Solute Carrier`
- `Epigenetic Regulator`
- `Other Enzyme / Protein`

---

## Output Artifacts

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Annotated Table (CSV)** | `data/bindingdb/target_annotations_bindingdb.csv` | Full table with 13 biological columns for all 2,671 targets. |
| **Annotated Table (Parquet)** | `data/bindingdb/target_annotations_bindingdb.parquet` | Snappy-compressed columnar format for rapid queries. |
| **Lookup Dictionary (PKL)** | `data/bindingdb/target_annotations_lookup.pkl` | Index-aligned Python dictionary matching `0..2670`. |
| **Distribution Report** | `results/bindingdb/target_distribution_report.md` | Statistical distribution, cross-tabulation matrix & diversity indices. |
| **Distribution Summary** | `results/bindingdb/target_distribution_summary.json` | Programmatic JSON of distributions and cross-tabs. |
| **Cold-Target Benchmark** | `results/bindingdb/cold_target_category_benchmark.md` | Stratified cold-target MAE, RMSE, Pearson r, and Concordance Index. |
| **Benchmark Summary (CSV)** | `results/bindingdb/cold_target_category_benchmark.csv` | Stratified metrics table across all 28 biological strata. |

---

## CLI Usage

### Check Artifact Status
```bash
python scripts/target_clustering/run_pipeline.py --status
```

### Run Entire Pipeline End-to-End
```bash
python scripts/target_clustering/run_pipeline.py --all
```

### Run Individual Stages
```bash
# Stage 1: Fetch raw annotations
python scripts/target_clustering/run_pipeline.py --fetch

# Stage 2: Standardize & build schemas
python scripts/target_clustering/run_pipeline.py --classify

# Stage 3: Analyze distributions & cross-tabulations
python scripts/target_clustering/run_pipeline.py --analyze

# Stage 4: Run stratified cold-target evaluation
python scripts/target_clustering/run_pipeline.py --evaluate
```
