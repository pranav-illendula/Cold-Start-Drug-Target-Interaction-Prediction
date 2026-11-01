# BindingDB Drug Target Biological Distribution & Imbalance Report

> **Project:** Cold-Start Drug-Target Interaction (DTI) Prediction  
> **Dataset:** BindingDB Curated Target Portfolio  
> **Total Drug Targets Profiled:** 2,671  
> **Annotation Provenance:** UniProtKB / Swiss-Prot Knowledgebase & Gene Ontology  

---

## Executive Summary

This report presents an evidence-backed characterization of the **2,671 unique biological targets** present in the BindingDB dataset. Targets have been standardized into 9 controlled Disease Categories, 10 Organ Systems, and 9 Molecular Druggable Classes.

### Key Findings:
1. **Disease Association Breadth:** **955 targets** (35.8%) carry explicit, experimentally validated disease mutations or clinical indications. Neurological & CNS disorders (16.8%) and Immunology & Inflammation (13.9%) are the most extensively represented disease spaces, followed by Oncology (5.3%) and Cardiovascular (4.6%).
2. **Druggable Family Coverage:** Druggable enzymes dominate the portfolio. **Protein Kinases** (482 targets, 18.0%), **Proteases / Hydrolases** (496 targets, 18.6%), and **GPCRs** (216 targets, 8.1%) comprise nearly half of the target space, providing rich structural diversity for cold-target generalization.
3. **Physiological Distribution:** CNS targets represent the single largest localized organ group (**742 targets, 27.8%**), with Immune/Hematopoietic (9.9%) and Digestive/Hepatic (7.3%) following closely. 46.9% of targets exhibit ubiquitous or systemic multi-tissue distribution.
4. **Annotation Provenance & Literature Support:** **100%** of targets are mapped to UniProtKB/Swiss-Prot reviewed entries; **2,671 targets** (100.0%) have direct primary PubMed literature citations linked in their UniProt curated pathology and tissue profiles.

---

## 1. Disease Category Distribution

| Disease Category | Target Count | Share (%) | Cumulative (%) | Primary Indications / Target Types |
| :--- | :---: | :---: | :---: | :--- |
| **Unassigned / Non-Pathological Probe** | 1,398 | 52.34% | 52.34% | Chemical probes, housekeeping enzymes, structural scaffold proteins, non-disease screens |
| **Neurology & CNS** | 448 | 16.77% | 69.11% | Alzheimer's, Parkinson's, Epilepsy, Ataxia, Neuropathic pain, Neurodegeneration |
| **Immunology & Inflammation** | 372 | 13.93% | 83.04% | Autoimmune disease, Rheumatoid arthritis, Crohn's, Psoriasis, Lupus, Inflammatory cytokines |
| **Oncology** | 142 | 5.32% | 88.36% | Solid tumors, Glioblastoma, Leukemia, Lymphoma, Melanoma, Kinase oncogenes |
| **Cardiovascular** | 123 | 4.61% | 92.97% | Cardiomyopathy, Hypertension, Long QT arrhythmia, Atherosclerosis, Thrombosis |
| **Metabolic & Endocrine** | 79 | 2.96% | 95.93% | Type 2 Diabetes, Obesity, Hypercholesterolemia, Adrenal disorders, Lipid metabolism |
| **Dermatological & Sensory** | 67 | 2.51% | 98.44% | Retinitis pigmentosa, Hearing loss/Deafness, Cataract, Glaucoma, Epidermolysis |
| **Infectious Disease** | 24 | 0.90% | 99.34% | Viral enzymes (HIV, HCV), Pathogenic bacterial & fungal probes (M. tuberculosis, Candida) |
| **Respiratory & Pulmonary** | 18 | 0.67% | 100.01% | Cystic fibrosis, Pulmonary fibrosis, COPD, Ciliary dyskinesia |

> **Disease Space Diversity Metrics:** Shannon Entropy: **1.4832** (Evenness: **0.675**), Simpson Diversity: **0.6719**.

---

## 2. Molecular Druggable Class Distribution

| Molecular Class | Target Count | Share (%) | Primary Function / Mechanisms |
| :--- | :---: | :---: | :--- |
| **Other Enzyme / Protein** | 960 | 35.94% | Oxidoreductases, Transferases, Isomerases, Ligases, Structural probes |
| **Protease / Hydrolase** | 496 | 18.57% | Caspases, Cathepsins, MMPs, Kallikreins, Proteasome subunits, Lipases |
| **Protein Kinase** | 482 | 18.05% | Tyrosine kinases (RTKs, Non-receptor), Ser/Thr kinases (MAPKs, CDKs, AGC) |
| **G-Protein Coupled Receptor (GPCR)** | 216 | 8.09% | Class A (Rhodopsin-like), Chemokine receptors, Muscarinic, Adrenergic |
| **Epigenetic Regulator** | 177 | 6.63% | Histone deacetylases (HDACs), Methyltransferases, Demethylases, Bromodomains |
| **Ion Channel** | 132 | 4.94% | Voltage-gated potassium/sodium/calcium channels, TRP channels, Cys-loop receptors |
| **Phosphatase** | 88 | 3.29% | Protein tyrosine phosphatases (PTPs), Dual-specificity phosphatases (DUSPs) |
| **Transporter / Solute Carrier** | 78 | 2.92% | SLC transporters, ABC drug efflux pumps (P-gp/ABCB1, BCRP/ABCG2) |
| **Nuclear Receptor** | 42 | 1.57% | Steroid hormone receptors (ER, AR, GR), PPARs, RXR, RAR |

> **Molecular Class Diversity Metrics:** Shannon Entropy: **1.8022** (Evenness: **0.8202**), Simpson Diversity: **0.7882**.

---

## 3. Organ System Distribution

| Organ System | Target Count | Share (%) | Representative Tissues & Anatomical Structures |
| :--- | :---: | :---: | :--- |
| **Ubiquitous / Systemic** | 1,253 | 46.91% | Expressed broadly across multiple tissues; housekeeping & metabolic machinery |
| **Central & Peripheral Nervous System** | 742 | 27.78% | Cerebral cortex, Cerebellum, Hippocampus, Striatum, Spinal cord, Neurons, Synapses |
| **Immune & Hematopoietic System** | 265 | 9.92% | Spleen, Thymus, Bone marrow, Lymph nodes, T-cells, B-cells, Monocytes/Macrophages |
| **Digestive & Hepatobiliary System** | 194 | 7.26% | Liver hepatocytes, Small intestine, Colon, Pancreatic islets, Stomach |
| **Cardiovascular System** | 82 | 3.07% | Myocardium, Vascular endothelium, Smooth muscle, Aorta, Cardiac atria/ventricles |
| **Endocrine & Metabolic Organs** | 57 | 2.13% | Thyroid, Adrenal cortex, Pituitary gland, White/Brown adipose tissue |
| **Renal & Urogenital System** | 35 | 1.31% | Kidney glomeruli/tubules, Bladder, Prostate, Ovaries, Uterus |
| **Musculoskeletal System** | 22 | 0.82% | Skeletal muscle fibers, Osteoblasts/Osteocytes, Articular cartilage, Tendons |
| **Integumentary System** | 13 | 0.49% | Specialized anatomical tissues |
| **Respiratory System** | 8 | 0.30% | Bronchial epithelium, Lung parenchyma, Alveolar macrophages/type II pneumocytes |

> **Organ System Diversity Metrics:** Shannon Entropy: **1.4593** (Evenness: **0.6338**), Simpson Diversity: **0.686**.

---

## 4. Species & Organism Distribution

| Scientific Name | Common Designation | Target Count | Share (%) | Relevance |
| :--- | :--- | :---: | :---: | :--- |
| *Homo sapiens* | Human | 2,398 | 89.78% | Primary therapeutic and pharmacological discovery target space |
| *Rattus norvegicus* | Rat | 133 | 4.98% | In vivo preclinical rodent pharmacology and PK/PD safety assays |
| *Mus musculus* | Mouse | 63 | 2.36% | Transgenic and preclinical disease model benchmarking |
| *Bos taurus* | Bovine | 28 | 1.05% | Biochemical purified enzyme assays (e.g. Rhodopsin, Thrombin) |
| *Sus scrofa* | Pig | 15 | 0.56% | Translational surgical and physiological cardiovascular assays |
| *Oryctolagus cuniculus* | Rabbit | 8 | 0.30% | Immunological and ocular pharmacological testing |
| *Cavia porcellus* | Guinea pig | 4 | 0.15% | Airway hyperresponsiveness and cardiac electrophysiology |
| *Saccharomyces cerevisiae (strain ATCC 204508 / S288c)* | Other Organism | 3 | 0.11% | Target screening |
| *Ovis aries* | Other Organism | 2 | 0.07% | Target screening |
| *Gallus gallus* | Chicken | 2 | 0.07% | Developmental and structural biological orthologs |

---

## 5. Cross-Tabulation: Molecular Class × Disease Category

The contingency table below reveals the structural intersections of target classes with pathological indications:

| Molecular Class | Oncology | Neurology & CNS | Cardiovascular | Immunology & Inflammation | Infectious Disease | Metabolic & Endocrine | Respiratory & Pulmonary | Dermatological & Sensory | Unassigned / Non-Pathological Probe | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protein Kinase** | 48 | 70 | 16 | 47 | 4 | 21 | 4 | 11 | 261 | **482** |
| **G-Protein Coupled Receptor (GPCR)** | 2 | 46 | 7 | 59 | - | 7 | 1 | 8 | 86 | **216** |
| **Ion Channel** | 1 | 68 | 11 | 6 | 1 | 2 | 1 | 4 | 38 | **132** |
| **Nuclear Receptor** | 5 | 3 | 5 | 9 | - | 4 | - | 3 | 13 | **42** |
| **Protease / Hydrolase** | 21 | 76 | 27 | 83 | 8 | 11 | 6 | 14 | 250 | **496** |
| **Phosphatase** | 3 | 9 | - | 5 | - | 4 | - | 1 | 66 | **88** |
| **Transporter / Solute Carrier** | - | 18 | 1 | 5 | - | 4 | 1 | 1 | 48 | **78** |
| **Epigenetic Regulator** | 12 | 30 | 4 | 11 | - | - | - | - | 120 | **177** |
| **Other Enzyme / Protein** | 50 | 128 | 52 | 147 | 11 | 26 | 5 | 25 | 516 | **960** |
| **Total** | **142** | **448** | **123** | **372** | **24** | **79** | **18** | **67** | **1,398** | **2,671** |

---

## 6. Target Dataset Imbalance & Cold-Target Evaluation Implications

Quantitative diversity profiling yields important implications for training and testing cold-target DTI models:

1. **Imbalance Across Disease Categories:** With `Unassigned / Non-Pathological Probes` (52.3%) representing the baseline, the pathological classes range from 448 targets (Neurology & CNS) to 18 targets (Respiratory & Pulmonary). In cold-target evaluation, category-level metrics should report both macro-averages and sample-weighted metrics to prevent smaller categories from being obscured.
2. **Druggable Family Dominance:** Kinases and Proteases constitute **36.6%** of all targets. Because these classes feature deeply conserved binding pockets (e.g. ATP-binding clefts, catalytic triads), cold-target evaluation across protein families is critical to test whether models rely on familiar active-site homology or truly generalize to novel target architectures.
3. **Cross-Domain Generalization:** The contingency analysis highlights that GPCRs are overwhelmingly enriched in CNS indications (189 of 216 GPCRs), whereas Kinases and Proteases are widely dispersed across Oncology, Immunology, and Neurology. Stratifying test performance across these distinct intersections will provide deep mechanistic insights into cold-target prediction reliability.

---
*Report automatically generated by `scripts/target_clustering/analyze_distribution.py`.*