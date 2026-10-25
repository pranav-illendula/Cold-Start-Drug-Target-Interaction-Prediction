# Cold-Start Drug–Target Affinity Prediction with Bond-Aware GINE and Gated Fusion

**Short research-paper draft — results and implementation details from the current project snapshot**

## Abstract

Drug–target affinity models are often evaluated with random pair-level splits, where a drug or protein in the test set may already occur during training. Such evaluation can overestimate performance in practical cold-start settings. We developed a two-tower model for pIC50 regression on BindingDB and evaluated it on scaffold-separated cold-drug, target-held-out cold-target, and joint cold-pair splits. The current v4 model encodes molecular graphs using four residual Graph Isomorphism Network with Edge features (GINE) layers and encodes target sequences using frozen ESM-2 150M embeddings. A learned gate combines drug, target, compatibility, discrepancy, and adaptive mixture features. Training uses capped pIC50-bin-weighted Smooth L1 loss, AdamW, warmup followed by cosine learning-rate decay, a target-held-out validation subset, and early stopping. On the evaluated cold-start sets, v4 achieved MAE values of 0.866, 1.062, and 1.025 pIC50 units for cold-drug, cold-target, and cold-pair prediction, respectively. The largest residual challenge is generalization to high-potency labels and unseen targets. These results are promising but are limited by inconsistent dataset row counts across project artifacts, and differing evaluation samples between model versions.

## 1. Data source and problem definition

Drug–target affinity prediction estimates a continuous binding measurement for a compound and a protein. This project selects BindingDB, a public repository of experimentally measured drug–target affinities, and focuses on IC50 records. IC50 values are converted to pIC50 as `pIC50 = 9 − log10(IC50 in nM)`, so larger values indicate greater potency. Censored and out-of-range measurements are excluded in the documented preprocessing, and repeated drug–target measurements are aggregated by their median.

BindingDB data and downloads are available at [BindingDB](https://www.bindingdb.org/) and its [download page](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp). The project report describes 998,586 unique drug–target pairs following aggregation, with a pIC50 range of approximately 3.0–13.3.

The prediction task takes a drug molecular graph and a target protein sequence and returns one pIC50 estimate. A conventional random split of pair rows is inadequate for assessing discovery of new compounds or targets: the same drug or protein can occur in both training and test data, enabling entity-specific memorization. This project instead constructs four partitions from drug-scaffold and target-group assignments:

| Partition | Drug status | Target status | Purpose | Rows in rebuilt v4 graphs |
|---|---|---|---|---:|
| Train | Seen | Seen | Fit model parameters | 574,367 |
| Cold-drug | Unseen scaffold/drug | Seen | Test chemical generalization | 139,724 |
| Cold-target | Seen | Unseen target group | Test protein generalization | 123,444 |
| Cold-pair | Unseen | Unseen | Test joint generalization | 31,839 |

The project split procedure assigns each drug a single primary Bemis–Murcko scaffold before splitting and checks for drug/target overlap; the project documentation reports zero overlap in its leakage checks. Cold-drug performance estimates transfer to new chemical scaffolds. Cold-target performance tests protein-sequence transfer to targets absent from training. Cold-pair is the strictest quadrant, requiring inference for both an unseen compound and an unseen target.



## 2. Model architecture and training

The current v4 model follows a two-tower design. The drug tower projects atom inputs to 256 channels and applies four residual GINE convolutional blocks. At each layer, an atom updates its representation by aggregating learned messages from its neighbors, where each message depends on both the neighboring atom and their bond feature; a two-layer MLP transforms the aggregate, dropout is applied, and a residual connection plus LayerNorm stabilizes the update. Unlike a conventional GCN that aggregates atom features without explicit bond descriptors, GINE can distinguish chemical connections and substructures.

Each atom has 18 categorical features: a nine-way element encoding (C, N, O, S, P, F, Cl, Br, or other), a six-way total-degree encoding (0–4 or 5+), and a three-way formal-charge sign encoding (negative, neutral, positive). Every directed bond has 14 features: five bond-type indicators (single, double, triple, aromatic, dative), conjugation and ring flags, six stereochemistry indicators, and a direction flag for dative bonds. Each chemical bond is represented as two directed edges; the direction flag distinguishes donor-to-acceptor orientation for dative bonds. Mean pooling summarizes typical atom context, max pooling retains strongly activated substructures, and sum pooling captures size-dependent accumulated signal. Their concatenation is projected to the 256-dimensional drug vector. The graph builder caches molecular graphs by SMILES and validates feature alignment before saving data.

For the target tower, frozen ESM-2 150M produces a 640-dimensional representation. The model concatenates the CLS vector and mean-pooled residue vector to form a 1,280-dimensional target representation, then projects it to 256 dimensions. Freezing ESM avoids backpropagating through the large language model during affinity training; cached embeddings can be reused. The protein sequence limit is 1,024 residues.

The fusion module is more expressive than simple concatenation. For drug vector **d** and target vector **p**, it derives elementwise compatibility (**d ⊙ p**) and absolute discrepancy (**|d − p|**), then uses a learned sigmoid gate to form a sample-specific mixture. The regression head receives all five 256-dimensional components—drug, target, compatibility, discrepancy, and gated mixture—concatenated into 1,280 dimensions, followed by fully connected layers (512 → 256 → 1), LayerNorm, GELU, and dropout.

The model is trained with training-only label standardization and Smooth L1 loss weighted by capped inverse-square-root pIC50-bin frequency. This increases the contribution of rare label bins while limiting extreme weights; validation and test metrics remain unweighted. AdamW uses a learning rate of 5×10⁻⁵ and weight decay of 10⁻⁴. A 10% linear warmup gradually raises the learning rate at the beginning of training; cosine decay then reduces it smoothly to support convergence. A deterministic target-held-out validation split reserves 15% of training targets. Early stopping monitors validation MAE: training stops after eight consecutive epochs without at least 0.001 improvement, up to 100 epochs. The selected checkpoint was epoch 7, with validation MAE 1.0098.

## 3. ESM adaptation across versions

The v2 model used frozen ESM-2 embeddings. The v3 design computed ESM representations live so the target encoder could adapt to the affinity task, alongside changes to loss weighting, batching, and normalization. The exact trainable ESM layers in the final stable v3 checkpoint are not documented, so the safest description is **live/task-adapted ESM**, not confirmed full-model fine-tuning. The reported cold-start results are:

| Split | v2 RMSE | v2 MAE | v2 Pearson *r* | v3 stable RMSE | v3 stable MAE | v3 stable Pearson *r* | Change in *r* (v3 − v2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold-drug | 1.2535 | 0.9956 | 0.6540 | 1.4835 | 1.2105 | 0.5528 | −0.1012 |
| Cold-target | 1.5866 | 1.2769 | 0.2973 | 1.8930 | 1.5363 | 0.2223 | −0.0750 |
| Cold-pair | 1.5454 | 1.2449 | 0.3208 | 1.8355 | 1.4740 | 0.2357 | −0.0851 |

Across these reported comparisons, the v3 run did not produce a measurable improvement over v2: RMSE and MAE increased and Pearson correlation decreased in all three settings. This is an observation about the **combined v3 training recipe**, not evidence that ESM fine-tuning alone caused the decline. Although the reports use the same stated sample counts, a controlled causal comparison would require identical examples, seeds, optimization budgets, and all other model settings held fixed.

| Possible explanation | Why it could matter | Evidence status |
|---|---|---|
| Multiple simultaneous changes | v3 changed live ESM computation, loss weighting, target-grouped batching, and normalization; their effects cannot be separated from these runs. | Documented confound |
| Limited target diversity / weak cold-target transfer | Fine-tuning on available training targets may adapt representations to seen-target patterns without improving transfer to held-out targets. | Plausible; not directly tested |
| Optimization and normalization instability | An earlier v3 run produced extreme predictions linked to degenerate BatchNorm statistics; LayerNorm fixed the instability, but optimization may still have been difficult. | Earlier failure and fix documented; contribution to final scores untested |
| Rare high-potency labels | Few examples in the highest pIC50 bins make task adaptation and weighted gradients noisy. | Supported by project label counts and bin-level errors |
| No isolated ESM ablation | The project did not compare frozen and fine-tuned ESM while keeping the rest of the pipeline identical. | Key limitation |

Accordingly, v4 returns to **frozen** ESM-2 150M embeddings and focuses changes on the drug encoder, interaction fusion, data validation, and training selection. This makes the ESM representation reusable and avoids backpropagation through the language model, but does not establish that frozen ESM is universally superior.

### Cold-start results from the project and reference study

The earlier and current v4 results below are from saved evaluation reports. The current v4 results use the newly rebuilt graph data and a different target vocabulary. The reference paper reports Davis pKd and KIBA scores, not the BindingDB pIC50 endpoint used here. Its results are therefore shown separately; their RMSE values are **not directly comparable** with the project's pIC50 RMSE.

| Evaluation | Split | N | RMSE | MAE | Pearson *r* | CI |
|---|---|---:|---:|---:|---:|---:|
| Earlier v4 | Cold-drug | 138,556 | 1.1295 | 0.8896 | 0.6021 | Not reported |
| Earlier v4 | Cold-target | 123,438 | 1.3041 | 1.0410 | 0.3210 | Not reported |
| Earlier v4 | Cold-pair | 31,839 | 1.2491 | 0.9992 | 0.3879 | Not reported |
| Current v4 | Cold-drug | 139,724 | 1.1188 | 0.8657 | 0.6321 | 0.7247 |
| Current v4 | Cold-target | 123,444 | 1.3386 | 1.0623 | 0.3542 | 0.6219 |
| Current v4 | Cold-pair | 31,839 | 1.2854 | 1.0253 | 0.4191 | 0.6391 |
| GraESM-FuseDTA, Davis | Cold-drug | Not stated here | 0.716 | Not stated | 0.590 | 0.754 |
| GraESM-FuseDTA, Davis | Cold-target | Not stated here | 0.649 | Not stated | 0.760 | 0.847 |
| GraESM-FuseDTA, Davis | Strict cold-pair | Not stated here | 0.704 | Not stated | 0.866 | 0.912 |
| GraESM-FuseDTA, KIBA | Cold-drug | Not stated here | 0.554 | Not stated | 0.790 | 0.805 |
| GraESM-FuseDTA, KIBA | Cold-target | Not stated here | 0.610 | Not stated | 0.699 | 0.725 |
| GraESM-FuseDTA, KIBA | Strict cold-pair | Not stated here | 0.607 | Not stated | 0.686 | 0.712 |

Compared with the earlier v4 evaluation, current v4 correlation improved on all three splits (+0.030 cold-drug, +0.033 cold-target, +0.031 cold-pair). Absolute error improved modestly on cold-drug but worsened on cold-target and cold-pair. Because sample counts and target-index artifacts differ, this is a descriptive comparison, not a controlled ablation or statistically tested improvement. The current v4 report also shows that high-potency errors remain large: for pIC50 10.5–12, MAE is 2.50 (cold-drug), 3.63 (cold-pair), and 4.08 (cold-target), with very small sample counts in these bins.

## 4. How this project differs from GraESM-FuseDTA

The reference study, He (2026), introduced GraESM-FuseDTA and motivated several v4 design choices: residual edge-aware GINE rather than GCN, frozen ESM-2 150M, mean/max/sum graph pooling, gated multimodal fusion using compatibility and discrepancy features, warmup plus cosine decay, and drug-cold, target-cold, and strict pair-cold evaluation. Its experiments use the established Davis and KIBA benchmarks, five-fold evaluation, and additional interpretability and docking analyses. Its ablations report benefits of GINE over a GCN variant and of gated fusion over simple concatenation on those benchmarks.

| Aspect | This project | GraESM-FuseDTA reference study |
|---|---|---|
| Affinity data | BindingDB IC50 transformed to pIC50; repeated pairs aggregated by median | Davis and KIBA benchmark affinity labels |
| Split design | Bemis–Murcko scaffold grouping plus target-group holdout, producing train, cold-drug, cold-target, and cold-pair quadrants | Five-fold warm-start plus drug-cold, target-cold, and strict pair-cold evaluation |
| Validation / selection | Separate target-held-out validation subset and early stopping on validation MAE | Inner validation within its fold-based protocol |
| Rare labels | Capped inverse-frequency pIC50-bin weighted Smooth L1 objective and per-bin reporting | No corresponding pIC50-bin weighting is described in the cited methods |
| Graph features | 18 categorical atom inputs and 14 directed bond features, including an explicit dative-bond direction indicator | Richer atom descriptors and bond type, conjugation, ring, and stereochemistry descriptors |
| Additional analyses | Cold-start regression metrics and pIC50-bin errors; no docking or sequence-attribution experiment in this project | Includes ablation, sequence attribution, and a docking case study |
| Efficiency claim | Frozen ESM embeddings are cached; no matched hardware benchmark was performed | Reports measured parameter, memory, training-speed, and inference analyses |

This project adapts the reference architecture to a BindingDB pIC50 workflow and emphasizes scaffold-aware cold-start evaluation, validation-based checkpoint selection, and rare-label weighting. Its processed BindingDB source is larger in intended coverage than either individual Davis or KIBA benchmark, but the current v4 graph totals do not match the documented split totals; dataset-scale superiority should not be claimed until this discrepancy is resolved. Both models use frozen ESM-2, so this project should not be described as lighter or faster than GraESM-FuseDTA without a same-hardware, same-workload comparison. The different datasets, affinity scales, split procedures, and evaluation samples also mean the reference paper's scores are contextual, not direct head-to-head results.

## 5. Limitations, conclusion, and future scope

The current results indicate that cold-drug prediction is stronger than cold-target or cold-pair prediction, and that ranking correlation can improve even when MAE/RMSE do not. Rare high-potency observations remain insufficient for reliable extrapolation. 

In conclusion, this project combines a bond-aware residual GINE drug encoder, frozen contextual protein embeddings, and adaptive gated fusion for pIC50 prediction under cold-start conditions. It adds an explicit validation-based model-selection procedure and rare-bin loss weighting to a large BindingDB workflow. The observed cold-drug performance is encouraging, while unseen-target generalization and high-potency accuracy remain open challenges.

Future work should (1) reconcile source, split, and graph row counts and publish exact filtering provenance; (2) compare versions on identical rows with multiple random seeds and confidence intervals; (3) cluster targets by sequence identity for stricter leakage control; (4) run ablations that isolate bond features, fusion, weighting, frozen versus fine-tuned ESM, and learning-rate schedules; (5) expand diverse high-pIC50 training data and consider censored-measurement methods; and (6) measure runtime and peak memory on matched hardware before making efficiency comparisons.

## References

1. He, J. (2026). GraESM-FuseDTA: adaptive gated multimodal fusion of graph neural networks and protein language models for robust drug-target affinity prediction. *Naunyn-Schmiedeberg's Archives of Pharmacology*. https://doi.org/10.1007/s00210-026-05732-9
2. Gilson, M. K., et al. (2016). BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology. *Nucleic Acids Research*, 44(D1), D1045–D1053. https://www.bindingdb.org/
3. Hu, W., et al. (2020). Strategies for pre-training graph neural networks. *International Conference on Learning Representations*.
4. Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379, 1123–1130.
5. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887–2893.
