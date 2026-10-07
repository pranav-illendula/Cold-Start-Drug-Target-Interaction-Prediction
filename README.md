# Cold-Start-Drug-Target-Interaction-Prediction-with-a-Two-Tower-GNN
Predicts drug–target binding affinity (pIC50) on BindingDB in the cold-start setting, where drugs, targets, or pairs are unseen. A two-tower model pairs a GCN drug encoder with a partially fine-tuned ESM-2 protein encoder, using range-weighted loss to fix poor accuracy on extreme affinities and unseen targets.
