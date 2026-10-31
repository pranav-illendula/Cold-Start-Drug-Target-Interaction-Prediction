"""Build v4 PyTorch Geometric graphs from the four verified split CSVs."""

import argparse
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from bond_features_v4_2 import BOND_FEATURE_DIM, molecule_bond_features


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
EVALUATION_DIR = os.path.join(PROJECT_DIR, "evaluation")
TARGET_TO_IDX_PATH = os.path.join(DATA_DIR, "target_to_idx.pkl")
NAME_TO_UNIPROT_PATH = os.path.join(SPLIT_DIR, "target_to_uniprot.pkl")
ATOMIC_NUMBERS = (6, 7, 8, 16, 15, 9, 17, 35)
ATOM_FEATURE_DIM = 18


def atom_features(atom):
    """Encode atom type (9), degree (6), and formal-charge sign (3)."""
    atomic_number = atom.GetAtomicNum()
    atom_type = (
        ATOMIC_NUMBERS.index(atomic_number)
        if atomic_number in ATOMIC_NUMBERS
        else 8
    )
    degree = min(atom.GetTotalDegree(), 5)
    charge = atom.GetFormalCharge()
    charge_type = 0 if charge < 0 else 2 if charge > 0 else 1
    return (
        *(float(index == atom_type) for index in range(9)),
        *(float(index == degree) for index in range(6)),
        *(float(index == charge_type) for index in range(3)),
    )


def load_target_indices(
    target_to_idx_path,
    name_to_uniprot_path,
    bindingdb_path,
    bindingdb_name_column,
    bindingdb_id_column,
):
    with open(target_to_idx_path, "rb") as source:
        target_to_idx = pickle.load(source)
    if not isinstance(target_to_idx, dict):
        raise ValueError(f"Expected a target-to-index dictionary in {target_to_idx_path}.")
    indices = {int(value) for value in target_to_idx.values()}
    if indices != set(range(len(target_to_idx))):
        raise ValueError("Target indices must be unique and contiguous from zero.")

    name_to_idx = {}
    if os.path.isfile(name_to_uniprot_path):
        with open(name_to_uniprot_path, "rb") as source:
            name_to_uniprot = pickle.load(source)
        if not isinstance(name_to_uniprot, dict):
            raise ValueError(f"Expected a target-name mapping in {name_to_uniprot_path}.")
    elif bindingdb_path:
        columns = [bindingdb_name_column, bindingdb_id_column]
        try:
            chunks = pd.read_csv(
                bindingdb_path,
                sep="\t",
                compression="infer",
                usecols=columns,
                chunksize=200_000,
                low_memory=False,
                on_bad_lines="skip",
            )
            name_to_uniprot = {}
            for chunk in chunks:
                chunk = chunk.dropna(subset=columns)
                for name, uniprot_id in zip(
                    chunk[bindingdb_name_column],
                    chunk[bindingdb_id_column],
                ):
                    name_to_uniprot.setdefault(str(name), str(uniprot_id))
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                "Could not read target names and UniProt IDs from the supplied "
                f"BindingDB file {bindingdb_path!r}: {error}"
            ) from error
    else:
        name_to_uniprot = {}

    for name, uniprot_id in name_to_uniprot.items():
        if uniprot_id in target_to_idx:
            name_to_idx[str(name)] = int(target_to_idx[uniprot_id])

    direct_ids = {str(key): int(value) for key, value in target_to_idx.items()}
    return direct_ids, name_to_idx


def target_index(row, target_column, target_id_column, direct_ids, name_to_idx):
    if target_id_column in row and pd.notna(row[target_id_column]):
        target_id = str(row[target_id_column]).strip()
        if target_id in direct_ids:
            return direct_ids[target_id]
        raise ValueError(
            f"UniProt target identifier {target_id!r} is not in "
            "data/target_to_idx.pkl. Extend target_to_idx.pkl and the aligned "
            "target_sequences.pkl with this UniProt ID before graph construction."
        )

    target_name = str(row[target_column]).strip()
    if target_name in direct_ids:
        return direct_ids[target_name]
    if target_name in name_to_idx:
        return name_to_idx[target_name]
    raise ValueError(
        f"Target {target_name!r} is absent from the target index and target-name "
        "mapping. Provide a valid UniProt ID column or mapping file."
    )


def build_split(
    csv_path,
    smiles_column,
    target_column,
    target_id_column,
    label_column,
    direct_ids,
    name_to_idx,
    molecule_cache,
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {smiles_column, target_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}."
        )

    graphs = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            smiles = str(row[smiles_column]).strip()
            if not smiles or smiles.lower() == "nan":
                raise ValueError("empty SMILES")
            if smiles not in molecule_cache:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError("RDKit could not parse SMILES")
                node_features = [atom_features(atom) for atom in molecule.GetAtoms()]
                edge_index, edge_attr = molecule_bond_features(molecule)
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
                if not edge_attr:
                    edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
                    edge_attr_tensor = torch.empty(
                        (0, BOND_FEATURE_DIM),
                        dtype=torch.float32,
                    )
                molecule_cache[smiles] = (x, edge_index_tensor, edge_attr_tensor)

            x, edge_index, edge_attr = molecule_cache[smiles]
            label = float(row[label_column])
            if not np.isfinite(label):
                raise ValueError(f"non-finite pIC50 label {label!r}")
            idx = target_index(
                row,
                target_column,
                target_id_column,
                direct_ids,
                name_to_idx,
            )
            graphs.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    target_idx=torch.tensor([idx], dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.float32),
                )
            )
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                f"Could not build graph for {csv_path}, CSV row {row_number}: "
                f"{error}"
            ) from error

    if not graphs:
        raise ValueError(f"Split CSV is empty: {csv_path}")
    return graphs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default=os.path.join(SPLIT_DIR, "split_train.csv"))
    parser.add_argument(
        "--cold-drug-csv",
        default=os.path.join(SPLIT_DIR, "split_cold_drug.csv"),
    )
    parser.add_argument(
        "--cold-target-csv",
        default=os.path.join(SPLIT_DIR, "split_cold_target.csv"),
    )
    parser.add_argument(
        "--cold-pair-csv",
        default=os.path.join(SPLIT_DIR, "split_cold_pair.csv"),
    )
    parser.add_argument("--smiles-column", default="Canonical_SMILES")
    parser.add_argument("--target-column", default="Target_Name")
    parser.add_argument("--target-id-column", default="UniProt_ID")
    parser.add_argument("--label-column", default="pIC50")
    parser.add_argument("--target-to-idx", default=TARGET_TO_IDX_PATH)
    parser.add_argument("--name-to-uniprot", default=NAME_TO_UNIPROT_PATH)
    parser.add_argument(
        "--bindingdb-tsv",
        help="Optional BindingDB TSV/ZIP used to map target names to UniProt IDs "
        "when --name-to-uniprot is unavailable.",
    )
    parser.add_argument(
        "--bindingdb-name-column",
        default="UniProt (SwissProt) Recommended Name of Target Chain 1",
    )
    parser.add_argument(
        "--bindingdb-id-column",
        default="UniProt (SwissProt) Primary ID of Target Chain 1",
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--evaluation-dir", default=EVALUATION_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    direct_ids, name_to_idx = load_target_indices(
        args.target_to_idx,
        args.name_to_uniprot,
        args.bindingdb_tsv,
        args.bindingdb_name_column,
        args.bindingdb_id_column,
    )
    split_csvs = {
        "train": args.train_csv,
        "cold_drug": args.cold_drug_csv,
        "cold_target": args.cold_target_csv,
        "cold_pair": args.cold_pair_csv,
    }
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.evaluation_dir, exist_ok=True)
    output_paths = {
        "train": os.path.join(args.data_dir, "train_data_v4.pt"),
        "cold_drug": os.path.join(args.evaluation_dir, "cold_drug_data_v4.pt"),
        "cold_target": os.path.join(args.evaluation_dir, "cold_target_data_v4.pt"),
        "cold_pair": os.path.join(args.evaluation_dir, "cold_pair_data_v4.pt"),
    }
    temporary_paths = {}
    try:
        for split_name, csv_path in split_csvs.items():
            molecule_cache = {}
            graphs = build_split(
                csv_path,
                args.smiles_column,
                args.target_column,
                args.target_id_column,
                args.label_column,
                direct_ids,
                name_to_idx,
                molecule_cache,
            )
            for graph in graphs:
                if graph.x.shape[1] != ATOM_FEATURE_DIM:
                    raise ValueError("Generated atom feature width is not 18.")
                if graph.edge_attr.shape != (
                    graph.edge_index.shape[1],
                    BOND_FEATURE_DIM,
                ):
                    raise ValueError(
                        "Generated bond features do not align with edge_index."
                    )
                if (
                    not torch.isfinite(graph.x).all()
                    or not torch.isfinite(graph.edge_attr).all()
                ):
                    raise ValueError(
                        f"Non-finite graph features found in {split_name}."
                    )

            output_dir = os.path.dirname(output_paths[split_name])
            with tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix=f".{split_name}_v4_",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_paths[split_name] = temporary.name
            torch.save(graphs, temporary_paths[split_name])
            print(
                f"{split_name}: built and validated {len(graphs):,} graphs "
                f"({len(molecule_cache):,} unique SMILES)"
            )
            del graphs

        for split_name, output_path in output_paths.items():
            os.replace(temporary_paths.pop(split_name), output_path)
            print(f"saved {output_path}")
    finally:
        for temporary_path in temporary_paths.values():
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)


if __name__ == "__main__":
    main()
