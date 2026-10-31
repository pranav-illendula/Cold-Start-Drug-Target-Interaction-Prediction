"""Build isolated v4.2 graph datasets for recovered_baseline_v1."""

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_graphs_v5 import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    build_split,
    load_target_indices,
)


VERSION_DIR = PROJECT_DIR / "data" / "recovered_baseline_v1"
DEFAULT_TARGET_MAP = VERSION_DIR / "target_to_idx.pkl"
DEFAULT_TRAIN_FIT = VERSION_DIR / "calibration" / "train_fit.csv"
DEFAULT_CALIBRATION = VERSION_DIR / "calibration" / "calibration.csv"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graphs(graphs, split_name, target_count):
    for graph_number, graph in enumerate(graphs, start=1):
        if graph.x.ndim != 2 or graph.x.shape[1] != ATOM_FEATURE_DIM:
            raise ValueError(
                f"{split_name} graph {graph_number}: expected "
                f"{ATOM_FEATURE_DIM} atom features"
            )
        if graph.edge_attr.shape != (
            graph.edge_index.shape[1],
            BOND_FEATURE_DIM,
        ):
            raise ValueError(
                f"{split_name} graph {graph_number}: bond features do not align "
                "with directed edges"
            )
        if graph.target_idx.numel() != 1:
            raise ValueError(
                f"{split_name} graph {graph_number}: expected one target index"
            )
        target_idx = int(graph.target_idx.item())
        if not 0 <= target_idx < target_count:
            raise ValueError(
                f"{split_name} graph {graph_number}: target index is out of range"
            )
        if (
            not torch.isfinite(graph.x).all()
            or not torch.isfinite(graph.edge_attr).all()
            or not torch.isfinite(graph.y).all()
        ):
            raise ValueError(
                f"{split_name} graph {graph_number}: non-finite graph or label value"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fit-csv", default=str(DEFAULT_TRAIN_FIT))
    parser.add_argument("--calibration-csv", default=str(DEFAULT_CALIBRATION))
    parser.add_argument(
        "--cold-drug-csv",
        required=True,
    )
    parser.add_argument(
        "--cold-target-csv",
        required=True,
    )
    parser.add_argument(
        "--cold-pair-csv",
        required=True,
    )
    parser.add_argument("--target-to-idx", default=str(DEFAULT_TARGET_MAP))
    parser.add_argument(
        "--name-to-uniprot",
        default=str(VERSION_DIR / "target_to_uniprot.pkl"),
    )
    parser.add_argument("--output-dir", default=str(VERSION_DIR / "graphs"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    mapping_path = Path(args.target_to_idx).resolve()
    input_paths = {
        "train_fit": Path(args.train_fit_csv).resolve(),
        "calibration": Path(args.calibration_csv).resolve(),
        "cold_drug": Path(args.cold_drug_csv).resolve(),
        "cold_target": Path(args.cold_target_csv).resolve(),
        "cold_pair": Path(args.cold_pair_csv).resolve(),
    }
    for split_name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split_name} CSV does not exist: {path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Target mapping does not exist: {mapping_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty graph output directory: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "train_fit": "train_fit_data_v4.pt",
        "calibration": "calibration_data_v4.pt",
        "cold_drug": "cold_drug_data_v4.pt",
        "cold_target": "cold_target_data_v4.pt",
        "cold_pair": "cold_pair_data_v4.pt",
    }
    temporary_paths = {}
    manifest_splits = {}
    try:
        direct_ids, name_to_idx = load_target_indices(
            str(mapping_path),
            args.name_to_uniprot,
            None,
            "",
            "",
        )
        target_count = len(direct_ids)
        if not target_count:
            raise ValueError("Target mapping is empty.")

        for split_name, csv_path in input_paths.items():
            print(
                f"Building {split_name} from {csv_path}; "
                "this full-list format requires substantial RAM.",
                flush=True,
            )
            graphs = build_split(
                str(csv_path),
                "Canonical_SMILES",
                "Target_Name",
                "UniProt_ID",
                "pIC50",
                direct_ids,
                name_to_idx,
                {},
            )
            _validate_graphs(graphs, split_name, target_count)

            descriptor, temporary_path = tempfile.mkstemp(
                dir=output_dir,
                prefix=f".{split_name}_",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_paths[split_name] = temporary_path
            torch.save(graphs, temporary_path)
            manifest_splits[split_name] = {
                "source_csv": str(csv_path),
                "source_sha256": _sha256(csv_path),
                "graph_count": len(graphs),
                "output_file": output_names[split_name],
            }
            print(
                f"Validated and serialized {len(graphs):,} {split_name} graphs.",
                flush=True,
            )
            del graphs
            gc.collect()

        manifest = {
            "dataset_version": "recovered_baseline_v1",
            "graph_schema": {
                "atom_feature_dimension": ATOM_FEATURE_DIM,
                "bond_feature_dimension": BOND_FEATURE_DIM,
                "required_fields": [
                    "x",
                    "edge_index",
                    "edge_attr",
                    "target_idx",
                    "y",
                ],
            },
            "target_mapping": str(mapping_path),
            "target_mapping_sha256": _sha256(mapping_path),
            "output_directory": str(output_dir),
            "splits": manifest_splits,
        }
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2)
            destination.write("\n")

        for split_name, filename in output_names.items():
            destination = output_dir / filename
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing graph dataset: {destination}"
                )
            os.replace(temporary_paths.pop(split_name), destination)
            print(f"Saved {destination}", flush=True)
        print(f"Saved run manifest {manifest_path}", flush=True)
    finally:
        for temporary_path in temporary_paths.values():
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)


if __name__ == "__main__":
    main()
