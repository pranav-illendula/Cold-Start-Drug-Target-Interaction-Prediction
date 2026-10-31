"""Append split UniProt targets missing from the sequence/index artifacts."""

import argparse
import os
import pickle
import re
import tempfile

import pandas as pd
import requests


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
DEFAULT_SPLITS = {
    "train": os.path.join(SPLIT_DIR, "split_train.csv"),
    "cold-drug": os.path.join(SPLIT_DIR, "split_cold_drug.csv"),
    "cold-target": os.path.join(SPLIT_DIR, "split_cold_target.csv"),
    "cold-pair": os.path.join(SPLIT_DIR, "split_cold_pair.csv"),
}
UNIPROT_ID_PATTERN = re.compile(r"^[A-Z0-9]+$")


def load_split_target_ids(csv_paths, target_id_column):
    target_ids = set()
    for split_name, csv_path in csv_paths.items():
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{split_name} split CSV not found: {csv_path}")
        try:
            chunks = pd.read_csv(
                csv_path,
                usecols=[target_id_column],
                dtype={target_id_column: str},
                chunksize=200_000,
            )
        except ValueError as error:
            raise ValueError(
                f"{csv_path} must contain a {target_id_column!r} column."
            ) from error

        for chunk in chunks:
            values = chunk[target_id_column].dropna().str.strip()
            target_ids.update(value for value in values if value)

    invalid_ids = sorted(
        target_id
        for target_id in target_ids
        if not UNIPROT_ID_PATTERN.fullmatch(target_id)
    )
    if invalid_ids:
        raise ValueError(
            "Invalid UniProt IDs found in split CSVs: "
            + ", ".join(invalid_ids[:20])
        )
    if not target_ids:
        raise ValueError("No UniProt IDs were found in the split CSVs.")
    return target_ids


def fetch_uniprot_sequence(uniprot_id):
    response = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
        timeout=30,
    )
    response.raise_for_status()
    lines = response.text.splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"UniProt returned invalid FASTA for {uniprot_id}.")
    sequence = "".join(line.strip() for line in lines[1:]).upper()
    if not sequence or not sequence.isalpha() or not sequence.isascii():
        raise ValueError(f"UniProt returned an invalid sequence for {uniprot_id}.")
    return sequence


def extend_target_index(target_to_idx, target_sequences, target_ids, sequence_fetcher):
    if not isinstance(target_to_idx, dict):
        raise ValueError("target_to_idx.pkl must contain a dictionary.")
    if not isinstance(target_sequences, list):
        raise ValueError("target_sequences.pkl must contain a list.")

    try:
        indices = [int(index) for index in target_to_idx.values()]
    except (TypeError, ValueError) as error:
        raise ValueError("Target indices must be integers.") from error
    if (
        len(indices) != len(set(indices))
        or set(indices) != set(range(len(target_to_idx)))
    ):
        raise ValueError("Target indices must be unique and contiguous from zero.")
    if len(target_sequences) != len(target_to_idx):
        raise ValueError(
            "target_to_idx.pkl and target_sequences.pkl have different lengths."
        )
    if any(
        not isinstance(sequence, str) or not sequence
        for sequence in target_sequences
    ):
        raise ValueError("Every existing target sequence must be a non-empty string.")

    updated_mapping = {
        str(uniprot_id): int(index)
        for uniprot_id, index in target_to_idx.items()
    }
    updated_sequences = list(target_sequences)
    missing_ids = sorted(set(target_ids) - set(updated_mapping))
    for uniprot_id in missing_ids:
        sequence = sequence_fetcher(uniprot_id)
        if (
            not isinstance(sequence, str)
            or not sequence
            or not sequence.isalpha()
            or not sequence.isascii()
        ):
            raise ValueError(
                f"Could not obtain a valid protein sequence for {uniprot_id}."
            )
        updated_mapping[uniprot_id] = len(updated_sequences)
        updated_sequences.append(sequence.upper())

    return updated_mapping, updated_sequences, missing_ids


def save_pickle_pair(mapping_path, sequence_path, target_to_idx, target_sequences):
    temporary_paths = []
    try:
        for path, value in (
            (mapping_path, target_to_idx),
            (sequence_path, target_sequences),
        ):
            output_dir = os.path.dirname(os.path.abspath(path))
            os.makedirs(output_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=".target_index_",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_paths.append((destination.name, path))
                pickle.dump(value, destination, protocol=pickle.HIGHEST_PROTOCOL)

        for temporary_path, output_path in temporary_paths:
            os.replace(temporary_path, output_path)
        temporary_paths.clear()
    finally:
        for temporary_path, _ in temporary_paths:
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default=DEFAULT_SPLITS["train"])
    parser.add_argument("--cold-drug-csv", default=DEFAULT_SPLITS["cold-drug"])
    parser.add_argument("--cold-target-csv", default=DEFAULT_SPLITS["cold-target"])
    parser.add_argument("--cold-pair-csv", default=DEFAULT_SPLITS["cold-pair"])
    parser.add_argument("--target-id-column", default="UniProt_ID")
    parser.add_argument(
        "--target-to-idx",
        default=os.path.join(DATA_DIR, "target_to_idx.pkl"),
    )
    parser.add_argument(
        "--target-sequences",
        default=os.path.join(DATA_DIR, "target_sequences.pkl"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = {
        "train": args.train_csv,
        "cold-drug": args.cold_drug_csv,
        "cold-target": args.cold_target_csv,
        "cold-pair": args.cold_pair_csv,
    }
    target_ids = load_split_target_ids(csv_paths, args.target_id_column)

    with open(args.target_to_idx, "rb") as source:
        target_to_idx = pickle.load(source)
    with open(args.target_sequences, "rb") as source:
        target_sequences = pickle.load(source)

    updated_mapping, updated_sequences, added_ids = extend_target_index(
        target_to_idx,
        target_sequences,
        target_ids,
        fetch_uniprot_sequence,
    )
    if added_ids:
        save_pickle_pair(
            args.target_to_idx,
            args.target_sequences,
            updated_mapping,
            updated_sequences,
        )
        print(
            f"Added {len(added_ids)} UniProt targets to the aligned index and "
            f"sequence files: {', '.join(added_ids)}"
        )
    else:
        print("All split UniProt IDs already exist in the target index.")


if __name__ == "__main__":
    main()
