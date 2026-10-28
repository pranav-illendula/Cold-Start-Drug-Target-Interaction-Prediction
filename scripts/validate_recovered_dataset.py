"""Validate recovered DTI split CSVs and their target index/sequence pair."""

import argparse
from contextlib import contextmanager
import math
import numbers
import os
import pickle
import sys
import zipfile

import pandas as pd


REQUIRED_COLUMNS = (
    "Canonical_SMILES",
    "Target_Name",
    "UniProt_ID",
    "pIC50",
)
CHUNK_SIZE = 100_000
SPLIT_ARGUMENTS = {
    "train": "train_csv",
    "cold_drug": "cold_drug_csv",
    "cold_target": "cold_target_csv",
    "cold_pair": "cold_pair_csv",
}


def _source_parts(source):
    if "::" not in source:
        return source, None
    archive_path, member_name = source.split("::", 1)
    if not archive_path or not member_name or member_name.startswith(("/", "\\")):
        raise ValueError(
            f"invalid ZIP-member path {source!r}; expected archive.zip::member/path"
        )
    return archive_path, member_name


@contextmanager
def _open_source(source, mode):
    archive_path, member_name = _source_parts(source)
    if member_name is None:
        try:
            with open(archive_path, mode) as stream:
                yield stream
        except OSError as error:
            raise ValueError(f"could not open {source}: {error}") from error
        return

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            try:
                with archive.open(member_name, "r") as stream:
                    yield stream
            except KeyError as error:
                raise ValueError(
                    f"ZIP member {member_name!r} not found in {archive_path!r}"
                ) from error
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"could not open ZIP archive {archive_path!r}: {error}") from error


def _missing_values(series):
    blank = series.astype("string").str.strip().eq("").fillna(False)
    return series.isna() | blank


def validate_mapping(target_to_idx, target_sequences):
    """Return mapping target IDs and validation errors without exposing sequences."""
    errors = []
    target_ids = set()

    if not isinstance(target_to_idx, dict):
        errors.append("target mapping must be a dictionary")
    else:
        invalid_keys = [
            key
            for key in target_to_idx
            if not isinstance(key, str) or not key.strip()
        ]
        if invalid_keys:
            errors.append("target mapping keys must be nonempty strings")
        else:
            target_ids = {key.strip() for key in target_to_idx}

        indices = list(target_to_idx.values())
        if any(
            isinstance(index, bool) or not isinstance(index, numbers.Integral)
            for index in indices
        ):
            errors.append("target mapping indices must be integers")
        else:
            integer_indices = [int(index) for index in indices]
            if set(integer_indices) != set(range(len(target_to_idx))):
                errors.append(
                    "target mapping indices must be unique and contiguous from zero"
                )

    if not isinstance(target_sequences, list):
        errors.append("target sequences must be a list")
    else:
        if not isinstance(target_to_idx, dict) or len(target_sequences) != len(
            target_to_idx
        ):
            errors.append("target sequence count must match target mapping size")
        if any(not isinstance(sequence, str) or not sequence.strip()
               for sequence in target_sequences):
            errors.append("every target sequence must be a nonempty string")

    return target_ids, errors


def _read_pickle(path, label):
    try:
        with _open_source(path, "rb") as source:
            return pickle.load(source)
    except Exception as error:
        raise ValueError(f"could not load {label} from {path}: {error}") from error


def _read_csv_header(path):
    with _open_source(path, "rb") as source:
        return set(pd.read_csv(source, nrows=0).columns)


def validate_split_csv(name, path, mapping_ids):
    errors = []
    missing_counts = {column: 0 for column in REQUIRED_COLUMNS}
    target_ids = set()
    row_count = 0
    invalid_pic50 = 0

    try:
        columns = _read_csv_header(path)
        missing_columns = sorted(set(REQUIRED_COLUMNS) - columns)
        if missing_columns:
            errors.append(
                f"{name}: missing required columns: {', '.join(missing_columns)}"
            )

        available_columns = [column for column in REQUIRED_COLUMNS if column in columns]
        with _open_source(path, "rb") as source:
            for chunk in pd.read_csv(
                source,
                usecols=available_columns,
                chunksize=CHUNK_SIZE,
            ):
                row_count += len(chunk)
                for column in available_columns:
                    missing = _missing_values(chunk[column])
                    missing_counts[column] += int(missing.sum())

                if "UniProt_ID" in chunk:
                    ids = chunk["UniProt_ID"]
                    valid_ids = ids[~_missing_values(ids)].astype("string").str.strip()
                    target_ids.update(valid_ids.tolist())

                if "pIC50" in chunk:
                    raw_values = chunk["pIC50"]
                    missing = _missing_values(raw_values)
                    numeric = pd.to_numeric(raw_values, errors="coerce")
                    invalid = (~missing & numeric.isna()) | (
                        numeric.notna()
                        & ~numeric.map(lambda value: math.isfinite(float(value)))
                    )
                    invalid_pic50 += int(invalid.sum())
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        errors.append(f"{name}: could not read CSV {path}: {error}")
        print(f"{name}: validation unavailable (CSV read failed)")
        return errors, target_ids

    print(f"{name}: rows={row_count}")
    for column in REQUIRED_COLUMNS:
        if column in columns:
            print(f"  missing {column}={missing_counts[column]}")
    if "pIC50" in columns:
        print(f"  invalid_or_nonfinite_pIC50={invalid_pic50}")
    if "UniProt_ID" in columns:
        print(f"  unique_target_ids={len(target_ids)}")

    for column in available_columns:
        if missing_counts[column]:
            errors.append(
                f"{name}: {missing_counts[column]} missing value(s) in {column}"
            )
    if "pIC50" in columns and invalid_pic50:
        errors.append(
            f"{name}: {invalid_pic50} invalid or non-finite pIC50 value(s)"
        )

    missing_target_ids = sorted(target_ids - mapping_ids)
    if missing_target_ids:
        errors.append(
            f"{name}: target IDs absent from mapping: "
            + ", ".join(missing_target_ids)
        )

    return errors, target_ids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cold-drug-csv", required=True)
    parser.add_argument("--cold-target-csv", required=True)
    parser.add_argument("--cold-pair-csv", required=True)
    parser.add_argument("--target-to-idx", required=True)
    parser.add_argument("--target-sequences", required=True)
    parser.epilog = (
        "Each input may be a filesystem path or a ZIP member written as "
        "archive.zip::member/path."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(
        "WARNING: pickle loading can execute arbitrary code; "
        "load only trusted mapping and sequence files.",
        file=sys.stderr,
    )

    errors = []
    try:
        target_to_idx = _read_pickle(args.target_to_idx, "target mapping")
        target_sequences = _read_pickle(args.target_sequences, "target sequences")
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    mapping_ids, mapping_errors = validate_mapping(target_to_idx, target_sequences)
    errors.extend(mapping_errors)
    mapping_size = len(target_to_idx) if isinstance(target_to_idx, dict) else 0
    sequence_size = len(target_sequences) if isinstance(target_sequences, list) else 0
    print(
        f"mapping_entries={mapping_size} sequence_entries={sequence_size} "
        f"mapping_valid={not mapping_errors}"
    )

    all_target_ids = set()
    for split_name, argument_name in SPLIT_ARGUMENTS.items():
        split_errors, split_target_ids = validate_split_csv(
            split_name,
            getattr(args, argument_name),
            mapping_ids,
        )
        errors.extend(split_errors)
        all_target_ids.update(split_target_ids)

    missing_union = sorted(all_target_ids - mapping_ids)
    if missing_union:
        print("target_ids_missing_from_mapping=" + ", ".join(missing_union))
    else:
        print("target_ids_missing_from_mapping=none")

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
