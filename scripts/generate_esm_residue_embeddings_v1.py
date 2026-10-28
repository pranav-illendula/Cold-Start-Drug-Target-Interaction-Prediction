"""Generate frozen ESM-2 window and residue features for recovered_baseline_v1."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import torch


MODEL_NAME = "esm2_t30_150M_UR50D"
LAYER = 30
WINDOW_LENGTH = 1022
WINDOW_STRIDE = 512
WINDOW_DIM = 1280
RESIDUE_DIM = 640
VERSION_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "recovered_baseline_v1"
)


def sequence_fingerprint(target_sequences):
    digest = hashlib.sha256()
    for index, sequence in enumerate(target_sequences):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(sequence.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def sequence_windows(sequence):
    if not sequence:
        raise ValueError("Target sequences must be nonempty.")
    windows = []
    start = 0
    while True:
        window = sequence[start : start + WINDOW_LENGTH]
        windows.append((start, window))
        if start + len(window) >= len(sequence):
            return windows
        start += WINDOW_STRIDE


def _validate_mapping(target_to_idx, target_sequences):
    if not isinstance(target_to_idx, dict) or not isinstance(target_sequences, list):
        raise ValueError("Expected a mapping dictionary and sequence list.")
    indices = list(target_to_idx.values())
    if any(type(index) is not int for index in indices):
        raise ValueError("Target mapping indices must be integers.")
    if len(set(indices)) != len(indices) or set(indices) != set(range(len(indices))):
        raise ValueError("Target mapping indices must be unique and contiguous from zero.")
    if len(target_sequences) != len(target_to_idx):
        raise ValueError("Target sequence list is not aligned with the mapping.")
    if any(not isinstance(target_id, str) or not target_id.strip()
           for target_id in target_to_idx):
        raise ValueError("Target mapping keys must be nonempty strings.")
    if any(not isinstance(sequence, str) or not sequence.isascii() or not sequence
           for sequence in target_sequences):
        raise ValueError("Every target sequence must be a nonempty ASCII string.")
    return [
        target_id
        for target_id, _ in sorted(target_to_idx.items(), key=lambda item: item[1])
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-to-idx",
        default=str(VERSION_DIR / "target_to_idx.pkl"),
    )
    parser.add_argument(
        "--target-sequences",
        default=str(VERSION_DIR / "target_sequences.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(VERSION_DIR / "esm2"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    map_path = Path(args.target_to_idx)
    sequence_path = Path(args.target_sequences)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "WARNING: pickle loading can execute arbitrary code; "
        "load only trusted mapping and sequence files.",
        flush=True,
    )
    with map_path.open("rb") as source:
        target_to_idx = pickle.load(source)
    with sequence_path.open("rb") as source:
        target_sequences = pickle.load(source)
    target_ids = _validate_mapping(target_to_idx, target_sequences)
    fingerprint = sequence_fingerprint(target_sequences)

    import esm

    print(f"Loading frozen {MODEL_NAME} on {args.device}.", flush=True)
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter()

    start_time = time.time()
    manifest_entries = []
    generated_paths = []
    try:
        with torch.inference_mode():
            for target_idx, (target_id, sequence) in enumerate(
                zip(target_ids, target_sequences)
            ):
                windows = sequence_windows(sequence)
                window_vectors = []
                residue_sums = torch.zeros(
                    (len(sequence), RESIDUE_DIM),
                    dtype=torch.float32,
                )
                residue_counts = torch.zeros(
                    (len(sequence), 1),
                    dtype=torch.float32,
                )
                for offset in range(0, len(windows), args.batch_size):
                    window_batch = windows[offset : offset + args.batch_size]
                    examples = [
                        (f"{target_idx}:{offset + index}", window)
                        for index, (_, window) in enumerate(window_batch)
                    ]
                    _, _, tokens = batch_converter(examples)
                    tokens = tokens.to(args.device)
                    autocast_context = (
                        torch.autocast("cuda", dtype=torch.float16)
                        if args.device == "cuda"
                        else nullcontext()
                    )
                    with autocast_context:
                        output = model(
                            tokens,
                            repr_layers=[LAYER],
                            return_contacts=False,
                        )
                    representations = output["representations"][LAYER].float()
                    for row, (start, window) in enumerate(window_batch):
                        residues = representations[row, 1 : len(window) + 1]
                        if residues.shape != (len(window), RESIDUE_DIM):
                            raise ValueError(
                                "Unexpected ESM residue representation shape."
                            )
                        window_vectors.append(
                            torch.cat(
                                [representations[row, 0], residues.mean(dim=0)]
                            ).cpu()
                        )
                        end = start + len(window)
                        residue_sums[start:end] += residues.cpu()
                        residue_counts[start:end] += 1
                    del output, representations, tokens
                counts = residue_counts
                if (counts == 0).any():
                    raise ValueError(
                        f"Windowing left uncovered residues for target index {target_idx}."
                    )
                window_tensor = torch.stack(window_vectors).to(torch.float16)
                residue_tensor = (residue_sums / counts).to(torch.float16)
                payload = {
                    "target_id": target_id,
                    "target_idx": target_idx,
                    "sequence_length": len(sequence),
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode("ascii")
                    ).hexdigest(),
                    "window_embeddings": window_tensor,
                    "residue_embeddings": residue_tensor,
                }
                destination = output_dir / f"{target_idx}.pt"
                descriptor, temporary = tempfile.mkstemp(
                    dir=output_dir,
                    prefix=f".{target_idx}_",
                    suffix=".tmp",
                )
                os.close(descriptor)
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, destination)
                    generated_paths.append(destination)
                finally:
                    if os.path.isfile(temporary):
                        os.remove(temporary)
                manifest_entries.append(
                    {
                        "target_id": target_id,
                        "target_idx": target_idx,
                        "sequence_length": len(sequence),
                        "window_count": int(window_tensor.shape[0]),
                        "sequence_sha256": payload["sequence_sha256"],
                        "file": destination.name,
                    }
                )
                del payload, window_tensor, residue_tensor
                del residue_sums, residue_counts, window_vectors
                if target_idx % 100 == 0:
                    gc.collect()
                    print(
                        f"Embedded target {target_idx + 1:,}/{len(target_sequences):,}",
                        flush=True,
                    )

        manifest = {
            "dataset_version": "recovered_baseline_v1",
            "model_name": MODEL_NAME,
            "layer": LAYER,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "window_vector_dimension": WINDOW_DIM,
            "residue_vector_dimension": RESIDUE_DIM,
            "window_vector_definition": "Concatenated CLS and masked mean residue vector.",
            "overlap_policy": "Average residue representations over covering windows.",
            "target_mapping_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            "sequence_fingerprint": fingerprint,
            "elapsed_seconds": time.time() - start_time,
            "targets": manifest_entries,
        }
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2)
            destination.write("\n")
    except Exception:
        for path in generated_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(
        f"Saved {len(target_sequences):,} target representations under {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
