"""Unified Pipeline Orchestrator for Target Clustering and Biological Profiling.

Provides a single CLI entrypoint to run, inspect, and reproduce all stages of
Requirement 1: Disease & Organ-Level Target Clustering for BindingDB.

Pipeline Stages:
1. Ingestion: Fetch raw UniProtKB biological annotations with local disk cache.
2. Classification: Standardize targets into controlled Disease Categories, Organ Systems, and Molecular Classes.
3. Profiling: Compute distributions, cross-tabulations, and Shannon/Simpson diversity metrics.
4. Evaluation: Run category-stratified cold-target evaluation benchmarks.

Usage:
    # Run entire pipeline end-to-end:
    python scripts/target_clustering/run_pipeline.py --all

    # Check status and integrity of all pipeline artifacts:
    python scripts/target_clustering/run_pipeline.py --status

    # Run individual stages:
    python scripts/target_clustering/run_pipeline.py --fetch
    python scripts/target_clustering/run_pipeline.py --classify
    python scripts/target_clustering/run_pipeline.py --analyze
    python scripts/target_clustering/run_pipeline.py --evaluate
"""

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.target_clustering import (
    BINDINGDB_DATA_DIR,
    BINDINGDB_RESULTS_DIR,
    TARGET_TO_IDX_PATH,
)
from scripts.target_clustering.fetch_annotations import run_ingestion
from scripts.target_clustering.classify_targets import process_all_targets, save_target_annotations, print_summary_report
from scripts.target_clustering.analyze_distribution import run_distribution_analysis
from scripts.target_clustering.evaluate_stratified_cold_target import run_stratified_evaluation

CACHE_PATH = os.path.join(BINDINGDB_DATA_DIR, "raw_annotations_cache.json")
CSV_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_bindingdb.csv")
PARQUET_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_bindingdb.parquet")
LOOKUP_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_lookup.pkl")
DIST_REPORT_PATH = os.path.join(BINDINGDB_RESULTS_DIR, "target_distribution_report.md")
DIST_SUMMARY_JSON = os.path.join(BINDINGDB_RESULTS_DIR, "target_distribution_summary.json")
BENCHMARK_MD = os.path.join(BINDINGDB_RESULTS_DIR, "cold_target_category_benchmark.md")
BENCHMARK_JSON = os.path.join(BINDINGDB_RESULTS_DIR, "cold_target_category_benchmark.json")
BENCHMARK_CSV = os.path.join(BINDINGDB_RESULTS_DIR, "cold_target_category_benchmark.csv")


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def check_status():
    """Audit the existence and size of all pipeline artifacts."""
    print_banner("Requirement 1 Pipeline Artifact Status")
    artifacts = [
        ("Input Target Index", TARGET_TO_IDX_PATH, "Prerequisite 2,671 BindingDB index mapping"),
        ("Raw UniProt Cache", CACHE_PATH, "Stage 1: Raw JSON API responses"),
        ("Target Annotations CSV", CSV_PATH, "Stage 2: Standardized CSV table"),
        ("Target Annotations Parquet", PARQUET_PATH, "Stage 2: Compressed Parquet table"),
        ("Target Annotations Lookup", LOOKUP_PATH, "Stage 2: Fast pickle dictionary (0..2670)"),
        ("Distribution Markdown Report", DIST_REPORT_PATH, "Stage 3: Publication-ready distribution report"),
        ("Distribution Summary JSON", DIST_SUMMARY_JSON, "Stage 3: Structured diversity metrics JSON"),
        ("Cold-Target Benchmark Report", BENCHMARK_MD, "Stage 4: Stratified cold-target evaluation report"),
        ("Cold-Target Benchmark JSON", BENCHMARK_JSON, "Stage 4: Detailed category evaluation metrics JSON"),
        ("Cold-Target Benchmark CSV", BENCHMARK_CSV, "Stage 4: Stratified metrics summary table"),
    ]

    all_present = True
    for name, path, desc in artifacts:
        exists = os.path.isfile(path)
        if exists:
            size_kb = os.path.getsize(path) / 1024
            status_str = f"[OK] ({size_kb:>8.1f} KB)"
        else:
            status_str = "[MISSING]"
            all_present = False
        print(f"  {status_str:<18} {name:<30} | {desc}")

    print("-" * 75)
    if all_present:
        print("  Status: All pipeline artifacts are generated, verified, and ready.")
    else:
        print("  Status: Some artifacts are missing. Run with --all to generate them.")
    print("=" * 75 + "\n")


def stage_fetch(batch_size: int = 50, limit: int = None):
    print_banner("Stage 1: Ingesting Annotations from UniProtKB")
    t0 = time.time()
    cache = run_ingestion(
        target_to_idx_path=TARGET_TO_IDX_PATH,
        cache_path=CACHE_PATH,
        batch_size=batch_size,
        limit=limit,
    )
    dt = time.time() - t0
    print(f"\n[Stage 1 Complete] Cached {len(cache):,d} targets in {dt:.1f}s.")


def stage_classify():
    print_banner("Stage 2: Classifying Targets & Building Canonical Schemas")
    t0 = time.time()
    df = process_all_targets(cache_path=CACHE_PATH, target_to_idx_path=TARGET_TO_IDX_PATH)
    save_target_annotations(df, csv_path=CSV_PATH, parquet_path=PARQUET_PATH, lookup_path=LOOKUP_PATH)
    print_summary_report(df)
    dt = time.time() - t0
    print(f"[Stage 2 Complete] Classified {len(df):,d} targets into CSV/Parquet/Lookup in {dt:.1f}s.")


def stage_analyze():
    print_banner("Stage 3: Profiling Biological Distributions & Imbalances")
    t0 = time.time()
    run_distribution_analysis(
        csv_path=CSV_PATH,
        report_path=DIST_REPORT_PATH,
        summary_path=DIST_SUMMARY_JSON,
    )
    dt = time.time() - t0
    print(f"[Stage 3 Complete] Distribution analysis & reports generated in {dt:.1f}s.")


def stage_evaluate(predictions_file: str = None):
    print_banner("Stage 4: Evaluating Stratified Cold-Target Generalization")
    t0 = time.time()
    run_stratified_evaluation(
        lookup_path=LOOKUP_PATH,
        predictions_path=predictions_file,
        report_path=BENCHMARK_MD,
        json_path=BENCHMARK_JSON,
        csv_path=BENCHMARK_CSV,
    )
    dt = time.time() - t0
    print(f"[Stage 4 Complete] Stratified benchmark reports generated in {dt:.1f}s.")


def run_full_pipeline(predictions_file: str = None):
    t_start = time.time()
    print_banner("Executing Full Requirement 1 Pipeline (Stages 1 - 4)")

    stage_fetch()
    stage_classify()
    stage_analyze()
    stage_evaluate(predictions_file=predictions_file)

    total_time = time.time() - t_start
    print_banner(f"Requirement 1 Pipeline Completed Successfully ({total_time:.1f}s Total)")
    check_status()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Execute all pipeline stages sequentially")
    parser.add_argument("--status", action="store_true", help="Check status and integrity of all pipeline artifacts")
    parser.add_argument("--fetch", action="store_true", help="Run Stage 1: UniProtKB annotation ingestion")
    parser.add_argument("--classify", action="store_true", help="Run Stage 2: Target classification & schema generation")
    parser.add_argument("--analyze", action="store_true", help="Run Stage 3: Target distribution & imbalance profiling")
    parser.add_argument("--evaluate", action="store_true", help="Run Stage 4: Stratified cold-target evaluation")
    parser.add_argument("--predictions-file", default=None, help="Optional custom predictions file for Stage 4")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for UniProt REST API (Stage 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of targets (testing only)")

    args = parser.parse_args()

    # Default action if no flags provided: show status
    if not (args.all or args.status or args.fetch or args.classify or args.analyze or args.evaluate):
        check_status()
        parser.print_help()
        return

    if args.status:
        check_status()

    if args.all:
        run_full_pipeline(predictions_file=args.predictions_file)
        return

    if args.fetch:
        stage_fetch(batch_size=args.batch_size, limit=args.limit)
    if args.classify:
        stage_classify()
    if args.analyze:
        stage_analyze()
    if args.evaluate:
        stage_evaluate(predictions_file=args.predictions_file)


if __name__ == "__main__":
    main()
