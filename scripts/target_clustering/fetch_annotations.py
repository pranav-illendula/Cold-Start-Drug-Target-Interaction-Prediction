"""Batch fetch target biological annotations from UniProtKB REST API with disk caching."""

import argparse
import json
import os
import pickle
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

# Resolve package paths
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
BINDINGDB_DATA_DIR = os.path.join(DATA_DIR, "bindingdb")
DEFAULT_CACHE_PATH = os.path.join(BINDINGDB_DATA_DIR, "raw_annotations_cache.json")
DEFAULT_TARGET_TO_IDX = os.path.join(DATA_DIR, "target_to_idx.pkl")

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = (
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "lineage",
    "cc_disease",
    "cc_tissue_specificity",
    "cc_similarity",
    "keyword",
    "go",
    "xref_chembl",
)
USER_AGENT = "Antigravity-DTI-Target-Clustering/1.0 (academic research; cold-start drug-target prediction)"


def load_cached_annotations(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"Warning: Could not parse existing cache file ({e}). Starting fresh.")
    return {}


def save_cached_annotations(cache, cache_path, max_retries=5):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, cache_path)
            return
        except PermissionError:
            time.sleep(0.3 * (attempt + 1))
    os.replace(tmp_path, cache_path)


def fetch_uniprot_batch(accessions, max_retries=3, backoff_factor=1.5):
    """Query UniProt REST API for a batch of accessions using field projection."""
    if not accessions:
        return {}

    query_str = " OR ".join(f"accession:{acc}" for acc in accessions)
    params = {
        "query": query_str,
        "format": "json",
        "size": str(max(500, len(accessions) * 2)),
        "fields": ",".join(UNIPROT_FIELDS),
    }
    url = f"{UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            batch_map = {}
            for item in results:
                primary_acc = item.get("primaryAccession")
                if primary_acc:
                    batch_map[primary_acc] = item
                # Also index any secondary accessions if present
                for sec in item.get("secondaryAccessions", []):
                    batch_map[sec] = item
            return batch_map
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            wait_time = backoff_factor ** attempt
            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                print(f"\nBatch query failed after {max_retries} attempts for {len(accessions)} accessions: {err}")
                return {}
        except Exception as err:
            print(f"\nUnexpected error querying UniProt: {err}")
            return {}
    return {}


def fetch_single_uniprot(accession, max_retries=2):
    """Fallback single-entry lookup for accessions not returned by batch search."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return {"primaryAccession": accession, "status": "not_found"}
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return {"primaryAccession": accession, "status": "failed"}


def run_ingestion(target_to_idx_path=DEFAULT_TARGET_TO_IDX, cache_path=DEFAULT_CACHE_PATH, batch_size=50, limit=None):
    if not os.path.exists(target_to_idx_path):
        raise FileNotFoundError(f"Target index file not found: {target_to_idx_path}")

    with open(target_to_idx_path, "rb") as f:
        target_to_idx = pickle.load(f)

    all_accessions = list(target_to_idx.keys())
    if limit is not None and limit > 0:
        all_accessions = all_accessions[:limit]

    cache = load_cached_annotations(cache_path)
    cached_count = len(cache)
    print(f"Total targets to annotate: {len(all_accessions)}")
    print(f"Already cached in {os.path.basename(cache_path)}: {cached_count}")

    missing_accessions = [acc for acc in all_accessions if acc not in cache]
    print(f"Remaining targets to fetch: {len(missing_accessions)}")

    if not missing_accessions:
        print("All targets already cached. No network calls needed.")
        return cache

    # Batch process
    total_batches = (len(missing_accessions) + batch_size - 1) // batch_size
    fetched_new = 0

    print(f"Fetching in {total_batches} batches of up to {batch_size} targets...")
    for b_idx in range(total_batches):
        chunk = missing_accessions[b_idx * batch_size : (b_idx + 1) * batch_size]
        t0 = time.time()
        batch_results = fetch_uniprot_batch(chunk)

        # Record fetched
        for acc in chunk:
            if acc in batch_results:
                cache[acc] = batch_results[acc]
                fetched_new += 1
            else:
                # Fallback to direct lookup
                item = fetch_single_uniprot(acc)
                cache[acc] = item
                fetched_new += 1

        # Periodically persist to disk
        save_cached_annotations(cache, cache_path)
        dt = time.time() - t0
        pct = (b_idx + 1) / total_batches * 100
        print(f"  Batch {b_idx + 1:3d}/{total_batches} ({pct:5.1f}%) | Chunk: {len(chunk):2d} | Total Cached: {len(cache):4d} | Time: {dt:4.1f}s")
        time.sleep(0.05)  # Polite request throttling

    save_cached_annotations(cache, cache_path)
    print(f"\nIngestion finished! Total targets in cache: {len(cache)} ({fetched_new} newly retrieved).")
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-to-idx", default=DEFAULT_TARGET_TO_IDX, help="Path to target_to_idx.pkl")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Output JSON cache path")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for UniProt search API")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of targets (for dry-run/testing)")
    args = parser.parse_args()

    run_ingestion(
        target_to_idx_path=args.target_to_idx,
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
