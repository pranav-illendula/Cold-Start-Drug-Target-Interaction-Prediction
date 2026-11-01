"""Classify and standardize drug targets into canonical biological schemas.

Maps targets from UniProtKB raw annotations into:
1. Controlled Disease Categories (9 classes)
2. Controlled Organ Systems (10 classes)
3. Controlled Molecular Classes (9 classes)

Generates:
- data/bindingdb/target_annotations_bindingdb.csv
- data/bindingdb/target_annotations_bindingdb.parquet
- data/bindingdb/target_annotations_lookup.pkl
"""

import argparse
import json
import os
import pickle
import re
import sys
import pandas as pd
from typing import Dict, List, Tuple, Any

# Ensure repository root is on sys.path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.target_clustering import (
    DISEASE_CATEGORIES,
    ORGAN_SYSTEMS,
    MOLECULAR_CLASSES,
    BINDINGDB_DATA_DIR,
    TARGET_TO_IDX_PATH,
)

DEFAULT_CACHE_PATH = os.path.join(BINDINGDB_DATA_DIR, "raw_annotations_cache.json")
DEFAULT_CSV_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_bindingdb.csv")
DEFAULT_PARQUET_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_bindingdb.parquet")
DEFAULT_LOOKUP_PATH = os.path.join(BINDINGDB_DATA_DIR, "target_annotations_lookup.pkl")

# Regex pattern definitions for Disease Categories
PATTERNS_ONCOLOGY = [
    r"\bcancer\b", r"\btumor\b", r"\bcarcinoma\b", r"\bsarcoma\b", r"\bglioma\b",
    r"\bleukemia\b", r"\blymphoma\b", r"\bmelanoma\b", r"\bmyeloma\b", r"\bblastoma\b",
    r"\bmalignan", r"\bneoplasm\b", r"\bneoplasia\b", r"\boncogen", r"\bwilms\b",
    r"\bneuroblastoma\b", r"\bglioblastoma\b", r"\bteratoma\b"
]
PATTERNS_NEUROLOGY = [
    r"\balzheimer\b", r"\bparkinson\b", r"\bepilep", r"\bneuropath", r"\bataxia\b",
    r"\bencephalopath", r"\bchorea\b", r"\bschizophrenia\b", r"\bseizure\b",
    r"\bneurodegenerat", r"\bamyotrophic\b", r"\bintellectual disability\b", r"\bdementia\b",
    r"\bautis", r"\bhuntington\b", r"\bdystonia\b", r"\bspastic\b", r"\bmicrocephaly\b",
    r"\bcentral nervous system\b", r"\bmigraine\b", r"\bdepression\b", r"\bbipolar\b"
]
PATTERNS_CARDIOVASCULAR = [
    r"\bcardiomyopath", r"\bhypertension\b", r"\barrhythm", r"\blong qt\b",
    r"\batherosclero", r"\bheart failure\b", r"\baneurysm\b", r"\bthrombos",
    r"\bmyocardial\b", r"\bcoronary\b", r"\bbrugada\b", r"\bischemi",
    r"\bcardiac\b", r"\barterial\b", r"\bvascul", r"\bstroke\b", r"\bangina\b"
]
PATTERNS_IMMUNOLOGY = [
    r"\bimmunodeficienc", r"\binflammat", r"\blupus\b", r"\brheumatoid\b",
    r"\bautoimmun", r"\barthrit", r"\bautoinflammat", r"\ballerg",
    r"\bcolitis\b", r"\bcrohn\b", r"\bpsoriasis\b", r"\bbehcet\b",
    r"\bgraft\b", r"\bimmune\b", r"\bsclerosis\b", r"\bmyasthenia\b", r"\bvasculitis\b"
]
PATTERNS_INFECTIOUS = [
    r"\binfecti", r"\bbacteri", r"\bviral\b", r"\bfungal\b", r"\bhiv\b",
    r"\bhepatitis\b", r"\btuberculosis\b", r"\bmalaria\b", r"\bpathogen",
    r"\bsepsis\b", r"\binfluenza\b", r"\bherpes\b", r"\bvirus\b", r"\bcovid\b"
]
PATTERNS_METABOLIC = [
    r"\bdiabet", r"\bobesit", r"\blipodystroph", r"\bhypercholesterol",
    r"\bhypoglycem", r"\bmetabolic\b", r"\badrenal\b", r"\bthyroid\b",
    r"\bcushing\b", r"\bglycogen\b", r"\bperoxisom", r"\bgaucher\b",
    r"\bniemann\b", r"\blipid\b", r"\baciduria\b", r"\bhyperinsulin", r"\bgout\b"
]
PATTERNS_RESPIRATORY = [
    r"\bpulmonary\b", r"\brespiratory\b", r"\bcystic fibrosis\b",
    r"\bciliary dyskinesia\b", r"\bbronch", r"\bemphysema\b", r"\basthma\b"
]
PATTERNS_DERM_SENSORY = [
    r"\bdeaf", r"\bblind", r"\bretin", r"\bhearing loss\b", r"\bichthyos",
    r"\bepidermoly", r"\balopecia\b", r"\bcataract\b", r"\bglaucoma\b",
    r"\busher syndrome\b", r"\bmacular\b", r"\bskin fragility\b", r"\bkeratod", r"\bucous\b"
]

# Regex pattern definitions for Organ Systems
PATTERNS_NERVOUS_SYSTEM = [
    r"\bbrain\b", r"\bcortex\b", r"\bcerebell", r"\bneuron", r"\bsynap",
    r"\bcns\b", r"\bspinal cord\b", r"\bhippocamp", r"\bnervous\b",
    r"\bforebrain\b", r"\bneuro", r"\bgangli"
]
PATTERNS_CARDIO_SYSTEM = [
    r"\bheart\b", r"\bmyocardi", r"\bvascular\b", r"\bendothel",
    r"\bblood vessel\b", r"\bartery\b", r"\barterial\b", r"\baorta\b",
    r"\batri", r"\bventricl"
]
PATTERNS_RESP_SYSTEM = [
    r"\blung\b", r"\bairway\b", r"\bbronch", r"\btrachea\b", r"\bpulmonar", r"\balveol"
]
PATTERNS_DIGESTIVE_SYSTEM = [
    r"\bliver\b", r"\bhepatic\b", r"\bintestin", r"\bcolon\b", r"\bstomach\b",
    r"\bgastric\b", r"\bpancreas\b", r"\bpancreatic\b", r"\bgut\b", r"\bbile\b",
    r"\bduoden", r"\bjejun", r"\bileum\b"
]
PATTERNS_ENDOCRINE_SYSTEM = [
    r"\bthyroid\b", r"\badrenal\b", r"\bpituitary\b", r"\bislet\b",
    r"\badipose\b", r"\bfat\b", r"\bovary\b", r"\btestis\b", r"\btestes\b",
    r"\bendocrine\b", r"\bplacenta\b"
]
PATTERNS_IMMUNE_SYSTEM = [
    r"\bspleen\b", r"\bthymus\b", r"\bbone marrow\b", r"\bleukocyte",
    r"\blymphocyte", r"\bt-cell", r"\bb-cell", r"\bmacrophage", r"\bneutrophil",
    r"\bmonocyte", r"\bblood\b", r"\bplatelet", r"\bhematopoietic", r"\blymph\b"
]
PATTERNS_MUSCULO_SYSTEM = [
    r"\bskeletal muscle\b", r"\bbone\b", r"\bosteoblast", r"\bchondrocyte",
    r"\bcartilage\b", r"\btendon\b", r"\bmuscle\b"
]
PATTERNS_RENAL_SYSTEM = [
    r"\bkidney\b", r"\brenal\b", r"\bbladder\b", r"\burinary\b",
    r"\bprostate\b", r"\buterus\b", r"\bglomerul", r"\bnephron\b"
]
PATTERNS_INTEG_SYSTEM = [
    r"\bskin\b", r"\bepiderm", r"\bkeratinocyte", r"\bhair\b",
    r"\bdermis\b", r"\bretina\b", r"\beye\b", r"\bcornea\b", r"\bcochlea\b"
]
PATTERNS_UBIQUITOUS_SYSTEM = [
    r"\bubiquitous", r"\bwidely expressed\b", r"\ball tissues\b",
    r"\bbroadly expressed\b", r"\bmost tissues\b"
]

# Regex pattern definitions for Molecular Classes
PATTERNS_KINASE = [
    r"\bprotein kinase\b", r"\btyr protein kinase\b", r"\bser/thr protein kinase\b",
    r"\bkinase superfamily\b", r"\btyrosine-protein kinase\b",
    r"\bserine/threonine-protein kinase\b"
]
PATTERNS_GPCR = [
    r"g[-\s]?protein[- ]coupled receptor", r"\bgpcr\b", r"\bseven-transmembrane\b",
    r"\brhodopsin-like\b", r"\bfrizzled\b", r"\bsecretin receptor\b",
    r"\bmetabotropic glutamate\b"
]
PATTERNS_ION_CHANNEL = [
    r"\bion channel\b", r"\bvoltage-gated\b", r"\bpotassium channel\b",
    r"\bsodium channel\b", r"\bcalcium channel\b", r"\btransient receptor potential\b",
    r"\bcys-loop\b", r"\bligand-gated ion\b", r"\binward rectifier\b", r"\bchloride channel\b"
]
PATTERNS_NUCLEAR_RECEPTOR = [
    r"\bnuclear receptor\b", r"\bnuclear hormone receptor\b", r"\bsteroid receptor\b",
    r"\bretinoid receptor\b", r"\bperoxisome proliferator-activated receptor\b",
    r"\bestrogen receptor\b", r"\bandrogen receptor\b", r"\bglucocorticoid receptor\b"
]
PATTERNS_PROTEASE = [
    r"\bpeptidase\b", r"\bprotease\b", r"\bcaspase\b", r"\bcathepsin\b",
    r"\bkallikrein\b", r"\bcalpain\b", r"\bchymotrypsin\b", r"\btrypsin\b",
    r"\bmatrix metalloproteinase\b", r"\bmetallopeptidase\b", r"\bhydrolase\b",
    r"\besterase\b", r"\blipase\b"
]
PATTERNS_PHOSPHATASE = [
    r"\bprotein phosphatase\b", r"\btyrosine-protein phosphatase\b",
    r"\bserine/threonine phosphatase\b", r"\bdual specificity phosphatase\b",
    r"\bphosphatase\b"
]
PATTERNS_TRANSPORTER = [
    r"\bsolute carrier\b", r"\btransporter\b", r"\babc transporter\b",
    r"\batp-binding cassette\b", r"\bsodium/potassium-transporting\b",
    r"\bmonocarboxylate transporter\b", r"\bfacilitative glucose transporter\b"
]
PATTERNS_EPIGENETIC = [
    r"\bhistone\b", r"\bbromodomain\b", r"\bchromodomain\b", r"\bdemethylase\b",
    r"\bmethyltransferase\b", r"\bdeacetylase\b", r"\bacetyltransferase\b",
    r"\bsirtuin\b", r"\bchromatin\b", r"\bpolycomb\b"
]


def _match_any(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    text_l = text.lower()
    return any(re.search(pat, text_l) is not None for pat in patterns)


def classify_disease_category(
    disease_text: str,
    organism: str,
    lineage: List[str],
    keywords: List[str],
    go_processes: List[str],
) -> str:
    """Classify target into one of the 9 canonical Disease Categories."""
    # 1. Non-mammalian Pathogens are Infectious Disease targets
    if any(k in lineage for k in ["Viruses", "Bacteria", "Apicomplexa", "Kinetoplastea"]) or "Candida" in organism:
        return "Infectious Disease"

    # 2. Text evidence from UniProt DISEASE comments
    if disease_text:
        if _match_any(disease_text, PATTERNS_ONCOLOGY):
            return "Oncology"
        if _match_any(disease_text, PATTERNS_NEUROLOGY):
            return "Neurology & CNS"
        if _match_any(disease_text, PATTERNS_CARDIOVASCULAR):
            return "Cardiovascular"
        if _match_any(disease_text, PATTERNS_IMMUNOLOGY):
            return "Immunology & Inflammation"
        if _match_any(disease_text, PATTERNS_METABOLIC):
            return "Metabolic & Endocrine"
        if _match_any(disease_text, PATTERNS_RESPIRATORY):
            return "Respiratory & Pulmonary"
        if _match_any(disease_text, PATTERNS_DERM_SENSORY):
            return "Dermatological & Sensory"
        if _match_any(disease_text, PATTERNS_INFECTIOUS):
            return "Infectious Disease"

    # 3. Supplemental evidence from keywords
    kws_set = set(keywords)
    if "Proto-oncogene" in kws_set or "Oncogene" in kws_set:
        return "Oncology"
    if "Deafness" in kws_set or "Blindness" in kws_set:
        return "Dermatological & Sensory"

    # 4. Supplemental evidence from GO biological processes
    go_text = " ".join(go_processes).lower()
    if go_text:
        if "inflammatory response" in go_text or "immune response" in go_text:
            return "Immunology & Inflammation"
        if "synaptic transmission" in go_text or "learning or memory" in go_text:
            return "Neurology & CNS"
        if "blood coagulation" in go_text or "regulation of blood pressure" in go_text:
            return "Cardiovascular"
        if "glucose metabolic process" in go_text or "insulin receptor signaling" in go_text:
            return "Metabolic & Endocrine"
        if "defense response to bacterium" in go_text or "viral process" in go_text:
            return "Infectious Disease"

    return "Unassigned / Non-Pathological Probe"


def classify_organ_system(tissue_text: str, disease_category: str) -> str:
    """Classify target into one of the 10 canonical Organ Systems."""
    if not tissue_text:
        # Fallback heuristic based on disease category if no tissue comment
        if disease_category == "Neurology & CNS":
            return "Central & Peripheral Nervous System"
        if disease_category == "Cardiovascular":
            return "Cardiovascular System"
        if disease_category == "Respiratory & Pulmonary":
            return "Respiratory System"
        return "Ubiquitous / Systemic"

    # Check ubiquitous first
    if _match_any(tissue_text, PATTERNS_UBIQUITOUS_SYSTEM):
        return "Ubiquitous / Systemic"
    if _match_any(tissue_text, PATTERNS_NERVOUS_SYSTEM):
        return "Central & Peripheral Nervous System"
    if _match_any(tissue_text, PATTERNS_IMMUNE_SYSTEM):
        return "Immune & Hematopoietic System"
    if _match_any(tissue_text, PATTERNS_DIGESTIVE_SYSTEM):
        return "Digestive & Hepatobiliary System"
    if _match_any(tissue_text, PATTERNS_CARDIO_SYSTEM):
        return "Cardiovascular System"
    if _match_any(tissue_text, PATTERNS_RENAL_SYSTEM):
        return "Renal & Urogenital System"
    if _match_any(tissue_text, PATTERNS_ENDOCRINE_SYSTEM):
        return "Endocrine & Metabolic Organs"
    if _match_any(tissue_text, PATTERNS_MUSCULO_SYSTEM):
        return "Musculoskeletal System"
    if _match_any(tissue_text, PATTERNS_RESP_SYSTEM):
        return "Respiratory System"
    if _match_any(tissue_text, PATTERNS_INTEG_SYSTEM):
        return "Integumentary System"

    return "Ubiquitous / Systemic"


def classify_molecular_class(
    similarity_text: str,
    keywords: List[str],
    protein_name: str,
) -> Tuple[str, str]:
    """Classify target into one of the 9 canonical Molecular Classes, plus granular family."""
    combined = f"{similarity_text} {' '.join(keywords)} {protein_name}".lower()
    kws_set = set(keywords)

    # 1. Protein Kinase
    if _match_any(combined, PATTERNS_KINASE) or "Kinase" in kws_set:
        mclass = "Protein Kinase"
    # 2. GPCR
    elif _match_any(combined, PATTERNS_GPCR) or "G-protein coupled receptor" in kws_set or "G protein-coupled receptor" in kws_set:
        mclass = "G-Protein Coupled Receptor (GPCR)"
    # 3. Nuclear Receptor
    elif _match_any(combined, PATTERNS_NUCLEAR_RECEPTOR) or "Nuclear receptor" in kws_set:
        mclass = "Nuclear Receptor"
    # 4. Ion Channel
    elif _match_any(combined, PATTERNS_ION_CHANNEL) or "Ion channel" in kws_set or "Voltage-gated channel" in kws_set:
        mclass = "Ion Channel"
    # 5. Phosphatase (higher priority than general hydrolase)
    elif _match_any(combined, PATTERNS_PHOSPHATASE) or "Phosphatase" in kws_set:
        mclass = "Phosphatase"
    # 6. Epigenetic Regulator
    elif _match_any(combined, PATTERNS_EPIGENETIC):
        mclass = "Epigenetic Regulator"
    # 7. Protease / Hydrolase
    elif _match_any(combined, PATTERNS_PROTEASE) or "Protease" in kws_set or "Hydrolase" in kws_set:
        mclass = "Protease / Hydrolase"
    # 8. Transporter / Solute Carrier
    elif _match_any(combined, PATTERNS_TRANSPORTER) or "Transporter" in kws_set:
        mclass = "Transporter / Solute Carrier"
    # 9. Other Enzyme / Protein
    else:
        mclass = "Other Enzyme / Protein"

    # Extract granular family
    family = similarity_text if similarity_text else "Unclassified"
    return mclass, family


def process_all_targets(
    cache_path: str = DEFAULT_CACHE_PATH,
    target_to_idx_path: str = TARGET_TO_IDX_PATH,
) -> pd.DataFrame:
    """Ingest raw UniProt cache, classify every target, and return validated DataFrame."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}. Run fetch_annotations.py first.")
    if not os.path.exists(target_to_idx_path):
        raise FileNotFoundError(f"Target index file not found: {target_to_idx_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    with open(target_to_idx_path, "rb") as f:
        target_to_idx = pickle.load(f)

    rows = []
    # Guarantee exact sorting by target_idx: 0, 1, 2, ..., N
    sorted_targets = sorted(target_to_idx.items(), key=lambda x: x[1])

    for accession, idx in sorted_targets:
        rec = cache.get(accession, {})
        entry_type = rec.get("entryType", "UniProtKB unreviewed (TrEMBL)")
        is_reviewed = "reviewed" in entry_type.lower()
        annotation_source = "UniProtKB/Swiss-Prot" if is_reviewed else "UniProtKB/TrEMBL"

        # Protein Name
        p_desc = rec.get("proteinDescription", {})
        rec_name = p_desc.get("recommendedName", {}).get("fullName", {}).get("value")
        if not rec_name:
            sub_names = p_desc.get("submissionNames", [])
            if sub_names:
                rec_name = sub_names[0].get("fullName", {}).get("value")
        if not rec_name:
            rec_name = rec.get("uniProtkbId", accession)

        # Gene Name
        genes = rec.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""

        # Organism
        org_dict = rec.get("organism", {})
        organism = org_dict.get("scientificName", "Unknown")
        lineage = org_dict.get("lineage", [])

        # Keywords
        keywords = [kw.get("name", "") for kw in rec.get("keywords", [])]

        # Comments extraction
        comments = rec.get("comments", [])
        disease_entries = []
        disease_texts = []
        disease_pubmeds = []
        tissue_texts = []
        tissue_pubmeds = []
        similarity_texts = []
        has_experimental_evidence = False

        for c in comments:
            c_type = c.get("commentType")
            if c_type == "DISEASE":
                d = c.get("disease")
                if d:
                    d_name = d.get("diseaseId", "")
                    d_acc = d.get("diseaseAccession", "")
                    d_desc = d.get("description", "")
                    entry_str = f"{d_name} [{d_acc}]" if d_acc else d_name
                    disease_entries.append(entry_str)
                    disease_texts.append(f"{d_name} {d_desc}")
                    for ev in d.get("evidences", []):
                        if ev.get("source") == "PubMed" and ev.get("id"):
                            disease_pubmeds.append(str(ev.get("id")))
                            has_experimental_evidence = True
            elif c_type == "TISSUE SPECIFICITY":
                for t in c.get("texts", []):
                    tissue_texts.append(t.get("value", ""))
                    for ev in t.get("evidences", []):
                        if ev.get("source") == "PubMed" and ev.get("id"):
                            tissue_pubmeds.append(str(ev.get("id")))
                            has_experimental_evidence = True
            elif c_type == "SIMILARITY":
                for t in c.get("texts", []):
                    similarity_texts.append(t.get("value", ""))

        # GO biological processes
        go_processes = []
        for xref in rec.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                for prop in xref.get("properties", []):
                    val = prop.get("value", "")
                    if val.startswith("P:"):
                        go_processes.append(val[2:])

        # Classifications
        disease_str = " ".join(disease_texts)
        disease_category = classify_disease_category(
            disease_text=disease_str,
            organism=organism,
            lineage=lineage,
            keywords=keywords,
            go_processes=go_processes,
        )

        tissue_str = " ".join(tissue_texts)
        organ_system = classify_organ_system(
            tissue_text=tissue_str,
            disease_category=disease_category,
        )

        sim_str = " ".join(similarity_texts)
        molecular_class, protein_family = classify_molecular_class(
            similarity_text=sim_str,
            keywords=keywords,
            protein_name=rec_name,
        )

        # Evidence level
        if is_reviewed:
            evidence_level = "Curated (Literature Supported)" if has_experimental_evidence else "Curated (Sequence Rule)"
        else:
            evidence_level = "Unreviewed / Computational"

        # Specific diseases string
        specific_diseases_str = "; ".join(disease_entries) if disease_entries else "None reported"

        # Unique PubMed IDs
        unique_pubmeds = list(dict.fromkeys(disease_pubmeds + tissue_pubmeds))
        pubmed_str = "; ".join(unique_pubmeds[:10]) if unique_pubmeds else "None"

        rows.append({
            "target_idx": int(idx),
            "uniprot_id": accession,
            "protein_name": rec_name,
            "gene_name": gene_name,
            "organism": organism,
            "disease_category": disease_category,
            "specific_diseases": specific_diseases_str,
            "organ_system": organ_system,
            "molecular_class": molecular_class,
            "protein_family": protein_family,
            "annotation_source": annotation_source,
            "evidence_level": evidence_level,
            "pubmed_ids": pubmed_str,
        })

    df = pd.DataFrame(rows)
    # Ensure index column order and types
    df["target_idx"] = df["target_idx"].astype(int)
    return df


def save_target_annotations(
    df: pd.DataFrame,
    csv_path: str = DEFAULT_CSV_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
    lookup_path: str = DEFAULT_LOOKUP_PATH,
):
    """Save processed dataframe into CSV, Parquet, and dictionary lookup pkl."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 1. CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved CSV: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")

    # 2. Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")

    # 3. Lookup Dictionary (target_idx -> record dict)
    lookup_dict = {row["target_idx"]: row for row in df.to_dict(orient="records")}
    with open(lookup_path, "wb") as f:
        pickle.dump(lookup_dict, f)
    print(f"  Saved Lookup: {lookup_path} ({os.path.getsize(lookup_path) / 1024:.1f} KB)")


def print_summary_report(df: pd.DataFrame):
    """Print structured biological summary distribution."""
    print("\n" + "=" * 70)
    print(f"TARGET CLUSTERING & ANNOTATION REPORT ({len(df):,d} Targets)")
    print("=" * 70)

    print("\n[1] DISEASE CATEGORY DISTRIBUTION:")
    d_counts = df["disease_category"].value_counts()
    for cat in DISEASE_CATEGORIES:
        cnt = d_counts.get(cat, 0)
        pct = (cnt / len(df)) * 100
        print(f"  - {cat:<36} {cnt:>5d} ({pct:>5.1f}%)")

    print("\n[2] ORGAN SYSTEM DISTRIBUTION:")
    o_counts = df["organ_system"].value_counts()
    for org in ORGAN_SYSTEMS:
        cnt = o_counts.get(org, 0)
        pct = (cnt / len(df)) * 100
        print(f"  - {org:<36} {cnt:>5d} ({pct:>5.1f}%)")

    print("\n[3] MOLECULAR CLASS DISTRIBUTION:")
    m_counts = df["molecular_class"].value_counts()
    for mol in MOLECULAR_CLASSES:
        cnt = m_counts.get(mol, 0)
        pct = (cnt / len(df)) * 100
        print(f"  - {mol:<36} {cnt:>5d} ({pct:>5.1f}%)")

    print("\n[4] ANNOTATION SOURCE & EVIDENCE QUALITY:")
    for src, cnt in df["annotation_source"].value_counts().items():
        print(f"  - {src:<36} {cnt:>5d} ({(cnt/len(df))*100:>5.1f}%)")
    for ev, cnt in df["evidence_level"].value_counts().items():
        print(f"  - {ev:<36} {cnt:>5d} ({(cnt/len(df))*100:>5.1f}%)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Path to raw UniProt JSON cache")
    parser.add_argument("--target-to-idx", default=TARGET_TO_IDX_PATH, help="Path to target_to_idx.pkl")
    parser.add_argument("--csv-out", default=DEFAULT_CSV_PATH, help="Output CSV path")
    parser.add_argument("--parquet-out", default=DEFAULT_PARQUET_PATH, help="Output Parquet path")
    parser.add_argument("--lookup-out", default=DEFAULT_LOOKUP_PATH, help="Output lookup pkl path")
    args = parser.parse_args()

    print("Processing and classifying drug targets from UniProt cache...")
    df = process_all_targets(cache_path=args.cache_path, target_to_idx_path=args.target_to_idx)
    save_target_annotations(df, csv_path=args.csv_out, parquet_path=args.parquet_out, lookup_path=args.lookup_out)
    print_summary_report(df)


if __name__ == "__main__":
    main()
