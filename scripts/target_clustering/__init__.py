"""Target Clustering and Biological Annotation Module for BindingDB DTI Prediction.

This module provides reproducible pipelines to:
1. Fetch authoritative target annotations from UniProtKB with local disk caching.
2. Standardize targets into controlled Disease Categories, Organ Systems, and Molecular Classes.
3. Profile target distributions across biological domains in BindingDB.
4. Perform category-stratified cold-target evaluation to analyze model generalization across biological families.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
BINDINGDB_DATA_DIR = os.path.join(DATA_DIR, "bindingdb")
BINDINGDB_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "bindingdb")

TARGET_TO_IDX_PATH = os.path.join(DATA_DIR, "target_to_idx.pkl")
TARGET_SEQUENCES_PATH = os.path.join(DATA_DIR, "target_sequences.pkl")

# Controlled Vocabularies
DISEASE_CATEGORIES = (
    "Oncology",
    "Neurology & CNS",
    "Cardiovascular",
    "Immunology & Inflammation",
    "Infectious Disease",
    "Metabolic & Endocrine",
    "Respiratory & Pulmonary",
    "Dermatological & Sensory",
    "Unassigned / Non-Pathological Probe",
)

ORGAN_SYSTEMS = (
    "Central & Peripheral Nervous System",
    "Cardiovascular System",
    "Respiratory System",
    "Digestive & Hepatobiliary System",
    "Endocrine & Metabolic Organs",
    "Immune & Hematopoietic System",
    "Musculoskeletal System",
    "Renal & Urogenital System",
    "Integumentary System",
    "Ubiquitous / Systemic",
)

MOLECULAR_CLASSES = (
    "Protein Kinase",
    "G-Protein Coupled Receptor (GPCR)",
    "Ion Channel",
    "Nuclear Receptor",
    "Protease / Hydrolase",
    "Phosphatase",
    "Transporter / Solute Carrier",
    "Epigenetic Regulator",
    "Other Enzyme / Protein",
)

__all__ = [
    "DISEASE_CATEGORIES",
    "ORGAN_SYSTEMS",
    "MOLECULAR_CLASSES",
    "BINDINGDB_DATA_DIR",
    "BINDINGDB_RESULTS_DIR",
    "TARGET_TO_IDX_PATH",
    "TARGET_SEQUENCES_PATH",
]
