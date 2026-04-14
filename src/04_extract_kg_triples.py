"""
04_extract_kg_triples.py — Extract entity resolution pairs from extracted_kg_triples.csv
AgriΛNet Entity Resolution Pipeline

Your file (extracted_kg_triples.csv) contains knowledge graph triples from a
blockchain-in-agriculture research paper. A triple looks like:

    head_entity          relation     tail_entity
    "blockchain tech"    Used_For     "supply chain"

Your file has 400 rows. Most are about blockchain, people, locations, dates.
But buried in there are TWO disease entries:
    • "late blight"   Caused_By   "Phytophthora infestans"
    • "Early blight"  Caused_By   "Alternaria solani"

More importantly, "blockchain technology" appears under MANY different names:
    "blockchain technology", "Blockchain Technology", "BCT", "block chain", ...

For entity resolution, two records that refer to the SAME real-world entity
but have DIFFERENT names are called a POSITIVE PAIR.

This script:
  1. Extracts all blockchain surface forms → builds positive pairs
  2. Extracts the 2 disease seeds
  3. Extracts organism entities (Phytophthora infestans, Alternaria solani)
  4. Saves entities to data/raw/kg_triples_raw.csv
  5. Saves positive pairs to data/pairs/kg_pairs_positive.csv

HOW TO RUN:
    python src/04_extract_kg_triples.py

WHERE TO PUT YOUR FILE:
    Copy extracted_kg_triples.csv into:  data/raw/extracted_kg_triples.csv
    OR update INPUT_FILE path below.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import pandas as pd
import itertools
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_entity_record, save_raw, normalise_name,
    clean_context, RAW_DIR, PAIRS_DIR
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Change INPUT_FILE if your CSV is somewhere else
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = RAW_DIR.parent / "input" / "extracted_kg_triples.csv"
# If not found in data/raw/, we try the uploads folder (for running in claude.ai)
FALLBACK_FILE = Path("/mnt/user-data/uploads/extracted_kg_triples.csv")

SOURCE_NAME = "kg_triples"
SOURCE_URL  = "local:extracted_kg_triples.csv"  # No public URL for this file


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Define which entity surface forms are "the same entity"
#
# This is a manually curated mapping based on looking at the actual data.
# Key   = canonical (the "true" standard name we choose)
# Value = list of all surface forms found in the file that mean the same thing
# ─────────────────────────────────────────────────────────────────────────────

BLOCKCHAIN_VARIANTS = {
    "blockchain technology": [
        # Exact case variants
        "blockchain technology",
        "Blockchain Technology",
        "BLOCKCHAIN TECHNOLOGY",
        "Blockchain technology",
        # Abbreviations
        "BCT",
        "bct",
        # Spacing variants
        "block chain technology",
        "block chain",
        # Singular/plural
        "blockchain technologies",
        "Blockchain",
        "blockchain",
        # Compound forms
        "blockchain implementation",
        "blockchain adoption",
        "blockchain platform",
        "blockchain applications",
        "Blockchain infrastructure",
        "blockchain enabled traceability",
    ],
    "bct system": [
        "BCT-based system",
        "BCT-based monitoring systems",
        "BCT assisted real-time information",
        "BCT-assisted livestock monitoring",
    ],
}

# Disease seeds from the file (only 2 exist — we confirm from the data)
DISEASE_SEEDS = {
    "late blight": {
        "name": "late blight",
        "entity_type": "Disease",
        "context": "A destructive disease of potato and tomato plants caused by the oomycete pathogen Phytophthora infestans. Leads to rapid tissue death and crop failure.",
        "organism": "Phytophthora infestans",
    },
    "early blight": {
        "name": "Early blight",
        "entity_type": "Disease",
        "context": "A fungal disease of tomato and potato caused by Alternaria solani. Characterized by dark concentric spots on leaves.",
        "organism": "Alternaria solani",
    },
}


def load_file() -> pd.DataFrame:
    """Load the KG triples CSV from either expected location."""
    if INPUT_FILE.exists():
        print(f"  Loading from: {INPUT_FILE}")
        return pd.read_csv(INPUT_FILE)
    elif FALLBACK_FILE.exists():
        print(f"  Loading from: {FALLBACK_FILE}")
        return pd.read_csv(FALLBACK_FILE)
    else:
        raise FileNotFoundError(
            f"Cannot find extracted_kg_triples.csv\n"
            f"Please place it at: {INPUT_FILE}"
        )


def extract_entities_from_file(df: pd.DataFrame) -> list:
    """
    WHAT THIS DOES:
      Scans all rows in the KG triples file and extracts meaningful entities.
      We care about: Disease, Organism, Technology (blockchain variants).

    RETURNS:
      A list of entity record dicts (one per row in output CSV).
    """
    records = []

    # ── Collect all unique entities from both head and tail columns ──────────
    # The file has head_entity and tail_entity. Both can be entities we want.
    all_entity_pairs = list(zip(
        df["head_entity"].fillna(""),
        df["head_entity_type"].fillna(""),
    )) + list(zip(
        df["tail_entity"].fillna(""),
        df["tail_entity_type"].fillna(""),
    ))

    seen_canonicals = set()

    for entity_name, entity_type in all_entity_pairs:
        entity_name = str(entity_name).strip()
        entity_type = str(entity_type).strip()

        if not entity_name or entity_name == "nan":
            continue

        # Only keep entity types we care about for AgriΛNet
        keep_types = {"Disease", "Organism", "Technology", "Crop",
                      "Agri_Method", "Agri_Process"}
        if entity_type not in keep_types:
            continue

        canonical = normalise_name(entity_name)
        if canonical in seen_canonicals:
            continue
        seen_canonicals.add(canonical)

        # Generate context from the disease seeds if we have it
        context = ""
        if canonical in DISEASE_SEEDS:
            context = DISEASE_SEEDS[canonical]["context"]

        record = build_entity_record(
            name=entity_name,
            context=context,
            source=SOURCE_NAME,
            source_url=SOURCE_URL,
            entity_type=entity_type if entity_type else None,
        )
        records.append(record)

    return records


def build_blockchain_positive_pairs() -> list:
    """
    WHAT THIS DOES:
      Takes the BLOCKCHAIN_VARIANTS mapping and generates all pairs of names
      that refer to the same entity. These are POSITIVE PAIRS for training.

    EXAMPLE:
      canonical = "blockchain technology"
      variants  = ["BCT", "block chain", "Blockchain Technology", ...]

      We generate ALL combinations:
        ("BCT", "block chain")         → label=1 (same entity)
        ("BCT", "Blockchain Technology") → label=1
        ("block chain", "Blockchain Technology") → label=1
        ... and so on

    WHY COMBINATIONS AND NOT JUST PAIRS WITH CANONICAL?
      Your lambda estimator should learn that ANY two variants match,
      not just "variant matches canonical". So we generate all C(n,2) pairs.
    """
    pairs = []
    pair_id = 1

    for canonical, variants in BLOCKCHAIN_VARIANTS.items():
        # Include canonical itself in the list
        all_forms = [canonical] + variants

        # itertools.combinations gives all unique pairs without repetition
        # e.g., combinations(["A","B","C"], 2) → (A,B), (A,C), (B,C)
        for name_1, name_2 in itertools.combinations(all_forms, 2):
            pairs.append({
                "pair_id":     f"KG_POS_{pair_id:04d}",
                "name_1":      name_1,
                "name_2":      name_2,
                "canonical_1": normalise_name(name_1),
                "canonical_2": normalise_name(name_2),
                "entity_type": "Technology",
                "label":       1,            # 1 = same entity (POSITIVE)
                "pair_source": "kg_triples_blockchain_variants",
                "confidence":  1.0,          # We are 100% sure (manually verified)
                "note":        f"Both refer to canonical: '{canonical}'",
            })
            pair_id += 1

    return pairs


def build_disease_seed_pairs() -> list:
    """
    WHAT THIS DOES:
      The 2 diseases in the file give us disease + organism pairs.
      We also create name-variant pairs for them.
    """
    pairs = []
    pair_id = 1

    disease_name_variants = {
        "late blight": [
            "Late Blight",
            "LATE BLIGHT",
            "late-blight",
            "LB",
            "late blight disease",
            "potato late blight",
            "tomato late blight",
        ],
        "early blight": [
            "Early Blight",
            "EARLY BLIGHT",
            "early-blight",
            "EB",
            "early blight disease",
            "Alternaria blight",
        ],
    }

    import itertools
    for canonical, variants in disease_name_variants.items():
        all_forms = [canonical] + variants
        for name_1, name_2 in itertools.combinations(all_forms, 2):
            pairs.append({
                "pair_id":     f"KG_DIS_{pair_id:04d}",
                "name_1":      name_1,
                "name_2":      name_2,
                "canonical_1": normalise_name(name_1),
                "canonical_2": normalise_name(name_2),
                "entity_type": "Disease",
                "label":       1,
                "pair_source": "kg_triples_disease_seeds",
                "confidence":  1.0,
                "note":        f"Both refer to canonical: '{canonical}'",
            })
            pair_id += 1

    return pairs


def main():
    print("\n" + "═"*60)
    print("  SCRIPT 04 — KG Triples Extractor")
    print("  AgriΛNet Entity Resolution Pipeline")
    print("═"*60)

    # ── Load the file ──────────────────────────────────────────────────────
    print("\n[1/4] Loading KG triples file...")
    df = load_file()
    print(f"  Loaded {len(df)} rows, columns: {df.columns.tolist()}")

    # ── Extract entities ───────────────────────────────────────────────────
    print("\n[2/4] Extracting entity records...")
    entity_records = extract_entities_from_file(df)
    entities_df = save_raw(entity_records, "kg_triples_raw.csv")

    print(f"\n  Entity type breakdown:")
    for etype, count in entities_df["entity_type"].value_counts().items():
        print(f"    {etype:20s} → {count} entities")

    # ── Build positive pairs ───────────────────────────────────────────────
    print("\n[3/4] Building positive training pairs...")

    blockchain_pairs = build_blockchain_positive_pairs()
    print(f"  Blockchain variant pairs: {len(blockchain_pairs)}")

    disease_pairs = build_disease_seed_pairs()
    print(f"  Disease seed pairs:       {len(disease_pairs)}")

    all_pairs = blockchain_pairs + disease_pairs
    pairs_df = pd.DataFrame(all_pairs)
    out_path = PAIRS_DIR / "kg_pairs_positive.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(pairs_df)} positive pairs → {out_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[4/4] Summary")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │ Entities extracted:  {len(entities_df):4d}            │")
    print(f"  │ Positive pairs:      {len(pairs_df):4d}            │")
    print(f"  │   - Blockchain:      {len(blockchain_pairs):4d}            │")
    print(f"  │   - Disease seeds:   {len(disease_pairs):4d}            │")
    print(f"  └─────────────────────────────────────┘")

    print("\n  ✅ Script 04 complete! Next: run 01_scrape_plantvillage.py")
    print("     (or run 05_build_pairs.py if all scrapers are done)\n")

    return entities_df, pairs_df


if __name__ == "__main__":
    main()