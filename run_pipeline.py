"""
run_pipeline.py
===============
Entry point for the Crop Disease Entity Resolution pipeline.
Runs all five steps end-to-end and produces the final ML training dataset.

Usage:
    python run_pipeline.py

Prerequisites:
    1. Activate the virtual environment:
           agrienv\\Scripts\\activate    (Windows)
           source agrienv/bin/activate  (Linux/macOS)
    2. Place extracted_kg_triples.csv in:
           data/input/extracted_kg_triples.csv

Output:
    data/processed/crop_diseases_clean.csv  -- curated disease dataset
    data/pairs/final_dataset.csv            -- ML training pairs
    data/pairs/positive_pairs.csv           -- positive pairs only
    data/pairs/negative_pairs.csv           -- negative pairs only
"""

import sys
import shutil
from pathlib import Path
import pandas as pd

# Add src/ to import path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

INPUT_DIR = Path(__file__).parent / "data" / "input"
RAW_DIR   = Path(__file__).parent / "data" / "raw"

print()
print("=" * 60)
print("  Crop Disease Entity Resolution Dataset Builder")
print("  Running Full Pipeline")
print("=" * 60)


# -----------------------------------------------------------------------------
# STEP 0: Verify input file
# -----------------------------------------------------------------------------

input_kg = INPUT_DIR / "extracted_kg_triples.csv"
raw_kg   = RAW_DIR / "kg_triples_raw.csv"

if not input_kg.exists():
    print(f"\n  [ERROR] Input file not found: {input_kg}")
    print("  Please place extracted_kg_triples.csv in:  data/input/")
    sys.exit(1)

print(f"\n  [OK] Input file found: {input_kg}")


# -----------------------------------------------------------------------------
# STEP 1: KG Triples Extraction
# -----------------------------------------------------------------------------

print("\n")
print("-" * 60)
print("  STEP 1 of 5 -- KG Triples Extraction")
print("-" * 60)

from importlib import import_module
kg_module = import_module("04_extract_kg_triples")
kg_entities, kg_pairs = kg_module.main()


# -----------------------------------------------------------------------------
# STEP 2: PlantVillage Disease Extraction
# -----------------------------------------------------------------------------

print("\n")
print("-" * 60)
print("  STEP 2 of 5 -- PlantVillage Disease Extraction")
print("-" * 60)

pv_module = import_module("01_scrape_plantvillage")
pv_entities, pv_pairs = pv_module.main()


# -----------------------------------------------------------------------------
# STEP 3: AGROVOC Disease Terms
# -----------------------------------------------------------------------------

print("\n")
print("-" * 60)
print("  STEP 3 of 5 -- AGROVOC Agricultural Vocabulary")
print("-" * 60)

av_module = import_module("02_scrape_agrovoc")
av_entities, av_pairs = av_module.main()


# -----------------------------------------------------------------------------
# STEP 4: Wikipedia Disease Context
# -----------------------------------------------------------------------------

print("\n")
print("-" * 60)
print("  STEP 4 of 5 -- Wikipedia Disease Context")
print("-" * 60)

wp_module = import_module("03_scrape_wikipedia")
wp_entities, wp_pairs = wp_module.main()


# -----------------------------------------------------------------------------
# STEP 5: Build Final Training Dataset
# -----------------------------------------------------------------------------

print("\n")
print("-" * 60)
print("  STEP 5 of 5 -- Building Final Training Dataset")
print("-" * 60)

build_module = import_module("05_build_pairs")
final_df, entities_df = build_module.main()


# -----------------------------------------------------------------------------
# FINAL REPORT
# -----------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print()

if final_df is not None:
    n_pos = (final_df["label"] == 1).sum()
    n_neg = (final_df["label"] == 0).sum()

    print("  Dataset Statistics:")
    print(f"    Entity records  : {len(entities_df):,}")
    print(f"    Training pairs  : {len(final_df):,}")
    print(f"      Positive (match)    : {n_pos:,}")
    print(f"      Negative (no match) : {n_neg:,}")
    print(f"      Pos/Neg ratio       : {n_pos / max(n_neg, 1):.2f}")

    if "pair_source" in final_df.columns:
        print()
        print("  Pair source breakdown:")
        for src, cnt in final_df["pair_source"].value_counts().items():
            print(f"    {str(src):<42} {cnt:>5}")

print()
print("  Files saved:")
print("    data/processed/crop_diseases_clean.csv   -- curated disease dataset")
print("    data/pairs/final_dataset.csv             -- ML training data")
print("    data/pairs/positive_pairs.csv            -- positives only")
print("    data/pairs/negative_pairs.csv            -- negatives only")
print()
print("  Load in your model:")
print("    import pandas as pd")
print("    df = pd.read_csv('data/pairs/final_dataset.csv')")
print("    X  = df[['name_1', 'name_2', 'context_1', 'context_2']]")
print("    y  = df['label']   # 1 = match, 0 = non-match")
print()
