"""
05_build_pairs.py — Merge all sources and build the final ML training dataset
AgriΛNet Entity Resolution Pipeline

═══════════════════════════════════════════════════════════════════════════════
BEGINNER EXPLANATION — What is this script doing?
═══════════════════════════════════════════════════════════════════════════════

This is the MOST IMPORTANT script. It takes all the data from steps 01-04
and builds the final dataset your lambda estimator will train on.

WHAT YOUR MODEL NEEDS:
  Your λ-estimator needs to learn: "are these two entity names the same?"
  For that, it needs EXAMPLES of:
    - Pairs that ARE the same (label=1) → POSITIVE PAIRS
    - Pairs that ARE NOT the same (label=0) → NEGATIVE PAIRS

  Right now from scripts 01-04, you have ONLY positive pairs.
  This script also generates hard negative pairs.

WHAT IS A "HARD NEGATIVE"?
  An easy negative: ("Late Blight", "Blockchain Technology") — obviously different
  A hard negative:  ("Late Blight", "Early Blight") — similar names, DIFFERENT diseases

  Hard negatives are more valuable for training because they force the model
  to learn fine-grained differences, not just surface similarity.

HOW WE BUILD NEGATIVES:
  Method 1: Same crop, different disease (hardest negatives)
    ("tomato late blight", "tomato early blight") → label=0
  Method 2: Same disease type word, different pathogen
    ("wheat leaf rust", "wheat stem rust") → label=0
  Method 3: Random pairs from different canonical entities
    ("late blight", "crown gall") → label=0

FINAL OUTPUT COLUMNS (what your model sees):
  name_1, name_2         — the two surface forms to compare
  context_1, context_2   — their descriptions (for semantic features)
  entity_type            — Disease / Technology / etc.
  label                  — 1 (same) or 0 (different)
  confidence             — how sure we are about the label
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import random
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_all_raw, deduplicate_entities, normalise_name,
    patch_missing_context, PROCESSED_DIR, PAIRS_DIR, RAW_DIR
)

random.seed(42)   # Fix random seed → reproducible results every time


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER TO YOUR QUESTION:
# "What if context is missing or wrong? Can I manually add fields?"
#
# YES! Add your manual patches here.
# Key = canonical name (lowercase). Value = dict of fields to fix.
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_PATCHES = {
    # Example: If "late blight" has empty context, this fills it in
    "late blight": {
        "context": "Devastating oomycete disease of potato and tomato caused by Phytophthora infestans. Causes rapid water-soaked lesions turning dark brown-black. Caused the Irish Potato Famine.",
        "entity_type": "Disease",
    },
    "early blight": {
        "context": "Fungal disease of tomato and potato caused by Alternaria solani. Produces concentric ring lesions giving a target-board appearance on lower leaves first.",
        "entity_type": "Disease",
    },
    "bct": {
        "context": "Abbreviation for Blockchain Technology — a distributed ledger system used in agriculture for supply chain traceability and food safety.",
        "entity_type": "Technology",
    },
    # Add more patches here as you discover missing/wrong entries:
    # "your disease name": {
    #     "context": "Your description here.",
    #     "entity_type": "Disease",
    # },
}


def load_and_merge_entities() -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Loads all raw CSVs from data/raw/, stacks them, deduplicates,
      applies manual patches, and saves the clean master entity list.
    """
    print("  Loading all raw entity files...")
    df = load_all_raw()

    # Ensure required columns exist (some scrapers might have extra columns)
    required_cols = ["entity_id", "name", "canonical", "entity_type",
                     "context", "source", "source_url"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[required_cols].copy()

    print(f"\n  Before deduplication: {len(df)} rows")
    df = deduplicate_entities(df)
    print(f"  After deduplication:  {len(df)} rows")

    # Apply manual patches for missing/wrong context
    print(f"\n  Applying {len(MANUAL_PATCHES)} manual patches...")
    df = patch_missing_context(df, MANUAL_PATCHES)

    # Save the master entity list
    out_path = PROCESSED_DIR / "all_entities.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Master entity list → {out_path}")

    return df


def load_all_positive_pairs() -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Loads all positive pair CSVs from data/pairs/ and merges them.
      All these have label=1 (same entity).
    """
    pair_files = list(PAIRS_DIR.glob("*_positive.csv"))
    if not pair_files:
        print("  ⚠ No positive pair files found. Run scripts 01-04 first.")
        return pd.DataFrame()

    frames = []
    for f in pair_files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  Loaded {len(df):4d} pairs from {f.name}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  ─── Total positive pairs: {len(combined)}")
    return combined


def enrich_pairs_with_context(pairs_df: pd.DataFrame,
                               entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Each pair currently has name_1, name_2 but no context.
      We add context by looking up each name in the entities table.

    WHY CONTEXT MATTERS:
      Your lambda estimator will use BOTH the name similarity AND the
      semantic similarity of contexts to make decisions.
      Without context, it can only use string/text features.
    """
    # Build a lookup: canonical name → context
    # We try to find the richest context for each canonical
    context_lookup = {}
    for _, row in entities_df.iterrows():
        canon = str(row.get("canonical", "")).strip()
        ctx   = str(row.get("context", "")).strip()
        if canon and ctx:
            # Keep the longest (most informative) context
            if canon not in context_lookup or len(ctx) > len(context_lookup[canon]):
                context_lookup[canon] = ctx

    def get_ctx(name: str) -> str:
        canon = normalise_name(str(name))
        return context_lookup.get(canon, "")

    pairs_df = pairs_df.copy()
    pairs_df["context_1"] = pairs_df["name_1"].apply(get_ctx)
    pairs_df["context_2"] = pairs_df["name_2"].apply(get_ctx)

    # Report how many pairs have context
    has_ctx = (pairs_df["context_1"].str.len() > 0) | (pairs_df["context_2"].str.len() > 0)
    print(f"  Pairs with at least one context: {has_ctx.sum()} / {len(pairs_df)}")

    return pairs_df


def generate_negative_pairs(entities_df: pd.DataFrame,
                             n_negatives: int = None) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Creates NEGATIVE PAIRS (label=0) — pairs of entities that look similar
      but are actually DIFFERENT diseases.

    THREE STRATEGIES for generating hard negatives:

    Strategy 1 — Same-crop negatives (hardest)
      "tomato late blight" vs "tomato early blight"
      Both have "tomato" in name, but different diseases.

    Strategy 2 — Same-type negatives
      "leaf rust" vs "stem rust" vs "stripe rust"
      All are rusts, but different pathogens.

    Strategy 3 — Random cross-entity negatives
      Any two different canonical entities.
    """
    disease_entities = entities_df[
        entities_df["entity_type"] == "Disease"
    ].drop_duplicates(subset=["canonical"]).copy()

    tech_entities = entities_df[
        entities_df["entity_type"] == "Technology"
    ].drop_duplicates(subset=["canonical"]).copy()

    negatives = []
    pair_id   = 1

    # ── Strategy 1: Same-crop negatives ──────────────────────────────────────
    print("  Strategy 1: Same-crop negatives...")

    # Group by first word (crop name tends to be first word)
    crop_groups = defaultdict(list)
    for _, row in disease_entities.iterrows():
        words = str(row["canonical"]).split()
        if len(words) > 1:
            crop_groups[words[0]].append(row)

    for crop, rows in crop_groups.items():
        if len(rows) < 2:
            continue
        for row_a, row_b in itertools.combinations(rows, 2):
            if row_a["canonical"] == row_b["canonical"]:
                continue
            negatives.append({
                "pair_id":     f"NEG_CROP_{pair_id:04d}",
                "name_1":      row_a["name"],
                "name_2":      row_b["name"],
                "canonical_1": row_a["canonical"],
                "canonical_2": row_b["canonical"],
                "context_1":   row_a.get("context", ""),
                "context_2":   row_b.get("context", ""),
                "entity_type": "Disease",
                "label":       0,
                "pair_source": "negative_same_crop",
                "confidence":  1.0,
                "note":        f"Different diseases sharing crop word: '{crop}'",
            })
            pair_id += 1

    print(f"    Generated {len(negatives)} same-crop negatives")

    # ── Strategy 2: Same disease-type word ──────────────────────────────────
    print("  Strategy 2: Same-type-word negatives (rust, blight, mildew)...")
    s2_start = len(negatives)

    type_keywords = ["rust", "blight", "mildew", "wilt", "spot", "rot", "smut"]
    type_groups   = defaultdict(list)

    for _, row in disease_entities.iterrows():
        for kw in type_keywords:
            if kw in str(row["canonical"]):
                type_groups[kw].append(row)
                break  # Only assign to one group

    for keyword, rows in type_groups.items():
        if len(rows) < 2:
            continue
        for row_a, row_b in itertools.combinations(rows, 2):
            if row_a["canonical"] == row_b["canonical"]:
                continue
            # Skip if already added in Strategy 1
            negatives.append({
                "pair_id":     f"NEG_TYPE_{pair_id:04d}",
                "name_1":      row_a["name"],
                "name_2":      row_b["name"],
                "canonical_1": row_a["canonical"],
                "canonical_2": row_b["canonical"],
                "context_1":   row_a.get("context", ""),
                "context_2":   row_b.get("context", ""),
                "entity_type": "Disease",
                "label":       0,
                "pair_source": f"negative_same_type_{keyword}",
                "confidence":  1.0,
                "note":        f"Different diseases sharing keyword: '{keyword}'",
            })
            pair_id += 1

    print(f"    Generated {len(negatives) - s2_start} same-type negatives")

    # ── Strategy 3: Random cross-entity negatives ────────────────────────────
    print("  Strategy 3: Random negatives...")
    s3_start = len(negatives)

    disease_list = disease_entities.to_dict("records")
    target_random = max(100, len(negatives))  # Match the positives count roughly

    attempts = 0
    while len(negatives) - s3_start < target_random and attempts < target_random * 10:
        attempts += 1
        row_a = random.choice(disease_list)
        row_b = random.choice(disease_list)
        if row_a["canonical"] == row_b["canonical"]:
            continue
        negatives.append({
            "pair_id":     f"NEG_RAND_{pair_id:04d}",
            "name_1":      row_a["name"],
            "name_2":      row_b["name"],
            "canonical_1": row_a["canonical"],
            "canonical_2": row_b["canonical"],
            "context_1":   row_a.get("context", ""),
            "context_2":   row_b.get("context", ""),
            "entity_type": "Disease",
            "label":       0,
            "pair_source": "negative_random",
            "confidence":  0.95,
            "note":        "Randomly sampled different entities",
        })
        pair_id += 1

    print(f"    Generated {len(negatives) - s3_start} random negatives")
    print(f"  ─── Total negatives: {len(negatives)}")

    return pd.DataFrame(negatives)


def build_final_dataset(pos_df: pd.DataFrame,
                         neg_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Combines positive and negative pairs into the final training CSV.

      Also:
      - Deduplicates pairs (same pair shouldn't appear twice)
      - Shuffles rows (important for training — don't want all positives first)
      - Standardises columns
      - Reports class balance (how many 1s vs 0s)
    """
    FINAL_COLS = [
        "pair_id", "name_1", "name_2", "canonical_1", "canonical_2",
        "context_1", "context_2", "entity_type", "label", "confidence",
        "pair_source", "note"
    ]

    # Ensure all columns exist in both DataFrames
    for col in FINAL_COLS:
        for df in [pos_df, neg_df]:
            if col not in df.columns:
                df[col] = ""

    combined = pd.concat(
        [pos_df[FINAL_COLS], neg_df[FINAL_COLS]],
        ignore_index=True
    )

    # Deduplicate: same (name_1, name_2) pair regardless of order
    combined["_pair_key"] = combined.apply(
        lambda r: tuple(sorted([
            normalise_name(str(r["name_1"])),
            normalise_name(str(r["name_2"]))
        ])),
        axis=1
    )
    combined = combined.drop_duplicates(subset=["_pair_key", "label"])
    combined = combined.drop(columns=["_pair_key"])

    # Shuffle rows
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print class balance
    n_pos = (combined["label"] == 1).sum()
    n_neg = (combined["label"] == 0).sum()
    ratio = n_pos / max(n_neg, 1)
    print(f"\n  Class balance:")
    print(f"    Positive (label=1): {n_pos:5d} pairs")
    print(f"    Negative (label=0): {n_neg:5d} pairs")
    print(f"    Ratio pos/neg:      {ratio:.2f}")

    if ratio > 3.0:
        print(f"  ⚠ Dataset is {ratio:.1f}x skewed toward positives.")
        print(f"    Consider generating more negatives in MANUAL_PATCHES.")

    return combined


def main():
    print("\n" + "═"*60)
    print("  SCRIPT 05 — Final Dataset Builder")
    print("  AgriΛNet Entity Resolution Pipeline")
    print("═"*60)

    # ── Step 1: Load and merge all entity records ─────────────────────────
    print("\n[1/5] Loading and merging all entity records...")
    entities_df = load_and_merge_entities()

    # ── Step 2: Load all positive pairs ──────────────────────────────────
    print("\n[2/5] Loading all positive pairs...")
    pos_df = load_all_positive_pairs()

    if pos_df.empty:
        print("\n  ERROR: No positive pairs found.")
        print("  Please run scripts 01, 02, 03, 04 first.")
        return

    # ── Step 3: Enrich pairs with context ────────────────────────────────
    print("\n[3/5] Enriching pairs with context descriptions...")
    pos_df = enrich_pairs_with_context(pos_df, entities_df)

    # ── Step 4: Generate negative pairs ──────────────────────────────────
    print("\n[4/5] Generating negative (non-matching) pairs...")
    neg_df = generate_negative_pairs(entities_df)
    neg_df = enrich_pairs_with_context(neg_df, entities_df)

    # ── Step 5: Build final dataset ───────────────────────────────────────
    print("\n[5/5] Building final training dataset...")
    final_df = build_final_dataset(pos_df, neg_df)

    # Save outputs
    final_path = PAIRS_DIR / "final_dataset.csv"
    pos_only_path = PAIRS_DIR / "positive_pairs.csv"
    neg_only_path = PAIRS_DIR / "negative_pairs.csv"

    final_df.to_csv(final_path, index=False)
    pos_df.to_csv(pos_only_path, index=False)
    neg_df.to_csv(neg_only_path, index=False)

    print(f"\n  ✓ final_dataset.csv   → {len(final_df):5d} pairs")
    print(f"  ✓ positive_pairs.csv  → {len(pos_df):5d} pairs")
    print(f"  ✓ negative_pairs.csv  → {len(neg_df):5d} pairs")
    print(f"  ✓ all_entities.csv    → {len(entities_df):5d} entities")

    print("\n" + "═"*60)
    print("  🎉 DATASET COMPLETE!")
    print("═"*60)
    print(f"\n  Your CropDP-KGAgriNER dataset is ready at:")
    print(f"    entity_resolution/data/pairs/final_dataset.csv")
    print(f"\n  Load it in your model training code with:")
    print(f"    import pandas as pd")
    print(f"    df = pd.read_csv('data/pairs/final_dataset.csv')")
    print(f"    X = df[['name_1','name_2','context_1','context_2']]")
    print(f"    y = df['label']")
    print()

    return final_df, entities_df


if __name__ == "__main__":
    main()