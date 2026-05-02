"""
STEP 1 — Fix all known issues in dataset_augmented.csv
=======================================================
Fixes applied (in order):
  1. Normalise name casing  → lowercase + strip whitespace
  2. Fix Type C lambda      → cap at 0.20 for all Type C rows
  3. Infer missing type_a/b → keyword rules from context text
  4. Flag ctx_a == ctx_b    → adds 'same_context' column for review
  5. Fix llm_match dtype    → converts float (0.0/1.0) back to bool

Run:
    python step1_fix_dataset.py \
        --input  dataset_augmented.csv \
        --output dataset_fixed.csv

After running, open dataset_fixed.csv and:
  - Review rows where 'same_context' == True and replace one context
  - Review rows where 'type_needs_scraping' == True (scraper will do this)
"""

import argparse
import re
import pandas as pd

# ── 1. CLI ────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Fix dataset_augmented.csv issues")
    p.add_argument("--input",  default="dataset_augmented.csv", help="Input CSV path")
    p.add_argument("--output", default="dataset_fixed.csv",     help="Output CSV path")
    return p.parse_args()


# ── 2. Name normalisation ─────────────────────────────────────────────────────

def normalise_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip all entity names.
    Why: 'POWDERY MILDEW' and 'powdery mildew' produce different embeddings
    in all-MiniLM-L6-v2. Consistent lowercase eliminates this noise.
    """
    before_a = (df["name_a"].str.isupper()).sum()
    before_b = (df["name_b"].str.isupper()).sum()

    df["name_a"] = df["name_a"].str.lower().str.strip()
    df["name_b"] = df["name_b"].str.lower().str.strip()

    print(f"[1] Name normalisation: fixed {before_a} ALL-CAPS in name_a, "
          f"{before_b} ALL-CAPS in name_b")
    return df


# ── 3. Type C lambda fix ──────────────────────────────────────────────────────

# The maximum sensible lambda for a Type C (hard negative) pair.
# Type C = "names look similar, but they are NOT the same thing."
# Lambda near 0 means "trust context, not the name" — correct for Type C.
# We use 0.15 as the cap because a tiny lambda signal is still informative;
# setting it to exactly 0 can cause numerical issues in some loss functions.
TYPE_C_LAMBDA_CAP = 0.15

def fix_type_c_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap lambda_val at TYPE_C_LAMBDA_CAP for all Type C rows.
    Rows that were > 0.5 are the worst offenders (70 rows found in audit).
    """
    mask_c = df["pair_type"] == "C"
    bad_mask = mask_c & (df["lambda_val"] > TYPE_C_LAMBDA_CAP)
    n_bad = bad_mask.sum()

    # Record original values before capping (for audit trail)
    df.loc[bad_mask, "lambda_original"] = df.loc[bad_mask, "lambda_val"]
    df["lambda_original"] = df.get("lambda_original", pd.NA)

    df.loc[bad_mask, "lambda_val"] = TYPE_C_LAMBDA_CAP
    df.loc[bad_mask, "lambda_source"] = "capped_type_c_fix"

    # Sanity check: confirm no Type C row exceeds cap now
    still_bad = (mask_c & (df["lambda_val"] > TYPE_C_LAMBDA_CAP)).sum()
    assert still_bad == 0, f"Capping failed — {still_bad} rows still above cap"

    print(f"[2] Type C lambda fix: capped {n_bad} rows from > {TYPE_C_LAMBDA_CAP} "
          f"down to {TYPE_C_LAMBDA_CAP}")
    print(f"    Type C lambda stats after fix:")
    print(f"    {df[mask_c]['lambda_val'].describe().to_dict()}")
    return df


# ── 4. Infer entity types from context ───────────────────────────────────────

# Keyword rules derived from analysing existing typed rows in the dataset.
# Order matters: check more specific terms first (bacteria before plant,
# oomycete before fungus, etc.)
TYPE_RULES = [
    # Specific pathogen genus names → fungus
    ("fungus", re.compile(
        r"fungal\s+disease|oomycete|ascomycete|basidiomycete"
        r"|puccinia|fusarium|alternaria|botrytis|sclerotinia"
        r"|magnaporthe|colletotrichum|phytophthora|plasmopara"
        r"|blumeria|venturia|cercospora|cochliobolus|septoria"
        r"|erysiphe|uncinula|podosphaera|guignardia|diplocarpon"
        r"|exserohilum|helminthosporium|mycosphaerella|gibberella",
        re.IGNORECASE
    )),
    # Bacteria
    ("bacteria", re.compile(
        r"bacterial\s+disease|bacterium|bacteria\b"
        r"|xanthomonas|pseudomonas|erwinia|agrobacterium"
        r"|clavibacter|ralstonia|streptomyces",
        re.IGNORECASE
    )),
    # Virus
    ("virus", re.compile(
        r"\bvirus\b|\bviral\s+disease\b|viridae|begomovirus"
        r"|potyvirus|tobamovirus|luteovirus|caulimovirus"
        r"|geminivirus|ilarvirus|closterovirus",
        re.IGNORECASE
    )),
    # Pest (insects, mites, nematodes)
    ("pest", re.compile(
        r"\binsect\b|\bmite\b|\bnematode\b|\baphid\b|\bwhitefly\b"
        r"|\bthrip\b|\bweevil\b|\bborers?\b|\bcaterpillar\b",
        re.IGNORECASE
    )),
    # Plant (host organism, not the disease itself)
    ("plant", re.compile(
        r"\bplant\s+species\b|\bcrop\s+species\b|\blegume\b|\bcereal\b"
        r"|\bgrass\b|\bweed\b|\btaxon\b|\bgenus\b|\bfamily\b",
        re.IGNORECASE
    )),
]

def infer_type_from_context(context: str) -> str | None:
    """
    Return the best entity type for a given context string,
    or None if no rule matches (needs manual / scraper assignment).
    """
    ctx = str(context)
    for entity_type, pattern in TYPE_RULES:
        if pattern.search(ctx):
            return entity_type
    return None  # caller will mark as 'needs_scraping'


def fill_missing_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row where type_a or type_b is NaN, try to infer
    the type from the corresponding context text.
    """
    needs_a = df["type_a"].isna()
    needs_b = df["type_b"].isna()

    print(f"[3] Type inference: {needs_a.sum()} rows missing type_a, "
          f"{needs_b.sum()} rows missing type_b")

    # Infer type_a from context_a
    df.loc[needs_a, "type_a"] = (
        df.loc[needs_a, "context_a"].apply(infer_type_from_context)
    )
    # Infer type_b from context_b
    df.loc[needs_b, "type_b"] = (
        df.loc[needs_b, "context_b"].apply(infer_type_from_context)
    )

    # Mark rows that inference could not resolve — scraper will handle these
    df["type_needs_scraping"] = (
        df["type_a"].isna() | df["type_b"].isna()
    ).astype(int)

    still_a = df["type_a"].isna().sum()
    still_b = df["type_b"].isna().sum()
    scrape_needed = df["type_needs_scraping"].sum()

    print(f"    After inference: {still_a} type_a still missing, "
          f"{still_b} type_b still missing")
    print(f"    Rows flagged for scraper: {scrape_needed}")

    return df


# ── 5. Flag ctx_a == ctx_b ────────────────────────────────────────────────────

def flag_same_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'same_context' column = 1 when context_a and context_b are identical.
    These rows need different contexts before training.

    We do NOT auto-fix them here because:
    - Replacing a context blindly could introduce wrong information
    - The scraper (step2) will fetch real contexts for these entities
    - After scraping, run step3_merge to replace them
    """
    df["same_context"] = (df["context_a"] == df["context_b"]).astype(int)
    n_same = df["same_context"].sum()
    print(f"[4] Same-context flag: {n_same} rows flagged (context_a == context_b)")
    print(f"    These need unique contexts — scraper will resolve them")
    return df


# ── 6. Fix llm_match dtype ────────────────────────────────────────────────────

def fix_llm_match_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    llm_match was bool in v1 but became float64 (0.0/1.0) after augmentation.
    Convert back to int (0/1) for consistency with the 'match' column.
    """
    df["llm_match"] = df["llm_match"].round().astype(int)
    print(f"[5] llm_match dtype fixed: float64 -> int")
    return df


# ── 7. Count and report disagreements ─────────────────────────────────────────

def report_disagreements(df: pd.DataFrame) -> None:
    """
    Print rows where match != llm_match so you can review them manually.
    These are either labelling errors or genuinely hard edge cases.
    """
    disagree = df[df["match"] != df["llm_match"]]
    print(f"\n[6] Label disagreements (match != llm_match): {len(disagree)} rows")
    if len(disagree) > 0:
        print("    Review these manually before training:")
        print(disagree[["name_a", "name_b", "pair_type",
                         "match", "llm_match", "lambda_val"]]
              .to_string(index=False))


# ── 8. Validation checks ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    Run a set of assertions to make sure all fixes landed correctly.
    Will raise AssertionError with a clear message if something is wrong.
    """
    errors = []

    # No Type C row should have lambda above the cap
    bad_c = df[(df["pair_type"] == "C") & (df["lambda_val"] > TYPE_C_LAMBDA_CAP)]
    if len(bad_c):
        errors.append(f"Type C lambda still > {TYPE_C_LAMBDA_CAP}: {len(bad_c)} rows")

    # All Type A/B rows should have match == 1
    ab_no_match = df[df["pair_type"].isin(["A", "B"]) & (df["match"] != 1)]
    if len(ab_no_match):
        errors.append(f"Type A/B rows with match != 1: {len(ab_no_match)}")

    # All Type C/D rows should have match == 0
    cd_match = df[df["pair_type"].isin(["C", "D"]) & (df["match"] != 0)]
    if len(cd_match):
        errors.append(f"Type C/D rows with match != 0: {len(cd_match)}")

    # No ALL-CAPS names
    caps_a = df["name_a"].str.isupper().sum()
    caps_b = df["name_b"].str.isupper().sum()
    if caps_a + caps_b > 0:
        errors.append(f"ALL-CAPS names still present: {caps_a} in name_a, {caps_b} in name_b")

    # No exact duplicate pairs
    dupes = df.duplicated(subset=["name_a", "name_b"]).sum()
    if dupes:
        errors.append(f"Duplicate name pairs found: {dupes}")

    if errors:
        print("\n[VALIDATION FAILED]")
        for e in errors:
            print(f"  X {e}")
        raise SystemExit("Fix the errors above before proceeding to step 2.")
    else:
        print("\n[VALIDATION PASSED] All checks passed *")


# ── 9. Summary report ─────────────────────────────────────────────────────────

def print_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY -- BEFORE vs AFTER")
    print("=" * 60)

    rows = len(df_after)
    print(f"Total rows:           {rows}")
    print(f"Pair type counts:     {df_after['pair_type'].value_counts().to_dict()}")
    print(f"Match distribution:   {df_after['match'].value_counts().to_dict()}")

    print(f"\nType coverage:")
    print(f"  type_a missing:     {df_after['type_a'].isna().sum()}")
    print(f"  type_b missing:     {df_after['type_b'].isna().sum()}")
    print(f"  needs scraping:     {df_after.get('type_needs_scraping', pd.Series([0])).sum()}")

    print(f"\nContext issues:")
    print(f"  same_context rows:  {df_after.get('same_context', pd.Series([0])).sum()}")

    print(f"\nLambda stats:")
    print(f"  Type C lambda max:  {df_after[df_after['pair_type']=='C']['lambda_val'].max():.4f}")
    print(f"  Overall mean:       {df_after['lambda_val'].mean():.4f}")

    # Lambda bucket distribution
    bins   = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
    df_after["_lbin"] = pd.cut(df_after["lambda_val"], bins=bins, labels=labels)
    print(f"\nLambda distribution:")
    print(df_after["_lbin"].value_counts().sort_index().to_string())
    df_after.drop(columns=["_lbin"], inplace=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)
    df_original = df.copy()  # keep for before/after comparison
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # Apply fixes in sequence
    df = normalise_names(df)
    df = fix_type_c_lambda(df)
    df = fill_missing_types(df)
    df = flag_same_context(df)
    df = fix_llm_match_dtype(df)
    report_disagreements(df)

    # Validate everything is correct
    validate(df)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved fixed dataset -> {args.output}")

    # Print summary
    print_summary(df_original, df)

    print("\nNEXT STEP:")
    print("  Review the 40 label disagreement rows above")
    print("  Fix any obvious labeling errors in dataset_fixed.csv")
    print("  Then run step2_scrape_types.py")


if __name__ == "__main__":
    main()
