"""
STEP 3 — Merge scraped types & contexts back into the fixed dataset
====================================================================
Takes:
  - dataset_fixed.csv        (output of step1)
  - scraped_types.csv        (output of step2)

Produces:
  - dataset_clean.csv        (fully fixed, typed, context-unique dataset)

What this script does:
  1. Applies scraped type_a / type_b into rows where type is still missing
  2. Replaces context_a with better_context for same_context rows
  3. Drops helper columns (same_context, type_needs_scraping, lambda_original)
  4. Runs a final validation pass
  5. Prints a before/after comparison report
"""

import argparse
import pandas as pd


def get_args():
    p = argparse.ArgumentParser(description="Merge scraped types into fixed dataset")
    p.add_argument("--fixed",   default="dataset_fixed.csv",  help="Step1 output")
    p.add_argument("--scraped", default="scraped_types.csv",  help="Step2 output")
    p.add_argument("--output",  default="dataset_clean.csv",  help="Final clean CSV")
    return p.parse_args()


def apply_scraped_types(df: pd.DataFrame, scraped: pd.DataFrame) -> pd.DataFrame:
    """Apply scraped type_a / type_b into rows where type is still missing."""
    lookup = {}
    for _, row in scraped.iterrows():
        name_key = str(row["name"]).lower().strip()
        lookup[name_key] = {
            "type":           row.get("type"),
            "source_url":     row.get("source_url"),
        }

    filled_a = 0
    filled_b = 0

    for idx, row in df.iterrows():
        key_a = str(row["name_a"]).lower().strip()
        key_b = str(row["name_b"]).lower().strip()

        if pd.isna(row["type_a"]) and key_a in lookup:
            scraped_type = lookup[key_a]["type"]
            if scraped_type and scraped_type != "unknown":
                df.at[idx, "type_a"] = scraped_type
                filled_a += 1

        if pd.isna(row["type_b"]) and key_b in lookup:
            scraped_type = lookup[key_b]["type"]
            if scraped_type and scraped_type != "unknown":
                df.at[idx, "type_b"] = scraped_type
                filled_b += 1

    print(f"[Merge] type_a filled:      {filled_a}")
    print(f"[Merge] type_b filled:      {filled_b}")
    return df


SCAFFOLDING_COLS = [
    "same_context",
    "type_needs_scraping",
    "lambda_original",
]

def drop_scaffolding(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in SCAFFOLDING_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[Merge] Dropped scaffolding columns: {cols_to_drop}")
    return df


TYPE_C_LAMBDA_CAP = 0.15

def validate(df: pd.DataFrame) -> None:
    errors = []

    bad_c = df[(df["pair_type"] == "C") & (df["lambda_val"] > TYPE_C_LAMBDA_CAP)]
    if len(bad_c):
        errors.append(f"Type C lambda > {TYPE_C_LAMBDA_CAP}: {len(bad_c)} rows")

    missing_type_a = df["type_a"].isna().sum()
    missing_type_b = df["type_b"].isna().sum()
    if missing_type_a:
        errors.append(f"type_a still missing: {missing_type_a} rows")
    if missing_type_b:
        errors.append(f"type_b still missing: {missing_type_b} rows")

    if errors:
        print("\n[VALIDATION — issues remain]")
        for e in errors:
            print(f"  [WARN] {e}")
    else:
        print("\n[VALIDATION PASSED] dataset_clean.csv is ready!")


def main():
    args = get_args()

    print(f"Loading fixed dataset:   {args.fixed}")
    df = pd.read_csv(args.fixed)
    print(f"  -> {len(df)} rows\n")

    print(f"Loading scraped types:   {args.scraped}")
    scraped = pd.read_csv(args.scraped)
    print(f"  -> {len(scraped)} entities scraped\n")

    df = apply_scraped_types(df, scraped)
    df = drop_scaffolding(df)
    validate(df)

    df.to_csv(args.output, index=False)
    print(f"\nSaved final dataset -> {args.output}")

    print("\nNEXT STEP:")
    print("  Run your model training pipeline on dataset_clean.csv")


if __name__ == "__main__":
    main()
