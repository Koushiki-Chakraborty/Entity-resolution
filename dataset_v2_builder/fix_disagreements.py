"""
fix_disagreements.py — Apply reviewed label corrections to dataset_fixed.csv
=============================================================================
Expert-reviewed corrections for the 40 match != llm_match rows from step1.
Each correction was assessed by domain knowledge:
  - LLM was right → change `match` (and sometimes `pair_type`)
  - You were right → leave as-is
  - Obvious wrong label → flip

Run:
    python dataset_v2_builder/fix_disagreements.py

Input:  dataset_v2_builder/data/dataset_fixed.csv
Output: dataset_v2_builder/data/dataset_fixed.csv  (overwritten in-place)
        dataset_v2_builder/data/fix_disagreements_log.txt
"""

import sys, io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from pathlib import Path

INPUT  = Path("dataset_v2_builder/data/dataset_fixed.csv")
OUTPUT = Path("dataset_v2_builder/data/dataset_fixed.csv")  # overwrite in-place
LOG    = Path("dataset_v2_builder/data/fix_disagreements_log.txt")

# ── Correction table ───────────────────────────────────────────────────────────
# Each entry: (name_a, name_b, new_match, new_pair_type or None, reason)
# name_a / name_b are ALREADY lowercase (step1 normalised them)
# new_pair_type = None means "don't change pair_type"

CORRECTIONS = [
    # ── LLM was RIGHT — change match 0 → 1 ─────────────────────────────────────

    # maize streak virus (msv) is just the abbreviation of maize streak virus
    ("maize streak virus", "maize streak virus (msv)",
     1, "A",
     "Same virus; (msv) is the abbreviation. LLM correct."),

    # Maize streak disease is caused by maize streak virus — same entity conceptually
    ("maize streak virus (msv)", "maize streak disease",
     1, "B",
     "Maize streak disease caused by MSV; synonymous entities. LLM correct."),

    # Isariopsis leaf spot = grape leaf spot (Isariopsis clavispora causes grape leaf spot)
    ("isariopsis leaf spot", "grape leaf spot",
     1, "B",
     "Isariopsis leaf spot = grape leaf spot. LLM correct."),

    # Septoria leaf spot is a category; tomato septoria leaf spot is a specific instance
    ("septoria leaf spot", "tomato septoria leaf spot",
     1, "B",
     "Tomato septoria leaf spot is a septoria leaf spot. LLM correct."),

    # Leaf spot of tomato = tomato septoria leaf spot
    ("tomato septoria leaf spot", "leaf spot of tomato",
     1, "A",
     "Same disease, different surface forms. LLM correct."),

    # Grape powdery mildew IS a powdery mildew (caused by Erysiphe necator)
    ("grape powdery mildew", "powdery mildew",
     1, "B",
     "Grape powdery mildew is a powdery mildew. LLM correct."),

    # Fusarium wilt of cucumber and Panama disease are both fusarium wilts
    # (panama disease = fusarium wilt of banana); they share the same disease class
    ("fusarium wilt of cucumber", "panama disease",
     1, "B",
     "Both are fusarium wilts; related disease entities. LLM correct."),

    # WSMV (wheat streak mosaic virus) is a potyvirus — genus/member relationship
    ("potyvirus", "wsmv",
     1, "B",
     "WSMV is a potyvirus; genus-member relationship. LLM correct."),

    # Kashmir bee virus belongs to the Iflavirus genus
    ("kashmir bee virus", "iflavirus",
     1, "B",
     "Kashmir bee virus is an iflavirus. LLM correct."),

    # Septoria leaf spot and leaf spot of tomato — close enough to treat as match
    # (septoria leaf spot of tomato = leaf spot of tomato when context is tomato)
    ("septoria leaf spot", "leaf spot of tomato",
     1, "B",
     "In tomato context, septoria leaf spot = leaf spot of tomato. LLM correct."),

    # ── YOU WERE WRONG — change match 1 → 0 ─────────────────────────────────────

    # Puumala virus and Hantaan virus are DIFFERENT hantavirus species — not the same entity
    ("puumala virus", "hantaan virus",
     0, "C",
     "Both hantaviruses but DIFFERENT species. Incorrect to mark as match=1."),

    # CCMV and BMV are both bromoviruses but distinctly different viruses
    ("cowpea chlorotic mottle bromovirus", "brome mosaic bromovirus",
     0, "C",
     "Both bromoviruses but distinct viruses. Incorrect match=1."),
]


def find_row(df: pd.DataFrame, na: str, nb: str):
    """Return the index of the row with name_a==na and name_b==nb, or None."""
    mask = (df["name_a"] == na) & (df["name_b"] == nb)
    hits = df[mask]
    if len(hits) == 1:
        return hits.index[0]
    # Try reversed order too
    mask2 = (df["name_a"] == nb) & (df["name_b"] == na)
    hits2 = df[mask2]
    if len(hits2) == 1:
        return hits2.index[0]
    return None


def main():
    print(f"Loading: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"  Loaded {len(df)} rows\n")

    log_lines = []
    applied = 0
    skipped = 0

    for (na, nb, new_match, new_type, reason) in CORRECTIONS:
        idx = find_row(df, na, nb)

        if idx is None:
            msg = f"  [NOT FOUND]  '{na}'  /  '{nb}'"
            print(msg)
            log_lines.append(msg)
            skipped += 1
            continue

        old_match     = df.at[idx, "match"]
        old_pair_type = df.at[idx, "pair_type"]

        # Apply match change
        df.at[idx, "match"] = new_match

        # Apply pair_type change if requested
        if new_type is not None:
            df.at[idx, "pair_type"] = new_type

        msg = (
            f"  [OK]  '{na}'  /  '{nb}'\n"
            f"        match: {old_match} -> {new_match}   "
            f"pair_type: {old_pair_type} -> {new_type or old_pair_type}\n"
            f"        Reason: {reason}"
        )
        print(msg)
        log_lines.append(msg)
        applied += 1

    # Verify: also fix match==1 for pair_type C/D rows that we just changed to match==1
    # and adjust lambda for newly-matched rows (give them a moderate lambda)
    # Newly matched Type B rows should have lambda in the balanced range
    for (na, nb, new_match, new_type, _) in CORRECTIONS:
        if new_match == 1 and new_type in ("A", "B"):
            idx = find_row(df, na, nb)
            if idx is not None and df.at[idx, "lambda_val"] <= 0.15:
                # Was a Type C with capped lambda — adjust to sensible value for a match
                old_lam = df.at[idx, "lambda_val"]
                df.at[idx, "lambda_val"] = 0.35
                df.at[idx, "lambda_source"] = "disagreement_correction"
                note = f"  [LAMBDA]  '{na}' / '{nb}': {old_lam:.3f} -> 0.35 (now a match)"
                print(note)
                log_lines.append(note)

    print(f"\nSummary: {applied} corrections applied, {skipped} rows not found")

    # Save
    df.to_csv(OUTPUT, index=False)
    print(f"Saved -> {OUTPUT}")

    # Write log
    LOG.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Log   -> {LOG}")

    # Count remaining disagreements
    remaining = (df["match"] != df["llm_match"]).sum()
    print(f"\nRemaining match != llm_match after fixes: {remaining} rows")
    print("(Some disagreements are intentional — we kept our label as ground truth)")


if __name__ == "__main__":
    main()
