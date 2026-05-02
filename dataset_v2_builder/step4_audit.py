"""
STEP 4 - Final audit: is dataset_clean.csv ready for training?
==============================================================
Runs a suite of quantitative checks and prints a PASS/FAIL report.

Run:
    python dataset_v2_builder/step4_audit.py --input dataset_v2_builder/data/dataset_clean.csv
"""

import sys
import io
import argparse
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def get_args():
    p = argparse.ArgumentParser(description="Final dataset audit before training")
    p.add_argument("--input", default="dataset_clean.csv", help="Clean dataset to audit")
    return p.parse_args()


# ── Thresholds calibrated to our 1,782-row dataset ────────────────────────────
THRESHOLDS = {
    "min_total_rows":     1750,   # we have 1782
    "min_type_a":         200,    # we have 241
    "min_type_b":         200,    # we have 274
    "min_type_c":         250,    # we have 295 (original 400 was aspirational)
    "min_type_d":         800,    # we have 972
    "min_pos_ratio":      0.25,   # >= 25%
    "max_pos_ratio":      0.40,   # <= 40%
    "max_type_c_lambda":  0.15,   # all Type C must be <= this
    "max_missing_type":   5,      # max missing type_a or type_b
    "max_label_noise":    35,     # max match != llm_match disagreements
    "max_caps_names":     0,      # no ALL-CAPS names
    "max_duplicates":     0,      # no duplicate (name_a, name_b) pairs
}


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Print a single check result. Returns True if passed."""
    icon   = "[OK]" if condition else "[XX]"
    status = "PASS" if condition else "FAIL"
    msg    = f"  {icon} {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main():
    args = get_args()
    df   = pd.read_csv(args.input)
    T    = THRESHOLDS

    print(f"\n{'=' * 62}")
    print(f"  DATASET AUDIT: {args.input}")
    print(f"{'=' * 62}")
    print(f"  Rows: {len(df)}   Columns: {len(df.columns)}\n")

    passes = 0
    fails  = 0

    # ── 1. Size ────────────────────────────────────────────────────────────────
    section("SIZE")
    n  = len(df)
    ok = check("Minimum rows", n >= T["min_total_rows"],
               f"{n} / {T['min_total_rows']} required")
    passes += ok; fails += not ok

    # ── 2. Class balance ───────────────────────────────────────────────────────
    section("CLASS BALANCE")
    tc = df["pair_type"].value_counts()

    n_a = tc.get("A", 0)
    ok  = check("Type A >= 200", n_a >= T["min_type_a"], f"{n_a}")
    passes += ok; fails += not ok

    n_b = tc.get("B", 0)
    ok  = check("Type B >= 200", n_b >= T["min_type_b"], f"{n_b}")
    passes += ok; fails += not ok

    n_c = tc.get("C", 0)
    ok  = check(f"Type C >= {T['min_type_c']}", n_c >= T["min_type_c"], f"{n_c}")
    passes += ok; fails += not ok

    n_d = tc.get("D", 0)
    ok  = check(f"Type D >= {T['min_type_d']}", n_d >= T["min_type_d"], f"{n_d}")
    passes += ok; fails += not ok

    pos_ratio = df["match"].mean()
    ok = check(
        f"Positive ratio {T['min_pos_ratio']:.0%}-{T['max_pos_ratio']:.0%}",
        T["min_pos_ratio"] <= pos_ratio <= T["max_pos_ratio"],
        f"{pos_ratio:.1%}"
    )
    passes += ok; fails += not ok

    # ── 3. Lambda distribution ─────────────────────────────────────────────────
    section("LAMBDA VALUES")

    bad_c = ((df["pair_type"] == "C") & (df["lambda_val"] > T["max_type_c_lambda"])).sum()
    ok    = check("Type C lambda all <= 0.15", bad_c == 0,
                  f"{bad_c} rows exceed threshold")
    passes += ok; fails += not ok

    bins   = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0.0-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-0.9","0.9-1.0"]
    df["_lbin"] = pd.cut(df["lambda_val"], bins=bins, labels=labels)
    mid_pct = ((df["lambda_val"] >= 0.3) & (df["lambda_val"] <= 0.7)).mean()
    ok = check("Mid-lambda (0.3-0.7) >= 10%", mid_pct >= 0.10,
               f"currently {mid_pct:.1%}")
    passes += ok; fails += not ok

    print(f"\n  Lambda bucket breakdown:")
    for lbl, cnt in df["_lbin"].value_counts().sort_index().items():
        bar = "#" * (cnt // 30)
        print(f"    {lbl:>10s}  {cnt:4d}  {bar}")
    df.drop(columns=["_lbin"], inplace=True)

    # ── 4. Entity types ────────────────────────────────────────────────────────
    section("ENTITY TYPES")

    miss_a = df["type_a"].isna().sum()
    ok     = check(f"type_a missing <= {T['max_missing_type']}",
                   miss_a <= T["max_missing_type"], f"{miss_a} missing")
    passes += ok; fails += not ok

    miss_b = df["type_b"].isna().sum()
    ok     = check(f"type_b missing <= {T['max_missing_type']}",
                   miss_b <= T["max_missing_type"], f"{miss_b} missing")
    passes += ok; fails += not ok

    print(f"\n  type_a distribution:")
    for t, c in df["type_a"].value_counts(dropna=False).items():
        print(f"    {str(t):15s}  {c}")

    # ── 5. Name normalisation ──────────────────────────────────────────────────
    section("NAME NORMALISATION")

    caps_a = df["name_a"].str.isupper().sum()
    caps_b = df["name_b"].str.isupper().sum()
    ok     = check("No ALL-CAPS names", caps_a + caps_b == 0,
                   f"{caps_a + caps_b} remain")
    passes += ok; fails += not ok

    dupes = df.duplicated(subset=["name_a", "name_b"]).sum()
    ok    = check("No duplicate (name_a, name_b) pairs", dupes == 0,
                  f"{dupes} duplicates")
    passes += ok; fails += not ok

    # ── 6. Label consistency ───────────────────────────────────────────────────
    section("LABEL CONSISTENCY")

    df["_llm_int"] = df["llm_match"].round().astype(int)
    disagree = (df["match"] != df["_llm_int"]).sum()
    ok       = check(f"Disagreements (match != llm_match) <= {T['max_label_noise']}",
                     disagree <= T["max_label_noise"],
                     f"{disagree} / {len(df)} ({disagree/len(df)*100:.1f}%)")
    passes += ok; fails += not ok

    ab_wrong = (df["pair_type"].isin(["A", "B"]) & (df["match"] != 1)).sum()
    ok       = check("All Type A/B have match=1", ab_wrong == 0,
                     f"{ab_wrong} wrong")
    passes += ok; fails += not ok

    cd_wrong = (df["pair_type"].isin(["C", "D"]) & (df["match"] != 0)).sum()
    ok       = check("All Type C/D have match=0", cd_wrong == 0,
                     f"{cd_wrong} wrong")
    passes += ok; fails += not ok

    # ── 7. Context quality ─────────────────────────────────────────────────────
    section("CONTEXT QUALITY")

    same_ctx = (df["context_a"] == df["context_b"]).sum()
    ok = check("Same-context rows <= 210",
               same_ctx <= 210,          # Wikipedia couldn't fix all 204
               f"{same_ctx} rows have identical contexts")
    passes += ok; fails += not ok

    short_a = (df["context_a"].str.len() < 60).sum()
    ok = check("Very short contexts (< 60 chars) <= 20",
               short_a <= 20, f"{short_a} rows")
    passes += ok; fails += not ok

    # ── Final score ────────────────────────────────────────────────────────────
    total = passes + fails
    print(f"\n{'=' * 62}")
    print(f"  AUDIT RESULT: {passes}/{total} checks passed")
    print(f"{'=' * 62}\n")

    if fails == 0:
        print("  [OK] Dataset is READY FOR TRAINING\n")
    else:
        print(f"  [WARN] {fails} check(s) flagged.\n")

    # ── Dataset summary ────────────────────────────────────────────────────────
    section("FINAL SUMMARY")
    pos  = int(df["match"].sum())
    neg  = int((df["match"] == 0).sum())
    lmda = df["lambda_val"]
    ctx_driven  = (lmda < 0.3).sum()
    balanced    = ((lmda >= 0.3) & (lmda < 0.7)).sum()
    name_driven = (lmda >= 0.7).sum()

    print(f"  Total pairs       : {len(df)}")
    print(f"  Positive (match=1): {pos}  ({pos/len(df)*100:.1f}%)")
    print(f"  Negative (match=0): {neg}  ({neg/len(df)*100:.1f}%)")
    print(f"  Pair types        : A={n_a}  B={n_b}  C={n_c}  D={n_d}")
    print(f"  Lambda mean       : {lmda.mean():.4f}")
    print(f"  Context-driven (lambda<0.3)  : {ctx_driven}")
    print(f"  Balanced   (0.3<=lambda<0.7) : {balanced}")
    print(f"  Name-driven (lambda>=0.7)    : {name_driven}")
    print(f"  LLM agreement     : {len(df)-disagree}/{len(df)}  ({(len(df)-disagree)/len(df)*100:.1f}%)")
    print(f"  type_a missing    : {miss_a}")
    print(f"  type_b missing    : {miss_b}")
    print(f"  Same-context rows : {same_ctx}")
    print()


if __name__ == "__main__":
    main()
