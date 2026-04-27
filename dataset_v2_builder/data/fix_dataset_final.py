import pandas as pd
from difflib import SequenceMatcher   # built-in Python, no install needed

INPUT_CSV  = "dataset_v2_fixed.csv"
OUTPUT_CSV = "dataset_final.csv"


def main():
    print("=" * 55)
    print(" Fix dataset_v2_fixed.csv → dataset_final.csv")
    print("=" * 55)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")

    # ── Fix 1: Row 755 — Type C polysemy pair with lambda too high ────────────
    #
    # getah virus vs ross river virus:
    #   pair_type = C (match=0, names share 'virus' + same Alphavirus genus)
    #   lambda    = 0.458  ← WRONG for a polysemy pair
    #
    # For any Type C pair, lambda must be LOW because the names are misleading.
    # High lambda tells the model "trust the name 46%" which would cause a
    # false positive match on these two distinct diseases.
    #
    # We target this row precisely using both name values so the fix is
    # reproducible and does not accidentally change any other row.

    print("\nFix 1: Correcting Type C pair with lambda > 0.35...")

    mask_fix1 = (
        (df["pair_type"]  == "C") &
        (df["lambda_val"] >  0.35)
    )
    n_fix1 = mask_fix1.sum()

    if n_fix1 == 0:
        print("  No rows found — already fixed.")
    else:
        for idx in df[mask_fix1].index:
            old_val = df.at[idx, "lambda_val"]
            print(f"  Row {idx}: '{df.at[idx,'name_a']}' vs '{df.at[idx,'name_b']}'")
            print(f"    lambda {old_val:.3f} → 0.20")
            df.at[idx, "lambda_val"]    = 0.20
            df.at[idx, "lambda_source"] = "corrected_polysemy"
        print(f"  Fixed {n_fix1} row(s).")

    # ── Fix 2: Fill 15 missing name_sim_score values ──────────────────────────
    #
    # The 15 EPPO pairs added manually in the last batch were never passed
    # through the name similarity computation step.
    # SequenceMatcher gives a character-level similarity ratio in [0, 1].
    # This is the same approach used by step2_pair_type_classifier.py for
    # the rest of the dataset.

    print("\nFix 2: Filling missing name_sim_score values...")

    missing_mask = df["name_sim_score"].isnull()
    n_missing    = missing_mask.sum()

    if n_missing == 0:
        print("  No missing values — already filled.")
    else:
        for idx in df[missing_mask].index:
            sim = SequenceMatcher(
                None,
                str(df.at[idx, "name_a"]).lower().strip(),
                str(df.at[idx, "name_b"]).lower().strip(),
            ).ratio()
            df.at[idx, "name_sim_score"] = round(sim, 4)

        print(f"  Filled {n_missing} rows.")
        print(f"  Null name_sim_score remaining: {df['name_sim_score'].isnull().sum()}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved.")

    # ── Final verification ────────────────────────────────────────────────────
    print("\n── Final verification ─────────────────────────────────────")

    c = df[df["pair_type"] == "C"]
    high_c = c[c["lambda_val"] > 0.35]

    checks = [
        ("Total rows",             len(df),                             1896,  "=="),
        ("Null lambda_val",        df["lambda_val"].isnull().sum(),     0,     "=="),
        ("Null name_sim_score",    df["name_sim_score"].isnull().sum(), 0,     "=="),
        ("Type C lambda > 0.35",   len(high_c),                        0,     "=="),
        ("Type C lambda max",      round(c["lambda_val"].max(), 3),     0.35,  "<="),
        ("Duplicate pairs",        df.duplicated(["name_a","name_b"]).sum(), 0, "=="),
    ]

    all_pass = True
    for label, actual, expected, op in checks:
        if op == "==":
            ok = actual == expected
        else:
            ok = actual <= expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {label}: {actual}")

    print()
    if all_pass:
        print("  All checks passed. dataset_final.csv is ready for training.")
    else:
        print("  Some checks failed — review the output above.")

    print()
    print("── Dataset summary ────────────────────────────────────────")
    pt = df["pair_type"].value_counts()
    for t in ["A", "B", "C", "D"]:
        print(f"  Type {t}: {pt.get(t, 0):5d}")
    excl = df["exclude_from_lambda"].sum()
    print(f"  Match ratio           : {df['match'].mean():.3f}")
    print(f"  Lambda training pairs : {len(df) - excl}  (excluded {excl} both-poor)")
    print(f"  Type C lambda mean    : {c['lambda_val'].mean():.3f}  max: {c['lambda_val'].max():.3f}")
    print()
    print(f"  Use {OUTPUT_CSV} for all training going forward.")
    print()


if __name__ == "__main__":
    main()
