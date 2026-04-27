"""
=============================================================================
Step 3 — Fix Dataset + Synthetic Expert Annotation
Agricultural Disease Entity Resolution
=============================================================================
"""

import os
import json
import time
import pathlib
import numpy as np
import pandas as pd
from collections import Counter

# ── Auto-load .env (project root is 2 levels up from this file) ───────────────
_HERE   = pathlib.Path(__file__).resolve().parent
_ROOT   = _HERE.parent.parent
_DOTENV = _ROOT / ".env"

if _DOTENV.exists():
    with open(_DOTENV, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_CSV        = str(_HERE / "dataset_v2_labeled.csv")
FIXED_CSV        = str(_HERE / "dataset_v2_fixed.csv")
EXPERT_50_CSV    = str(_HERE / "expert_annotation_50.csv")
EXPERT_LABEL_CSV = str(_HERE / "expert_lambda_50.csv")
REPORT_TXT       = str(_HERE / "agreement_report.txt")


# =============================================================================
# PART A — FIX DATASET
# =============================================================================

def fix_dataset(df):
    log = []
    df  = df.copy()

    if "lambda_source" not in df.columns:
        df["lambda_source"] = "original_llm"

    # Fix 1: Safety net — assign lambda=0.15 to any remaining null rows
    null_mask = df["lambda_val"].isnull()
    n_nulls   = null_mask.sum()
    if n_nulls > 0:
        df.loc[null_mask, "lambda_val"]    = 0.15
        df.loc[null_mask, "lambda_source"] = "assigned_polysemy_prior"
        log.append(f"Fix 1: Assigned lambda=0.15 to {n_nulls} null rows (safety net)")
    else:
        log.append("Fix 1: No null lambda rows — all 15 pairs labeled correctly ✓")

    # Fix 2: Correct corrupt Type C lambdas (polysemy pairs with lambda > 0.5)
    # For polysemy pairs the model MUST trust context, not name → lambda must be LOW.
    corrupt_mask = (
        (df["pair_type"]     == "C") &
        (df["lambda_val"]    >  0.5) &
        (df["lambda_source"] == "original_llm")
    )
    n_corrupt = corrupt_mask.sum()
    if n_corrupt > 0:
        for idx in df[corrupt_mask].index:
            orig = df.at[idx, "lambda_val"]
            log.append(
                f"Fix 2: '{df.at[idx,'name_a']}' vs '{df.at[idx,'name_b']}'"
                f"  lambda {orig:.3f} → 0.20"
            )
        df.loc[corrupt_mask, "lambda_val"]    = 0.20
        df.loc[corrupt_mask, "lambda_source"] = "corrected_polysemy"
    else:
        log.append("Fix 2: No corrupt Type C lambdas found ✓")

    # Fix 3: Flag both-poor context pairs — exclude from lambda loss but keep
    # for contrastive (match) training.
    both_poor = (
        (df["context_quality_a"] == "poor") &
        (df["context_quality_b"] == "poor")
    )
    df["exclude_from_lambda"] = both_poor.astype(int)
    n_flagged = both_poor.sum()
    log.append(f"Fix 3: Flagged {n_flagged} both-poor pairs with exclude_from_lambda=1")

    # Fix 4: Nudge exact lambda=0.0 on Type C pairs to 0.10
    # Exact zero can cause gradient saturation in sigmoid / ReLU networks.
    # 0.10 still means "almost completely trust context" but is numerically safe.
    zero_c_mask = (
        (df["pair_type"]  == "C") &
        (df["lambda_val"] == 0.0)
    )
    n_zeros = zero_c_mask.sum()
    if n_zeros > 0:
        df.loc[zero_c_mask, "lambda_val"]    = 0.10
        df.loc[zero_c_mask, "lambda_source"] = "nudged_from_zero"
        log.append(f"Fix 4: Nudged {n_zeros} exact-zero Type C lambdas → 0.10")
    else:
        log.append("Fix 4: No exact-zero Type C lambdas found ✓")

    return df, log


# =============================================================================
# PART B — BUILD 50-PAIR EXPERT ANNOTATION SET
# =============================================================================

def build_expert_50(df):
    selected = []

    # 10 x Type A — mid-range lambda, both contexts good
    type_a = df[
        (df["pair_type"]         == "A") &
        (df["context_quality_a"] == "good") &
        (df["context_quality_b"] == "good") &
        (df["lambda_val"].between(0.3, 0.7))
    ].head(10)
    selected.append(type_a)

    # 10 x Type B — hardest synonyms (lowest name similarity)
    type_b = df[
        (df["pair_type"]         == "B") &
        (df["context_quality_a"] == "good") &
        (df["context_quality_b"] == "good")
    ].sort_values("name_sim_score").head(10)
    selected.append(type_b)

    # 10 x Type C — all polysemy pairs with usable contexts
    type_c = df[
        (df["pair_type"]         == "C") &
        (df["context_quality_a"].isin(["good", "medium"])) &
        (df["context_quality_b"].isin(["good", "medium"]))
    ].head(10)
    selected.append(type_c)

    # 10 x Type D — clear non-matches, both good, low lambda
    type_d = df[
        (df["pair_type"]         == "D") &
        (df["context_quality_a"] == "good") &
        (df["context_quality_b"] == "good") &
        (df["lambda_val"]        <  0.2)
    ].head(10)
    selected.append(type_d)

    # 10 x suspicious — B/C pairs where lambda > 0.6 (worth expert review)
    suspicious = df[
        (df["pair_type"].isin(["B", "C"])) &
        (df["lambda_val"]        >  0.6) &
        (df["context_quality_a"] == "good")
    ].head(10)
    selected.append(suspicious)

    expert_df = (
        pd.concat(selected)
        .drop_duplicates()
        .head(50)
        .reset_index(drop=True)
    )
    expert_df["annotation_id"] = [f"E{i+1:03d}" for i in range(len(expert_df))]
    return expert_df


# =============================================================================
# PART C — SYNTHETIC EXPERT ANNOTATION (OPTIONAL, uses GPT-4o)
# =============================================================================

EXPERT_PROMPT = """You are a plant pathologist and expert in agricultural disease taxonomy with 20 years of experience.

--- RECORD A ---
Name: {name_a}
Description: {ctx_a}

--- RECORD B ---
Name: {name_b}
Description: {ctx_b}

JUDGMENT 1: Are these the SAME disease? (true/false)
JUDGMENT 2: Lambda — how important is the NAME alone (0=need context entirely, 1=name alone is enough)?

Return ONLY valid JSON:
{{"match": true_or_false, "lambda": 0.0_to_1.0, "reasoning": "one sentence"}}"""


def get_expert_label(client, name_a, ctx_a, name_b, ctx_b, retries=3):
    prompt = EXPERT_PROMPT.format(
        name_a=name_a, ctx_a=str(ctx_a)[:300],
        name_b=name_b, ctx_b=str(ctx_b)[:300],
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            assert isinstance(result.get("match"), bool)
            lam = float(result.get("lambda", -1))
            assert 0.0 <= lam <= 1.0
            result["lambda"] = lam
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"match": None, "lambda": None, "reasoning": f"ERROR: {e}"}


def run_expert_annotation(expert_df):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n  [SKIP] OPENAI_API_KEY not set — skipping Part C")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("\n  [SKIP] openai not installed — pip install openai")
        return None

    client = OpenAI(api_key=api_key)
    print(f"\n  Calling GPT-4o for {len(expert_df)} pairs ...")
    print("  Estimated cost: ~$0.15–0.25   (~3–5 min)")

    results = []
    for i, (_, row) in enumerate(expert_df.iterrows()):
        result = get_expert_label(
            client,
            row["name_a"], row["context_a"],
            row["name_b"], row["context_b"],
        )
        results.append({
            "annotation_id":    row["annotation_id"],
            "name_a":           row["name_a"],
            "name_b":           row["name_b"],
            "pair_type":        row["pair_type"],
            "original_match":   int(row["match"]),
            "original_lambda":  row["lambda_val"],
            "expert_match":     result.get("match"),
            "expert_lambda":    result.get("lambda"),
            "expert_reasoning": result.get("reasoning", ""),
        })
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(expert_df)} done")
        time.sleep(1.2)

    return pd.DataFrame(results)


def compute_agreement(results_df):
    from scipy.stats import pearsonr

    valid  = results_df.dropna(subset=["original_lambda", "expert_lambda"])
    orig   = valid["original_lambda"].astype(float).values
    expert = valid["expert_lambda"].astype(float).values

    r, p_val    = pearsonr(orig, expert)
    mae         = float(np.mean(np.abs(orig - expert)))
    match_agree = (
        valid["original_match"].astype(int) ==
        valid["expert_match"].apply(lambda x: 1 if x else 0)
    ).mean()

    lines = ["=" * 60, " EXPERT AGREEMENT REPORT", "=" * 60]
    lines.append(f"\nPairs evaluated : {len(valid)} / {len(results_df)}")
    lines.append(f"\n── Lambda agreement ──────────────────────────────────")
    lines.append(f"  Pearson r : {r:.3f}")
    lines.append(f"  p-value   : {p_val:.4f}")
    lines.append(f"  MAE       : {mae:.3f}")
    lines.append(f"\n── Match label agreement ─────────────────────────────")
    lines.append(f"  {match_agree*100:.1f}%")
    lines.append(f"\n── Interpretation ────────────────────────────────────")
    if r >= 0.6:
        lines.append(f"  r={r:.2f} ≥ 0.6 → LLM labels TRUSTWORTHY ✓")
    elif r >= 0.4:
        lines.append(f"  r={r:.2f} — MODERATE. Consider re-labeling B/C pairs.")
    else:
        lines.append(f"  r={r:.2f} < 0.4 — WEAK. Re-label with GPT-4o.")

    return "\n".join(lines), {"r": r, "p": p_val, "mae": mae, "match_agree": match_agree}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" Step 3 — Fix Dataset + Synthetic Expert Annotation")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[PART A] Loading {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n  ERROR: {INPUT_CSV} not found.")
        print("  Make sure dataset_v2_labeled.csv is in the same folder.")
        return

    print(f"  Rows        : {len(df)}")
    print(f"  Null lambdas: {df['lambda_val'].isnull().sum()}")
    print(f"  Type C pairs: {(df['pair_type']=='C').sum()}")

    # ── Fix ───────────────────────────────────────────────────────────────────
    df_fixed, fix_log = fix_dataset(df)

    print("\n  Fixes applied:")
    for line in fix_log:
        print(f"    * {line}")

    df_fixed.to_csv(FIXED_CSV, index=False)
    print(f"  Saved -> {FIXED_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n-- Final dataset summary -------------------------------------")
    print(f"  Total rows              : {len(df_fixed)}")
    pt = df_fixed["pair_type"].value_counts()
    for t in ["A", "B", "C", "D"]:
        n = pt.get(t, 0)
        print(f"  Type {t}                 : {n:5d}")
    excl = df_fixed["exclude_from_lambda"].sum()
    print(f"  Lambda training pairs   : {len(df_fixed) - excl}  (excluded {excl} both-poor)")
    print(f"  Match ratio             : {df_fixed['match'].mean():.3f}  (ideal 0.25–0.40)")

    c_lambdas = df_fixed[df_fixed["pair_type"] == "C"]["lambda_val"]
    print(f"\n  Type C lambda after fixes:")
    print(f"    min={c_lambdas.min():.3f}  max={c_lambdas.max():.3f}  mean={c_lambdas.mean():.3f}")
    print(f"    (all should be <= 0.35 — confirms polysemy pairs have low lambda)")

    # ── Part B ────────────────────────────────────────────────────────────────
    print(f"\n[PART B] Building 50-pair expert annotation set ...")
    expert_df = build_expert_50(df_fixed)
    expert_df.to_csv(EXPERT_50_CSV, index=False)
    print(f"  Saved -> {EXPERT_50_CSV}")
    pt_dist = expert_df["pair_type"].value_counts()
    for t in ["A", "B", "C", "D"]:
        print(f"    Type {t}: {pt_dist.get(t, 0)}")

    # ── Part C ────────────────────────────────────────────────────────────────
    print(f"\n[PART C] Synthetic expert annotation via GPT-4o ...")
    results_df = run_expert_annotation(expert_df)

    if results_df is not None:
        results_df.to_csv(EXPERT_LABEL_CSV, index=False)
        print(f"  Saved -> {EXPERT_LABEL_CSV}")
        try:
            report_text, metrics = compute_agreement(results_df)
            print("\n" + report_text)
            with open(REPORT_TXT, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"\n  Report saved -> {REPORT_TXT}")
        except ImportError:
            print("  [SKIP] scipy not installed — pip install scipy")
    else:
        print(f"\n  Fixed dataset ready  : {FIXED_CSV}")
        print(f"  Expert 50 set ready  : {EXPERT_50_CSV}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" NEXT STEPS")
    print("=" * 60)
    print("""
  Training dataset : dataset_v2_fixed.csv

  Quick-start in your training code:
    df = pd.read_csv("dataset_v2_fixed.csv")

    # Lambda loss  → exclude both-poor pairs
    df_lambda = df[df["exclude_from_lambda"] == 0]

    # Contrastive match loss → use all pairs
    df_match = df

    # Oversample Type C (polysemy) 3x during training
    weights = df["pair_type"].map({"C": 3.0, "B": 2.0, "A": 1.0, "D": 1.0})
""")


if __name__ == "__main__":
    main()
