"""
ML Validation Script for dataset_v2_fixed.csv
==============================================
Checks every data quality issue that would hurt model training.
Prints a PASS/WARN/FAIL per check with a plain-English explanation.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
from collections import Counter

DATA_DIR = pathlib.Path(__file__).resolve().parent
CSV      = DATA_DIR / "dataset_v2_fixed.csv"

PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

results = []   # (status, check_name, detail)

def log(status, name, detail):
    results.append((status, name, detail))
    tag = {"[PASS]": "PASS", "[WARN]": "WARN", "[FAIL]": "FAIL"}[status]
    print(f"  {tag}  {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")

print()
print("=" * 65)
print("  ML VALIDATION  —  dataset_v2_fixed.csv")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV)
    print(f"\n  Loaded: {len(df)} rows  x  {len(df.columns)} columns\n")
except FileNotFoundError:
    print(f"\n  FATAL: {CSV} not found. Run step3_fix_and_expert.py first.")
    sys.exit(1)

REQUIRED_COLS = [
    "name_a","context_a","name_b","context_b",
    "match","llm_match","lambda_val",
    "context_quality_a","context_quality_b",
    "pair_type","name_sim_score","exclude_from_lambda",
]

# =============================================================================
# CHECK 1 — Required columns present
# =============================================================================
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    log(FAIL, "Schema: required columns present",
        f"Missing columns: {missing_cols}\n"
        f"Add them before training — model will crash without these.")
else:
    log(PASS, "Schema: required columns present", "")

# =============================================================================
# CHECK 2 — No nulls in critical training columns
# =============================================================================
critical = ["name_a","context_a","name_b","context_b","match","lambda_val"]
null_counts = {c: int(df[c].isnull().sum()) for c in critical if c in df.columns}
bad = {c:n for c,n in null_counts.items() if n > 0}
if bad:
    detail = "\n".join(f"  {c}: {n} nulls" for c,n in bad.items())
    detail += "\nNull inputs will crash the tokenizer / loss function."
    log(FAIL, "Nulls: critical training columns", detail)
else:
    log(PASS, "Nulls: critical training columns", "")

# =============================================================================
# CHECK 3 — lambda_val range [0, 1]
# =============================================================================
bad_lambda = df[(df["lambda_val"] < 0) | (df["lambda_val"] > 1)]
if len(bad_lambda):
    log(FAIL, "Lambda range [0,1]",
        f"{len(bad_lambda)} rows outside [0,1]. "
        f"Lambda goes into a sigmoid — values outside range corrupt training.")
else:
    log(PASS, "Lambda range [0,1]", "")

# =============================================================================
# CHECK 4 — No exact lambda = 0.0 on Type C
# =============================================================================
zero_c = df[(df["pair_type"] == "C") & (df["lambda_val"] == 0.0)]
if len(zero_c):
    log(FAIL, "No exact lambda=0.0 on Type C (polysemy)",
        f"{len(zero_c)} rows still have lambda=0.0 on polysemy pairs.\n"
        f"Exact zero causes gradient vanishing in sigmoid networks.")
else:
    log(PASS, "No exact lambda=0.0 on Type C (polysemy)", "")

# =============================================================================
# CHECK 5 — Class balance (match ratio)
# =============================================================================
match_ratio = df["match"].mean()
n_pos = df["match"].sum()
n_neg = len(df) - n_pos
detail = f"Positives: {int(n_pos)}  Negatives: {int(n_neg)}  Ratio: {match_ratio:.3f}"
if match_ratio < 0.20:
    log(FAIL, "Class balance (match ratio)",
        detail + "\nToo few positives (<20%). Model will predict all-negative. "
        "Use WeightedRandomSampler or class_weight='balanced'.")
elif match_ratio > 0.50:
    log(FAIL, "Class balance (match ratio)",
        detail + "\nToo many positives (>50%). Model will over-predict matches.")
elif match_ratio < 0.25:
    log(WARN, "Class balance (match ratio)",
        detail + "\nSlightly low. Use WeightedRandomSampler during training.")
else:
    log(PASS, "Class balance (match ratio)", detail)

# =============================================================================
# CHECK 6 — Pair type distribution
# =============================================================================
pt = df["pair_type"].value_counts()
detail_lines = []
status = PASS
for t in ["A","B","C","D"]:
    n = pt.get(t, 0)
    pct = n / len(df) * 100
    detail_lines.append(f"  Type {t}: {n:5d} ({pct:.1f}%)")
    if t == "C" and n < 15:
        detail_lines.append(f"    -> Too few polysemy pairs! Model won't learn polysemy disambiguation.")
        status = WARN
    if t == "D" and pct > 70:
        detail_lines.append(f"    -> Type D dominates ({pct:.0f}%). Model may ignore hard cases.")
        if status == PASS: status = WARN

log(status, "Pair type distribution", "\n".join(detail_lines))

# =============================================================================
# CHECK 7 — LLM vs ground-truth match agreement
# =============================================================================
if "llm_match" in df.columns:
    df["llm_int"] = df["llm_match"].apply(lambda x: 1 if str(x).lower() == "true" else 0)
    agree = (df["match"] == df["llm_int"]).mean()
    disagree_n = (df["match"] != df["llm_int"]).sum()
    detail = f"Agreement: {agree*100:.1f}%  ({int(disagree_n)} disagreements)"
    if agree < 0.80:
        log(FAIL, "LLM vs ground-truth match agreement",
            detail + "\nLess than 80% agreement means noisy labels. "
            "Bad labels = bad model. Re-label disagreements manually.")
    elif agree < 0.90:
        log(WARN, "LLM vs ground-truth match agreement",
            detail + "\nBelow 90%. Review the disagreeing rows; "
            "they may be ambiguous cases worth keeping but annotate them.")
    else:
        log(PASS, "LLM vs ground-truth match agreement", detail)

# =============================================================================
# CHECK 8 — Lambda distribution (bimodal expected)
# =============================================================================
lam = df["lambda_val"]
pct_low  = (lam < 0.2).mean()   # context-heavy
pct_mid  = ((lam >= 0.2) & (lam <= 0.8)).mean()
pct_high = (lam > 0.8).mean()   # name-heavy
detail = (
    f"  Low (<0.2):  {pct_low*100:.1f}%  — 'trust context'\n"
    f"  Mid (0.2-0.8): {pct_mid*100:.1f}%\n"
    f"  High (>0.8): {pct_high*100:.1f}%  — 'trust name'"
)
if pct_low + pct_high < 0.40:
    log(WARN, "Lambda distribution (bimodal expected)",
        detail + "\nMost lambdas are in the middle range.\n"
        "AgriLambdaNet learns best from a clear bimodal signal.\n"
        "Consider adding more extreme cases (Type A high-lambda, Type C low-lambda).")
else:
    log(PASS, "Lambda distribution (bimodal expected)", detail)

# =============================================================================
# CHECK 9 — Type C (polysemy) lambda must be low (<= 0.35)
# =============================================================================
c_lam = df[df["pair_type"] == "C"]["lambda_val"]
c_high = c_lam[c_lam > 0.35]
detail = (
    f"  Type C lambda: min={c_lam.min():.3f}  max={c_lam.max():.3f}  "
    f"mean={c_lam.mean():.3f}\n"
    f"  Rows with lambda > 0.35: {len(c_high)}"
)
if len(c_high) > 0:
    log(WARN, "Type C polysemy lambda <= 0.35",
        detail + "\nHigh lambda on polysemy pairs tells the model to trust the "
        "name — which is the OPPOSITE of what polysemy needs.\n"
        "These rows will push the model in the wrong direction.")
else:
    log(PASS, "Type C polysemy lambda <= 0.35", detail)

# =============================================================================
# CHECK 10 — Duplicate pairs
# =============================================================================
dupes = df.duplicated(subset=["name_a","name_b"], keep=False)
n_dupes = dupes.sum()
if n_dupes > 0:
    log(WARN, "Duplicate pairs (name_a, name_b)",
        f"{n_dupes} rows share the same (name_a, name_b).\n"
        f"Exact duplicates cause data leakage if they appear in both train/val.\n"
        f"Run: df.drop_duplicates(subset=['name_a','name_b'], keep='first')")
else:
    log(PASS, "Duplicate pairs (name_a, name_b)", "")

# =============================================================================
# CHECK 11 — Empty or very short context strings
# =============================================================================
for col in ["context_a","context_b"]:
    if col not in df.columns: continue
    short = df[df[col].str.len() < 10]
    if len(short):
        log(WARN, f"Short context in {col} (<10 chars)",
            f"{len(short)} rows have near-empty context.\n"
            f"Sentence encoders get no signal from empty strings.\n"
            f"Replace with 'No context available' or drop these rows.")
    else:
        log(PASS, f"Short context in {col} (<10 chars)", "")

# =============================================================================
# CHECK 12 — Name similarity score present and in range
# =============================================================================
if "name_sim_score" in df.columns:
    null_sim = df["name_sim_score"].isnull().sum()
    bad_sim  = df[(df["name_sim_score"] < 0) | (df["name_sim_score"] > 1)].shape[0]
    if null_sim > 0:
        log(WARN, "name_sim_score: no nulls",
            f"{null_sim} rows missing name_sim_score.\n"
            f"If used as a feature, fill with 0.0 or compute it.")
    elif bad_sim > 0:
        log(FAIL, "name_sim_score in [0,1]",
            f"{bad_sim} values outside [0,1]. Normalize before use.")
    else:
        log(PASS, "name_sim_score: present and in [0,1]",
            f"min={df['name_sim_score'].min():.3f}  "
            f"max={df['name_sim_score'].max():.3f}")

# =============================================================================
# CHECK 13 — exclude_from_lambda flag
# =============================================================================
if "exclude_from_lambda" in df.columns:
    n_excl = int(df["exclude_from_lambda"].sum())
    n_usable = len(df) - n_excl
    detail = (
        f"  Excluded (both-poor context): {n_excl}\n"
        f"  Usable for lambda training:   {n_usable}\n"
        f"  Make sure your loss function respects this flag."
    )
    if n_usable < 500:
        log(WARN, "exclude_from_lambda: enough lambda-training rows",
            detail + "\nFewer than 500 pairs for lambda training. Model may underfit.")
    else:
        log(PASS, "exclude_from_lambda: enough lambda-training rows", detail)

# =============================================================================
# CHECK 14 — Suspicious outlier: high lambda on Type D (random negatives)
# =============================================================================
d_high = df[(df["pair_type"] == "D") & (df["lambda_val"] > 0.85)]
if len(d_high) > 10:
    log(WARN, "Type D outliers: lambda > 0.85",
        f"{len(d_high)} random-negative pairs have very high lambda.\n"
        f"High lambda on Type D means 'name alone distinguishes them' which is fine,\n"
        f"but double-check a few rows — some may have been mis-typed as D.")
else:
    log(PASS, "Type D outliers: lambda > 0.85",
        f"{len(d_high)} rows — acceptable.")

# =============================================================================
# CHECK 15 — Train/val split feasibility (no entity leakage)
# =============================================================================
# Check that each disease name appears in enough pairs to enable a clean split
all_names = pd.concat([df["name_a"], df["name_b"]])
name_freq = all_names.value_counts()
singletons = (name_freq == 1).sum()
detail = (
    f"  Unique entity names: {len(name_freq)}\n"
    f"  Names appearing only once: {singletons} "
    f"({singletons/len(name_freq)*100:.1f}%)"
)
if singletons / len(name_freq) > 0.5:
    log(WARN, "Train/val split: entity leakage risk",
        detail + "\nMore than 50% of entity names appear in only one pair.\n"
        "A random 80/20 split will likely put the same entity in both train and val.\n"
        "Use a grouped split: group_by=canonical_id, then split groups.")
else:
    log(PASS, "Train/val split: entity leakage risk", detail)

# =============================================================================
# CHECK 16 — lambda_source column distribution
# =============================================================================
if "lambda_source" in df.columns:
    src = df["lambda_source"].value_counts()
    detail = "\n".join(f"  {k}: {v}" for k,v in src.items())
    log(PASS if "corrected_polysemy" in src.index else WARN,
        "lambda_source: fix tracking column present", detail)

# =============================================================================
# SUMMARY
# =============================================================================
n_pass = sum(1 for s,_,_ in results if s == PASS)
n_warn = sum(1 for s,_,_ in results if s == WARN)
n_fail = sum(1 for s,_,_ in results if s == FAIL)

print()
print("=" * 65)
print("  VALIDATION SUMMARY")
print("=" * 65)
print(f"  PASS : {n_pass}")
print(f"  WARN : {n_warn}")
print(f"  FAIL : {n_fail}")
print()

if n_fail == 0 and n_warn == 0:
    print("  Dataset is CLEAN. Ready for training.")
elif n_fail == 0:
    print("  Dataset is USABLE with warnings. Review WARNs before publishing.")
else:
    print("  Dataset has FAILURES. Fix them before training.")

# Print all WARNs and FAILs again for quick reference
if n_warn + n_fail > 0:
    print()
    print("  Issues to address:")
    for status, name, detail in results:
        if status in (WARN, FAIL):
            tag = "FAIL" if status == FAIL else "WARN"
            print(f"    [{tag}] {name}")
print()
