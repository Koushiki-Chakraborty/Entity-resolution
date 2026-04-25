# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
  EXPERT DATA VALIDATION REPORT - training_ready_final.csv
  AgriΛNet Entity Resolution Project
  Validation standard: Research-grade ML dataset review
=============================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
import textwrap

# ── Config ────────────────────────────────────────────────────────────────
CSV_PATH = "training_ready_final.csv"
EXPECTED_COLS = [
    "name_a", "context_a", "name_b", "context_b",
    "canonical_id_a", "canonical_id_b",
    "source_url_a", "source_url_b",
    "match", "llm_match", "lambda_val"
]
MIN_CONTEXT_LEN   = 20   # chars — below this is suspiciously short
LAMBDA_MATCH_LO   = 0.5  # lambda >= this for match=1 is "high confidence"
LAMBDA_NOMATCH_HI = 0.5  # lambda <  this for match=0 is "low confidence"

SEPARATOR = "=" * 72

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def err(msg):  print(f"  [ERR]  {msg}")
def info(msg): print(f"  [INFO] {msg}")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════
section("0. LOADING DATASET")
try:
    df = pd.read_csv(CSV_PATH)
    ok(f"Loaded successfully  →  {len(df):,} rows × {len(df.columns)} columns")
except Exception as e:
    err(f"Could not load file: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA CHECK
# ═══════════════════════════════════════════════════════════════════════════
section("1. SCHEMA VALIDATION")

missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
extra_cols   = [c for c in df.columns if c not in EXPECTED_COLS]

if not missing_cols:
    ok("All 11 required columns present")
else:
    err(f"Missing columns: {missing_cols}")

if extra_cols:
    warn(f"Unexpected extra columns: {extra_cols}")
else:
    ok("No unexpected extra columns")

# ═══════════════════════════════════════════════════════════════════════════
# 2. MISSING / NULL VALUES
# ═══════════════════════════════════════════════════════════════════════════
section("2. MISSING VALUE AUDIT")

null_counts = df.isnull().sum()
total_cells = len(df)

for col in EXPECTED_COLS:
    n = null_counts.get(col, 0)
    pct = 100 * n / total_cells
    if n == 0:
        ok(f"{col:<22} → 0 nulls")
    else:
        err(f"{col:<22} → {n} nulls  ({pct:.2f}%)")

# ── Detect "nan" strings (common pipeline artifact) ────────────────────────
nan_str_cols = []
for col in ["context_a", "context_b", "name_a", "name_b"]:
    count = (df[col].astype(str).str.strip().str.lower() == "nan").sum()
    if count > 0:
        nan_str_cols.append((col, count))
        warn(f'String "nan" found in {col}: {count} rows')

if not nan_str_cols:
    ok('No "nan" string artifacts in text columns')

# ═══════════════════════════════════════════════════════════════════════════
# 3. LABEL INTEGRITY  (match / llm_match / lambda_val)
# ═══════════════════════════════════════════════════════════════════════════
section("3. LABEL INTEGRITY")

# -- match column ----------------------------------------------------------
valid_match = df["match"].isin([0, 1])
bad_match   = (~valid_match).sum()
if bad_match == 0:
    ok("match column: only 0/1 values")
else:
    err(f"match column: {bad_match} rows with values outside {{0,1}}")
    print(df[~valid_match]["match"].value_counts().to_string())

pos  = (df["match"] == 1).sum()
neg  = (df["match"] == 0).sum()
ratio = neg / pos if pos > 0 else float("inf")
info(f"match=1 (positives): {pos:,}  |  match=0 (negatives): {neg:,}  |  ratio 1:{ratio:.1f}")

if ratio < 3:
    warn("Negative-to-positive ratio < 1:3 — model may not learn to reject well")
elif ratio > 10:
    warn("Negative-to-positive ratio > 1:10 — heavy class imbalance, consider rebalancing")
else:
    ok(f"Class ratio 1:{ratio:.1f} is within acceptable range [1:3 – 1:10]")

# -- llm_match column ------------------------------------------------------
# Normalise: may be stored as TRUE/FALSE string or bool
df["llm_match_norm"] = df["llm_match"].astype(str).str.strip().str.upper()
valid_llm = df["llm_match_norm"].isin(["TRUE", "FALSE", "1", "0"])
bad_llm   = (~valid_llm).sum()
if bad_llm == 0:
    ok("llm_match column: valid TRUE/FALSE values throughout")
else:
    err(f"llm_match column: {bad_llm} rows with unexpected values")
    print(df[~valid_llm]["llm_match"].value_counts().head(10).to_string())

# -- lambda_val column -----------------------------------------------------
df["lambda_val"] = pd.to_numeric(df["lambda_val"], errors="coerce")
bad_lambda = df["lambda_val"].isnull().sum()
if bad_lambda > 0:
    err(f"lambda_val: {bad_lambda} non-numeric values (coerced to NaN)")
else:
    ok("lambda_val: all values numeric")

out_of_range = ((df["lambda_val"] < 0) | (df["lambda_val"] > 1)).sum()
if out_of_range == 0:
    ok("lambda_val: all values in [0, 1]")
else:
    err(f"lambda_val: {out_of_range} values outside [0, 1]")

lmin  = df["lambda_val"].min()
lmax  = df["lambda_val"].max()
lmean = df["lambda_val"].mean()
lstd  = df["lambda_val"].std()
info(f"lambda_val stats — min: {lmin:.4f}  max: {lmax:.4f}  mean: {lmean:.4f}  std: {lstd:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LABEL CONSISTENCY CROSS-CHECKS
# ═══════════════════════════════════════════════════════════════════════════
section("4. LABEL CONSISTENCY CROSS-CHECKS")

# 4a. match vs canonical_id agreement
df["id_agree"] = df["canonical_id_a"] == df["canonical_id_b"]

match1_id_disagree = ((df["match"] == 1) & (~df["id_agree"])).sum()
match0_id_agree    = ((df["match"] == 0) & (df["id_agree"])).sum()

if match1_id_disagree == 0:
    ok("All match=1 rows have matching canonical IDs")
else:
    warn(f"{match1_id_disagree} rows: match=1 but canonical_id_a ≠ canonical_id_b  (label noise or cross-disease synonym?)")

if match0_id_agree == 0:
    ok("No match=0 rows share the same canonical ID")
else:
    err(f"{match0_id_agree} rows: match=0 but canonical_id_a == canonical_id_b  (FALSE NEGATIVES — critical error!)")
    samples = df[(df["match"] == 0) & (df["id_agree"])][["name_a","name_b","canonical_id_a","match","llm_match"]].head(5)
    print(samples.to_string(index=False))

# 4b. match vs llm_match disagreement
df["llm_bool"] = df["llm_match_norm"].map({"TRUE": True, "FALSE": False, "1": True, "0": False})
df["match_bool"] = df["match"].astype(bool)
df["label_conflict"] = df["match_bool"] != df["llm_bool"]

conflict_count = df["label_conflict"].sum()
conflict_pct   = 100 * conflict_count / len(df)
if conflict_pct <= 5:
    ok(f"match vs llm_match disagreement: {conflict_count} rows ({conflict_pct:.1f}%)  [acceptable ≤5%]")
elif conflict_pct <= 15:
    warn(f"match vs llm_match disagreement: {conflict_count} rows ({conflict_pct:.1f}%)  [moderate — investigate]")
else:
    err(f"match vs llm_match disagreement: {conflict_count} rows ({conflict_pct:.1f}%)  [high — data quality issue]")

# 4c. Lambda vs match semantic alignment
pos_lambda_mean = df[df["match"] == 1]["lambda_val"].mean()
neg_lambda_mean = df[df["match"] == 0]["lambda_val"].mean()
info(f"Mean lambda for match=1: {pos_lambda_mean:.4f}")
info(f"Mean lambda for match=0: {neg_lambda_mean:.4f}")

if pos_lambda_mean > neg_lambda_mean:
    ok("Lambda signal semantically aligned: positives have higher mean lambda than negatives")
else:
    err("Lambda signal INVERTED: negatives have higher lambda than positives — check lambda construction!")

# Suspicious: match=0 but high lambda
suspicious_neg = ((df["match"] == 0) & (df["lambda_val"] >= LAMBDA_NOMATCH_HI)).sum()
suspicious_pos = ((df["match"] == 1) & (df["lambda_val"] < 0.3)).sum()
if suspicious_neg > 0:
    warn(f"{suspicious_neg} match=0 rows have lambda ≥ {LAMBDA_NOMATCH_HI}  (potential noisy negatives)")
if suspicious_pos > 0:
    warn(f"{suspicious_pos} match=1 rows have lambda < 0.3  (low-confidence positives)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. CONTEXT QUALITY
# ═══════════════════════════════════════════════════════════════════════════
section("5. CONTEXT / TEXT QUALITY")

for col in ["context_a", "context_b"]:
    lengths = df[col].astype(str).str.len()
    short   = (lengths < MIN_CONTEXT_LEN).sum()
    truncated = df[col].astype(str).str.endswith(("...", "…", " T")).sum()
    info(f"{col} — avg len: {lengths.mean():.0f}  min: {lengths.min()}  max: {lengths.max()}")
    if short > 0:
        warn(f"{col}: {short} entries shorter than {MIN_CONTEXT_LEN} chars (low signal for encoder)")
    else:
        ok(f"{col}: all entries ≥ {MIN_CONTEXT_LEN} chars")
    if truncated > 0:
        warn(f"{col}: ~{truncated} entries may be truncated (end with '...' or similar)")

# Check for boilerplate / placeholder context
boilerplate_hits = 0
boilerplate_patterns = [
    "non-pathogenic aquatic species",
    "Marine invertebrate",
    "list gives some examples",
    "vast number of freshwater",
    "archaeological site",
]
for pat in boilerplate_patterns:
    for col in ["context_a", "context_b"]:
        n = df[col].astype(str).str.contains(pat, case=False, na=False).sum()
        if n > 0:
            boilerplate_hits += n
            warn(f'Off-domain boilerplate in {col}: "{pat[:45]}" — {n} rows')

if boilerplate_hits == 0:
    ok("No known boilerplate/off-domain context patterns detected")

# ═══════════════════════════════════════════════════════════════════════════
# 6. DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
section("6. DUPLICATE PAIR DETECTION")

pair_key = df.apply(
    lambda r: tuple(sorted([str(r["name_a"]).lower().strip(),
                             str(r["name_b"]).lower().strip()])),
    axis=1
)
dup_mask  = pair_key.duplicated(keep=False)
dup_count = dup_mask.sum()

if dup_count == 0:
    ok("No duplicate (name_a, name_b) pairs found")
else:
    warn(f"{dup_count} rows are part of duplicate (name_a, name_b) pairs")
    dup_groups = pair_key[dup_mask].value_counts().head(5)
    for pair, cnt in dup_groups.items():
        print(f"    {pair}  ×{cnt}")

# Exact-duplicate rows
exact_dups = df.duplicated().sum()
if exact_dups == 0:
    ok("No fully identical duplicate rows")
else:
    err(f"{exact_dups} fully identical duplicate rows detected — remove before training!")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SELF-PAIRS (name_a == name_b)
# ═══════════════════════════════════════════════════════════════════════════
section("7. SELF-PAIR CHECK")

self_pairs = (df["name_a"].str.lower().str.strip() == df["name_b"].str.lower().str.strip()).sum()
if self_pairs == 0:
    ok("No self-pairs (name_a == name_b) detected")
else:
    warn(f"{self_pairs} self-pairs detected — these should always be match=1")
    self_mask = df["name_a"].str.lower().str.strip() == df["name_b"].str.lower().str.strip()
    wrong_self = ((self_mask) & (df["match"] == 0)).sum()
    if wrong_self > 0:
        err(f"  └─ {wrong_self} self-pairs are incorrectly labeled match=0!")

# ═══════════════════════════════════════════════════════════════════════════
# 8. CANONICAL ID COVERAGE
# ═══════════════════════════════════════════════════════════════════════════
section("8. CANONICAL ID COVERAGE")

all_ids = pd.concat([df["canonical_id_a"], df["canonical_id_b"]])
unique_ids = all_ids.nunique()
agrovoc_ids = all_ids.str.startswith("agrovoc_").sum()
kaggle_ids  = (~all_ids.str.startswith("agrovoc_")).sum()

info(f"Unique canonical IDs in dataset: {unique_ids}")
info(f"AGROVOC-sourced IDs: {agrovoc_ids}  |  PlantVillage/other IDs: {kaggle_ids}")

missing_ids = (df["canonical_id_a"].isnull() | df["canonical_id_b"].isnull()).sum()
if missing_ids == 0:
    ok("All rows have canonical IDs for both entities")
else:
    err(f"{missing_ids} rows missing at least one canonical ID")

# ═══════════════════════════════════════════════════════════════════════════
# 9. LAMBDA DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
section("9. LAMBDA DISTRIBUTION ANALYSIS (for dual-loss training)")

bins   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5",
          "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]
df["lambda_bin"] = pd.cut(df["lambda_val"], bins=bins, labels=labels, right=False)
dist = df["lambda_bin"].value_counts().sort_index()

print()
for bin_label, count in dist.items():
    bar = "█" * int(count / max(dist) * 40)
    print(f"  {bin_label} | {bar:<40} {count:>5}")

extreme_low  = (df["lambda_val"] < 0.05).sum()
extreme_high = (df["lambda_val"] > 0.95).sum()
mid_range    = ((df["lambda_val"] >= 0.3) & (df["lambda_val"] <= 0.7)).sum()
info(f"\nExtreme low  (< 0.05): {extreme_low} rows")
info(f"Extreme high (> 0.95): {extreme_high} rows")
info(f"Mid-range  [0.3-0.7]: {mid_range} rows")

if mid_range / len(df) > 0.3:
    ok("Good mid-range lambda coverage — model will learn gradual confidence transitions")
else:
    warn("Low mid-range lambda coverage — bimodal distribution may cause sharp decision boundaries")

# ═══════════════════════════════════════════════════════════════════════════
# 10. SOURCE URL VALIDITY
# ═══════════════════════════════════════════════════════════════════════════
section("10. SOURCE URL AUDIT")

for col in ["source_url_a", "source_url_b"]:
    total_urls = df[col].notna().sum()
    http_ok    = df[col].astype(str).str.startswith("http").sum()
    kaggle     = df[col].astype(str).str.contains("kaggle.com", na=False).sum()
    agrovoc    = df[col].astype(str).str.contains("aims.fao.org", na=False).sum()
    wikipedia  = df[col].astype(str).str.contains("wikipedia.org", na=False).sum()
    info(f"{col}: total={total_urls}  kaggle={kaggle}  agrovoc={agrovoc}  wikipedia={wikipedia}  valid_http={http_ok}")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
section("FINAL VALIDATION SUMMARY")

pos_total  = (df["match"] == 1).sum()
neg_total  = (df["match"] == 0).sum()
agree_pct  = 100 * (1 - conflict_pct / 100)
lam_pos    = df[df["match"] == 1]["lambda_val"].mean()
lam_neg    = df[df["match"] == 0]["lambda_val"].mean()

print(f"""
  Dataset:          training_ready_final.csv
  Total pairs:      {len(df):,}
  Positives:        {pos_total:,}
  Negatives:        {neg_total:,}
  Class ratio:      1 : {ratio:.1f}
  GT/LLM agreement: {agree_pct:.1f}%
  Lambda (pos avg): {lam_pos:.4f}
  Lambda (neg avg): {lam_neg:.4f}
  Unique can. IDs:  {unique_ids}
  Exact duplicates: {exact_dups}
  False negatives:  {match0_id_agree}
""")

issues_found = (bad_match + bad_llm + missing_ids + exact_dups +
                match0_id_agree + len(nan_str_cols))

if issues_found == 0:
    print("  [PASS] VERDICT: Dataset passed all critical checks. Ready for training.")
elif issues_found <= 3:
    print("  [WARN] VERDICT: Minor issues found -- review warnings before training.")
else:
    print("  [FAIL] VERDICT: Multiple issues found -- fix errors before training.")

print(f"\n{SEPARATOR}\n")
