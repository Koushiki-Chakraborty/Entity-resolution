# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
  EXPERT DATA VALIDATION REPORT - dataset_v2.csv
  AgriΛNet Entity Resolution Project
  Validation standard: Research-grade ML dataset review
=============================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
import textwrap
import json
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────
CSV_PATH = "dataset_v2.csv"
EXPECTED_COLS = [
    "name_a", "context_a", "name_b", "context_b",
    "canonical_id_a", "canonical_id_b",
    "source_url_a", "source_url_b",
    "match", "llm_match", "lambda_val",
    "context_quality_a", "context_quality_b",
    "pair_type", "name_sim_score", "source_a", "source_b"
]
MIN_CONTEXT_LEN   = 20   # chars — below this is suspiciously short
LAMBDA_MATCH_LO   = 0.5  # lambda >= this for match=1 is "high confidence"
LAMBDA_NOMATCH_HI = 0.5  # lambda <  this for match=0 is "low confidence"

SEPARATOR = "=" * 80

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg):   print(f"  [✓]   {msg}")
def warn(msg): print(f"  [⚠]   {msg}")
def err(msg):  print(f"  [✗]   {msg}")
def info(msg): print(f"  [ℹ]   {msg}")

# Store validation results
validation_results = {
    "timestamp": datetime.now().isoformat(),
    "file": CSV_PATH,
    "sections": {}
}

# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════
section("0. LOADING DATASET")
try:
    df = pd.read_csv(CSV_PATH)
    ok(f"Loaded successfully  →  {len(df):,} rows × {len(df.columns)} columns")
    validation_results["sections"]["loading"] = {
        "status": "success",
        "rows": len(df),
        "columns": len(df.columns)
    }
except Exception as e:
    err(f"Could not load file: {e}")
    validation_results["sections"]["loading"] = {
        "status": "failed",
        "error": str(e)
    }
    raise

# ═══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA CHECK
# ═══════════════════════════════════════════════════════════════════════════
section("1. SCHEMA VALIDATION")

missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
extra_cols   = [c for c in df.columns if c not in EXPECTED_COLS]

schema_ok = True
if not missing_cols:
    ok(f"All {len(EXPECTED_COLS)} required columns present")
else:
    err(f"Missing columns: {missing_cols}")
    schema_ok = False

if extra_cols:
    warn(f"Unexpected extra columns: {extra_cols}")
else:
    ok("No unexpected extra columns")

validation_results["sections"]["schema"] = {
    "status": "ok" if schema_ok else "error",
    "missing_cols": missing_cols,
    "extra_cols": extra_cols,
    "total_expected": len(EXPECTED_COLS),
    "total_found": len(df.columns)
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. MISSING / NULL VALUES
# ═══════════════════════════════════════════════════════════════════════════
section("2. MISSING VALUE AUDIT")

null_counts = df.isnull().sum()
total_cells = len(df)
null_issues = []

for col in EXPECTED_COLS:
    if col in df.columns:
        n = null_counts.get(col, 0)
        pct = 100 * n / total_cells
        if n == 0:
            ok(f"{col:<25} → 0 nulls")
        else:
            err(f"{col:<25} → {n} nulls  ({pct:.2f}%)")
            null_issues.append((col, n, pct))

# ── Detect "nan" strings (common pipeline artifact) ────────────────────────
nan_str_cols = []
for col in ["context_a", "context_b", "name_a", "name_b"]:
    if col in df.columns:
        count = (df[col].astype(str).str.strip().str.lower() == "nan").sum()
        if count > 0:
            nan_str_cols.append((col, count))
            warn(f'String "nan" found in {col}: {count} rows')

if not nan_str_cols:
    ok('No "nan" string artifacts in text columns')

validation_results["sections"]["missing_values"] = {
    "null_issues": null_issues,
    "nan_string_issues": nan_str_cols
}

# ═══════════════════════════════════════════════════════════════════════════
# 3. LABEL INTEGRITY  (match / llm_match / lambda_val)
# ═══════════════════════════════════════════════════════════════════════════
section("3. LABEL INTEGRITY")

# -- match column ----------------------------------------------------------
if "match" in df.columns:
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

    validation_results["sections"]["label_integrity"] = {
        "match_col": {
            "valid_values": bad_match == 0,
            "bad_count": bad_match,
            "positives": int(pos),
            "negatives": int(neg),
            "ratio": float(ratio) if ratio != float("inf") else "inf"
        }
    }

# -- llm_match column ------------------------------------------------------
if "llm_match" in df.columns:
    df["llm_match_norm"] = df["llm_match"].astype(str).str.strip().str.upper()
    valid_llm = df["llm_match_norm"].isin(["TRUE", "FALSE", "1", "0"])
    bad_llm   = (~valid_llm).sum()
    if bad_llm == 0:
        ok("llm_match column: valid TRUE/FALSE values throughout")
    else:
        err(f"llm_match column: {bad_llm} rows with unexpected values")
        print(df[~valid_llm]["llm_match"].value_counts().head(10).to_string())

# -- lambda_val column -----------------------------------------------------
if "lambda_val" in df.columns:
    df["lambda_val_numeric"] = pd.to_numeric(df["lambda_val"], errors="coerce")
    bad_lambda = df["lambda_val_numeric"].isnull().sum()
    if bad_lambda > 0:
        err(f"lambda_val: {bad_lambda} non-numeric values (coerced to NaN)")
    else:
        ok("lambda_val: all values numeric")

    out_of_range = ((df["lambda_val_numeric"] < 0) | (df["lambda_val_numeric"] > 1)).sum()
    if out_of_range == 0:
        ok("lambda_val: all values in [0, 1]")
    else:
        err(f"lambda_val: {out_of_range} values outside [0, 1]")

    lmin  = df["lambda_val_numeric"].min()
    lmax  = df["lambda_val_numeric"].max()
    lmean = df["lambda_val_numeric"].mean()
    info(f"lambda_val distribution: min={lmin:.3f}, max={lmax:.3f}, mean={lmean:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTEXT QUALITY
# ═══════════════════════════════════════════════════════════════════════════
section("4. CONTEXT QUALITY ANALYSIS")

short_ctx_a = 0
short_ctx_b = 0
empty_ctx_a = 0
empty_ctx_b = 0

if "context_a" in df.columns:
    for idx, val in enumerate(df["context_a"]):
        if pd.isna(val):
            empty_ctx_a += 1
        elif len(str(val).strip()) < MIN_CONTEXT_LEN:
            short_ctx_a += 1

if "context_b" in df.columns:
    for idx, val in enumerate(df["context_b"]):
        if pd.isna(val):
            empty_ctx_b += 1
        elif len(str(val).strip()) < MIN_CONTEXT_LEN:
            short_ctx_b += 1

if short_ctx_a > 0:
    warn(f"context_a: {short_ctx_a} rows with context < {MIN_CONTEXT_LEN} chars")
else:
    ok(f"context_a: all contexts >= {MIN_CONTEXT_LEN} chars")

if short_ctx_b > 0:
    warn(f"context_b: {short_ctx_b} rows with context < {MIN_CONTEXT_LEN} chars")
else:
    ok(f"context_b: all contexts >= {MIN_CONTEXT_LEN} chars")

# Check context_quality columns
if "context_quality_a" in df.columns:
    quality_vals_a = df["context_quality_a"].value_counts()
    info(f"context_quality_a values: {dict(quality_vals_a)}")

if "context_quality_b" in df.columns:
    quality_vals_b = df["context_quality_b"].value_counts()
    info(f"context_quality_b values: {dict(quality_vals_b)}")

validation_results["sections"]["context_quality"] = {
    "short_context_a": short_ctx_a,
    "short_context_b": short_ctx_b,
    "empty_context_a": empty_ctx_a,
    "empty_context_b": empty_ctx_b
}

# ═══════════════════════════════════════════════════════════════════════════
# 5. PAIR TYPE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
section("5. PAIR TYPE ANALYSIS")

if "pair_type" in df.columns:
    pair_counts = df["pair_type"].value_counts()
    info(f"Pair type distribution:")
    for ptype, count in pair_counts.items():
        pct = 100 * count / len(df)
        print(f"    {ptype:<30} {count:>6,}  ({pct:>5.1f}%)")
    
    validation_results["sections"]["pair_type"] = pair_counts.to_dict()

# ═══════════════════════════════════════════════════════════════════════════
# 6. SOURCE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
section("6. SOURCE DISTRIBUTION")

if "source_a" in df.columns and "source_b" in df.columns:
    source_combos = (df["source_a"].astype(str) + " ↔ " + df["source_b"].astype(str)).value_counts()
    info(f"Top source combinations:")
    for combo, count in source_combos.head(10).items():
        pct = 100 * count / len(df)
        print(f"    {combo:<50} {count:>6,}  ({pct:>5.1f}%)")
    
    validation_results["sections"]["source_distribution"] = source_combos.head(10).to_dict()

# ═══════════════════════════════════════════════════════════════════════════
# 7. NAME SIMILARITY SCORE
# ═══════════════════════════════════════════════════════════════════════════
section("7. NAME SIMILARITY SCORE")

if "name_sim_score" in df.columns:
    df["name_sim_score_numeric"] = pd.to_numeric(df["name_sim_score"], errors="coerce")
    nsim_min = df["name_sim_score_numeric"].min()
    nsim_max = df["name_sim_score_numeric"].max()
    nsim_mean = df["name_sim_score_numeric"].mean()
    nsim_median = df["name_sim_score_numeric"].median()
    
    info(f"name_sim_score statistics:")
    print(f"    min: {nsim_min:.3f}  |  max: {nsim_max:.3f}  |  mean: {nsim_mean:.3f}  |  median: {nsim_median:.3f}")
    
    validation_results["sections"]["name_similarity"] = {
        "min": float(nsim_min),
        "max": float(nsim_max),
        "mean": float(nsim_mean),
        "median": float(nsim_median)
    }

# ═══════════════════════════════════════════════════════════════════════════
# 8. DUPLICATE CHECK
# ═══════════════════════════════════════════════════════════════════════════
section("8. DUPLICATE DETECTION")

# Check for exact row duplicates
duplicates = df.duplicated().sum()
if duplicates == 0:
    ok("No exact row duplicates found")
else:
    warn(f"Found {duplicates} exact row duplicates")

# Check for potential semantic duplicates (same names, different order)
if "name_a" in df.columns and "name_b" in df.columns:
    df["pair_set"] = df.apply(lambda row: tuple(sorted([str(row["name_a"]), str(row["name_b"])])), axis=1)
    semantic_dups = df.duplicated(subset=["pair_set"]).sum()
    if semantic_dups > 0:
        warn(f"Found {semantic_dups} potential semantic duplicates (same pair, possibly reversed)")
    else:
        ok("No semantic duplicates found")

validation_results["sections"]["duplicates"] = {
    "exact_duplicates": int(duplicates),
    "semantic_duplicates": int(semantic_dups) if "name_a" in df.columns else 0
}

# ═══════════════════════════════════════════════════════════════════════════
# 9. CONFIDENCE ANALYSIS (lambda_val vs match labels)
# ═══════════════════════════════════════════════════════════════════════════
section("9. CONFIDENCE vs LABEL AGREEMENT")

if "match" in df.columns and "lambda_val_numeric" in df.columns:
    # Matches with high confidence
    high_conf_matches = (
        (df["match"] == 1) & 
        (df["lambda_val_numeric"] >= LAMBDA_MATCH_LO)
    ).sum()
    
    # Matches with low confidence
    low_conf_matches = (
        (df["match"] == 1) & 
        (df["lambda_val_numeric"] < LAMBDA_MATCH_LO)
    ).sum()
    
    # Non-matches with high confidence
    high_conf_nonmatches = (
        (df["match"] == 0) & 
        (df["lambda_val_numeric"] < LAMBDA_NOMATCH_HI)
    ).sum()
    
    # Non-matches with low confidence
    low_conf_nonmatches = (
        (df["match"] == 0) & 
        (df["lambda_val_numeric"] >= LAMBDA_NOMATCH_HI)
    ).sum()
    
    info(f"Matches (match=1) with high confidence (λ≥{LAMBDA_MATCH_LO}): {high_conf_matches:,}")
    info(f"Matches (match=1) with low confidence (λ<{LAMBDA_MATCH_LO}):  {low_conf_matches:,}")
    info(f"Non-matches (match=0) with high confidence (λ<{LAMBDA_NOMATCH_HI}): {high_conf_nonmatches:,}")
    info(f"Non-matches (match=0) with low confidence (λ≥{LAMBDA_NOMATCH_HI}):  {low_conf_nonmatches:,}")
    
    if low_conf_matches > 0 or low_conf_nonmatches > 0:
        warn("Found label-confidence misalignment; review required")
    else:
        ok("Strong agreement between labels and confidence scores")
    
    validation_results["sections"]["confidence_analysis"] = {
        "high_conf_matches": int(high_conf_matches),
        "low_conf_matches": int(low_conf_matches),
        "high_conf_nonmatches": int(high_conf_nonmatches),
        "low_conf_nonmatches": int(low_conf_nonmatches)
    }

# ═══════════════════════════════════════════════════════════════════════════
# 10. SUMMARY & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
section("10. SUMMARY & RECOMMENDATIONS")

print(f"\n  📊 Dataset Size: {len(df):,} rows × {len(df.columns)} columns")
print(f"\n  ✓ Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Collect issues
total_issues = len(null_issues) + len(nan_str_cols) + len([x for x in [short_ctx_a, short_ctx_b, duplicates, semantic_dups] if x > 0])

if total_issues == 0:
    ok("✨ Dataset PASSED validation with no critical issues!")
    print("\n  📝 Next steps:")
    print("     1. Ready for model training")
    print("     2. Recommended: Use for baseline model development")
    print("     3. Consider data augmentation for better coverage")
else:
    warn(f"⚠  Dataset has {total_issues} issue(s) that should be reviewed")
    print("\n  📝 Recommended actions:")
    if null_issues:
        print("     • Address null values in:", [x[0] for x in null_issues])
    if short_ctx_a > 0 or short_ctx_b > 0:
        print("     • Enrich short contexts (< 20 chars)")
    if duplicates > 0 or semantic_dups > 0:
        print("     • Remove duplicate pairs")

# Save JSON report
json_report_path = "dataset_v2_validation_report.json"
with open(json_report_path, 'w') as f:
    json.dump(validation_results, f, indent=2)
print(f"\n  💾 Detailed report saved to: {json_report_path}\n")

print(SEPARATOR)
print("  END OF VALIDATION REPORT")
print(SEPARATOR)
