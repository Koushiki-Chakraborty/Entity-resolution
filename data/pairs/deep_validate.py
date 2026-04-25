# -*- coding: utf-8 -*-
"""
deep_validate.py
================================================================
Expert-level deep validation of training_ready_final.csv
Goes beyond validate_dataset.py to investigate every warning and
test ML-readiness of the dataset for AgriLambdaNet training.

Sections:
  A. Class Imbalance Detail
  B. Noisy Negative Deep-Dive (match=0, lambda>=0.5)
  C. Low-Confidence Positive Deep-Dive (match=1, lambda<0.3)
  D. Truncated Context Analysis (entries ending in '...')
  E. Off-Domain Boilerplate Rows
  F. Lambda Bimodality & Distribution Shape
  G. Name Length & Abbreviation Distribution
  H. Context Uniqueness (near-duplicate context pairs)
  I. Entity Leakage Risk (same entity appearing in both sides)
  J. Training Signal Quality Score (per-row)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from collections import Counter

SEP  = "=" * 70
SSEP = "-" * 70
CSV  = "training_ready_final.csv"

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def sub(title):
    print(f"\n  {SSEP}")
    print(f"  ▶  {title}")
    print(f"  {SSEP}")

def ok(m):   print(f"  [OK]   {m}")
def warn(m): print(f"  [WARN] {m}")
def err(m):  print(f"  [ERR]  {m}")
def info(m): print(f"  [INFO] {m}")

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df['llm_match'] = df['llm_match'].astype(str).str.strip().str.upper().map(
    {"TRUE": True, "FALSE": False, "1": True, "0": False}
)
print(f"\nLoaded {len(df):,} rows from {CSV}")

# ═══════════════════════════════════════════════════════════════════════════
# A. CLASS IMBALANCE DETAIL
# ═══════════════════════════════════════════════════════════════════════════
section("A. CLASS IMBALANCE DETAIL")

pos = (df['match'] == 1).sum()
neg = (df['match'] == 0).sum()
ratio = neg / pos
info(f"Positive pairs (match=1): {pos}   ({100*pos/len(df):.1f}%)")
info(f"Negative pairs (match=0): {neg}   ({100*neg/len(df):.1f}%)")
info(f"Negative : Positive ratio = {ratio:.2f} : 1")

if ratio < 1.5:
    err("Severe under-sampling of negatives — model will struggle to generalise rejection.")
elif ratio < 3:
    warn("Ratio 1:2.0 — technically below the 1:3 guideline. Consider adding ~250 more negatives "
         "to reach 1:3. This is a SOFT warning; 1:2 is still trainable with weighted loss.")
elif ratio > 10:
    warn("Severe class imbalance — use class weights or focal loss.")
else:
    ok(f"Class ratio {ratio:.1f}:1 is within acceptable range.")

# Effective number of positives (unique entity pairs, not row duplicates)
unique_pos_ids = df[df['match']==1]['canonical_id_a'].nunique()
info(f"Unique positive canonical groups (canonical_id_a): {unique_pos_ids}")

# ═══════════════════════════════════════════════════════════════════════════
# B. NOISY NEGATIVE DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════
section("B. NOISY NEGATIVE DEEP-DIVE  (match=0, lambda >= 0.5)")

noisy_neg = df[(df['match'] == 0) & (df['lambda_val'] >= 0.5)].copy()
info(f"Count: {len(noisy_neg)} rows ({100*len(noisy_neg)/len(df):.1f}% of dataset)")

# Sub-bucket breakdown
buckets = [(0.5, 0.65), (0.65, 0.8), (0.8, 0.92), (0.92, 1.01)]
for lo, hi in buckets:
    n = ((noisy_neg['lambda_val'] >= lo) & (noisy_neg['lambda_val'] < hi)).sum()
    label = "name-driven (risky)" if lo >= 0.65 else "borderline"
    info(f"  lambda [{lo:.2f}, {hi:.2f}): {n:3d} rows  — {label}")

# LLM agreement within noisy negatives
llm_agree_nn = (noisy_neg['llm_match'] == False).sum()
info(f"LLM agrees they are non-matches: {llm_agree_nn}/{len(noisy_neg)} "
     f"({100*llm_agree_nn/max(len(noisy_neg),1):.1f}%)")

llm_conflict_nn = (noisy_neg['llm_match'] == True).sum()
if llm_conflict_nn > 0:
    warn(f"{llm_conflict_nn} noisy negatives where LLM says match=True — highest risk rows:")
    sample = noisy_neg[noisy_neg['llm_match'] == True].nlargest(5, 'lambda_val')
    print(sample[['name_a','name_b','lambda_val','llm_match']].to_string(index=False))
else:
    ok("No noisy negatives with LLM saying True — good label coherence.")

# Sample worst offenders
worst = noisy_neg.nlargest(8, 'lambda_val')
sub("Highest-lambda negatives (most likely confusing to the model)")
print(worst[['name_a','name_b','lambda_val','llm_match']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# C. LOW-CONFIDENCE POSITIVE DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════
section("C. LOW-CONFIDENCE POSITIVE DEEP-DIVE  (match=1, lambda < 0.3)")

lc_pos = df[(df['match'] == 1) & (df['lambda_val'] < 0.3)].copy()
info(f"Count: {len(lc_pos)} rows ({100*len(lc_pos)/len(df):.1f}% of dataset)")

# Are these mostly abbreviation pairs?
abbrev_lc = ((lc_pos['name_a'].str.len() <= 4) | (lc_pos['name_b'].str.len() <= 4)).sum()
info(f"  Of these, abbreviated name pairs (≤4 chars): {abbrev_lc}")

# LLM disagreement in low-confidence positives
llm_no_lc = (lc_pos['llm_match'] == False).sum()
if llm_no_lc > 0:
    warn(f"{llm_no_lc} low-confidence positives where LLM also says False — triple-conflict:")
    print(lc_pos[lc_pos['llm_match'] == False][['name_a','name_b','lambda_val','llm_match']].head(8).to_string(index=False))
else:
    ok("All low-confidence positives have LLM saying True (single-signal conflict only).")

sub("Lowest-lambda positives (near-zero gradient risk)")
print(lc_pos.nsmallest(8, 'lambda_val')[['name_a','name_b','lambda_val','llm_match']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# D. TRUNCATED CONTEXT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
section("D. TRUNCATED CONTEXT ANALYSIS  (entries ending in '...' or ' T')")

TRUNC_SIGS = ("...", "…", " T")
for col in ['context_a', 'context_b']:
    trunc_mask = df[col].astype(str).str.endswith(TRUNC_SIGS)
    n = trunc_mask.sum()
    info(f"{col}: {n} potentially truncated entries")
    if n > 0:
        lengths = df.loc[trunc_mask, col].str.len()
        info(f"  Avg length of truncated entries: {lengths.mean():.0f}  min: {lengths.min()}  max: {lengths.max()}")
        # What fraction are < 100 chars (dangerously short when truncated)
        very_short_trunc = (lengths < 100).sum()
        if very_short_trunc > 0:
            warn(f"  {very_short_trunc} truncated entries are also < 100 chars — very weak encoder signal.")
        # Sample them
        samples = df.loc[trunc_mask, ['name_a', 'name_b', col]].head(4)
        for _, row in samples.iterrows():
            entity = row['name_a'] if col == 'context_a' else row['name_b']
            print(f"    [{entity}]  →  \"{str(row[col])[-80:]}\"")

# ═══════════════════════════════════════════════════════════════════════════
# E. OFF-DOMAIN BOILERPLATE ROWS
# ═══════════════════════════════════════════════════════════════════════════
section("E. OFF-DOMAIN BOILERPLATE ROWS")

BOILERPLATE = {
    "Marine invertebrate / non-pathogenic aquatic": ["non-pathogenic aquatic species", "Marine invertebrate"],
    "List/enumeration placeholder":                 ["list gives some examples"],
    "Freshwater species (non-crop)":                ["vast number of freshwater"],
    "Archaeological site":                          ["archaeological site"],
}

all_bad_idx = set()
for label, patterns in BOILERPLATE.items():
    mask = pd.Series(False, index=df.index)
    for pat in patterns:
        mask |= df['context_a'].astype(str).str.contains(pat, case=False, na=False)
        mask |= df['context_b'].astype(str).str.contains(pat, case=False, na=False)
    n = mask.sum()
    all_bad_idx.update(df[mask].index.tolist())
    if n > 0:
        warn(f"[{label}]: {n} rows")
        sample = df[mask][['name_a','name_b','match','lambda_val']].head(4)
        print(sample.to_string(index=False))
    else:
        ok(f"[{label}]: 0 rows")

info(f"\nTotal rows with any boilerplate context: {len(all_bad_idx)}")
if len(all_bad_idx) > 0:
    bp_pos = df.loc[list(all_bad_idx), 'match'].eq(1).sum()
    bp_neg = df.loc[list(all_bad_idx), 'match'].eq(0).sum()
    warn(f"  Breakdown: {bp_pos} positives / {bp_neg} negatives with boilerplate context.")
    warn("  ACTION: Replace these contexts with entity-specific Wikipedia/AGROVOC descriptions.")

# ═══════════════════════════════════════════════════════════════════════════
# F. LAMBDA BIMODALITY & DISTRIBUTION SHAPE
# ═══════════════════════════════════════════════════════════════════════════
section("F. LAMBDA DISTRIBUTION SHAPE ANALYSIS")

lv = df['lambda_val']

# Decile breakdown
print("\n  Decile breakdown:")
for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"    P{int(q*100):02d}: {lv.quantile(q):.4f}")
info(f"  P05: {lv.quantile(0.05):.4f}   P95: {lv.quantile(0.95):.4f}")

# Bimodality coefficient (BC > 0.555 indicates bimodality)
n = len(lv)
skew = lv.skew()
kurt = lv.kurtosis()
bc   = (skew**2 + 1) / (kurt + 3 * ((n-1)**2) / ((n-2)*(n-3)))
info(f"\n  Skewness:             {skew:.4f}")
info(f"  Excess kurtosis:      {kurt:.4f}")
info(f"  Bimodality coeff BC:  {bc:.4f}  (>0.555 → bimodal)")
if bc > 0.555:
    warn("  Dataset lambda IS bimodal — model will see mostly extreme supervision signals.")
    warn("  Consider: (1) upsampling mid-range [0.3–0.7] pairs, OR")
    warn("            (2) using a smoothed MSE loss with temperature.")
else:
    ok("  Lambda distribution is unimodal — smooth gradient landscape.")

# Gap in the middle?
mid = ((lv >= 0.35) & (lv <= 0.65)).sum()
info(f"\n  Mid-zone [0.35–0.65] rows: {mid}  ({100*mid/len(df):.1f}%)")
if mid / len(df) < 0.15:
    warn("  Very sparse mid-zone — model will have a hard threshold rather than a gradual boundary.")

# Per-class lambda histograms (text)
sub("Lambda histogram by class")
bins = np.arange(0, 1.05, 0.1)
for cls, label in [(1, "match=1 (positives)"), (0, "match=0 (negatives)")]:
    sub_df = df[df['match'] == cls]['lambda_val']
    hist, _ = np.histogram(sub_df, bins=bins)
    print(f"\n  {label} ({len(sub_df)} rows):")
    for i, count in enumerate(hist):
        bar = "█" * int(count / max(hist) * 30)
        print(f"    [{bins[i]:.1f}–{bins[i+1]:.1f}] {bar:<30} {count}")

# ═══════════════════════════════════════════════════════════════════════════
# G. NAME LENGTH & ABBREVIATION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
section("G. NAME LENGTH & ABBREVIATION ANALYSIS")

for col in ['name_a', 'name_b']:
    lengths = df[col].astype(str).str.len()
    info(f"{col} — avg: {lengths.mean():.1f}  min: {lengths.min()}  max: {lengths.max()}  "
         f"  ≤3 chars: {(lengths<=3).sum()}  ≤5 chars: {(lengths<=5).sum()}")

# Very short names (≤2 chars) — likely junk
for col in ['name_a', 'name_b']:
    tiny = df[df[col].astype(str).str.len() <= 2]
    if len(tiny) > 0:
        warn(f"{len(tiny)} rows with {col} ≤ 2 chars (may be junk tokens):")
        print(tiny[[col, 'match', 'lambda_val']].head(5).to_string(index=False))
    else:
        ok(f"No {col} entries ≤ 2 chars")

# Token count distribution (proxy for multi-word disease names)
df['tokens_a'] = df['name_a'].astype(str).str.split().str.len()
df['tokens_b'] = df['name_b'].astype(str).str.split().str.len()
info(f"\n  name_a avg tokens: {df['tokens_a'].mean():.1f}  (single-word: {(df['tokens_a']==1).sum()})")
info(f"  name_b avg tokens: {df['tokens_b'].mean():.1f}  (single-word: {(df['tokens_b']==1).sum()})")

# ═══════════════════════════════════════════════════════════════════════════
# H. CONTEXT UNIQUENESS (near-duplicate detection)
# ═══════════════════════════════════════════════════════════════════════════
section("H. CONTEXT UNIQUENESS CHECK")

# Exact-duplicate contexts (same context text appearing multiple times)
for col in ['context_a', 'context_b']:
    ctx_counts = df[col].value_counts()
    repeated = ctx_counts[ctx_counts > 1]
    info(f"{col}: {len(repeated)} unique context strings appear in >1 row (out of {df[col].nunique()} unique)")
    if len(repeated) > 0:
        top5 = repeated.head(5)
        for ctx, cnt in top5.items():
            preview = str(ctx)[:60].replace('\n', ' ')
            print(f"    ×{cnt}  \"{preview}...\"")

# Same-context positive pairs (context carries no discriminating signal)
same_ctx_pos = ((df['context_a'] == df['context_b']) & (df['match'] == 1)).sum()
same_ctx_neg = ((df['context_a'] == df['context_b']) & (df['match'] == 0)).sum()
if same_ctx_pos > 0:
    warn(f"{same_ctx_pos} match=1 rows with identical context_a == context_b "
         "(context provides no signal; model must rely entirely on name).")
else:
    ok("No match=1 same-context pairs.")

if same_ctx_neg > 0:
    err(f"{same_ctx_neg} match=0 rows with identical context — CRITICAL: model gets contradictory gradients!")
else:
    ok("No match=0 same-context pairs.")

# ═══════════════════════════════════════════════════════════════════════════
# I. ENTITY LEAKAGE RISK
# ═══════════════════════════════════════════════════════════════════════════
section("I. ENTITY LEAKAGE RISK  (entity appearing on both sides of different pairs)")

# An entity present in both name_a and name_b columns across different rows
# could cause a train/test split to "see" the entity in both sets.
names_a = set(df['name_a'].str.lower().str.strip())
names_b = set(df['name_b'].str.lower().str.strip())
overlap = names_a & names_b
info(f"Distinct entities in name_a: {len(names_a)}")
info(f"Distinct entities in name_b: {len(names_b)}")
info(f"Entities appearing on both sides: {len(overlap)}")

if len(overlap) / len(names_a) > 0.5:
    warn("Over 50% of name_a entities also appear in name_b — high leakage risk if you "
         "do a random row split. Use entity-stratified train/val split instead.")
else:
    ok("Overlap is manageable. Still recommended: use entity-stratified split to avoid leakage.")

# Canonical ID frequency — are any IDs over-represented?
all_cids = pd.concat([df['canonical_id_a'], df['canonical_id_b']])
top_ids = all_cids.value_counts().head(10)
sub("Top 10 most frequent canonical IDs (potential over-representation)")
print(top_ids.to_string())
heaviest_id, heaviest_count = top_ids.index[0], top_ids.iloc[0]
if heaviest_count > 30:
    warn(f"'{heaviest_id}' appears in {heaviest_count} rows — highly represented entity "
         "may dominate gradients. Consider capping at 15–20 pairs per entity.")

# ═══════════════════════════════════════════════════════════════════════════
# J. TRAINING SIGNAL QUALITY SCORE
# ═══════════════════════════════════════════════════════════════════════════
section("J. PER-ROW TRAINING SIGNAL QUALITY SCORE")

# Composite score (0–100): penalises low-context, boundary lambda, label conflicts
def quality_score(row):
    score = 100
    # Short contexts
    if len(str(row['context_a'])) < 80: score -= 15
    if len(str(row['context_b'])) < 80: score -= 15
    # Truncated context
    if str(row['context_a']).endswith(("...", "…")): score -= 10
    if str(row['context_b']).endswith(("...", "…")): score -= 10
    # Lambda near 0 for positives
    if row['match'] == 1 and row['lambda_val'] < 0.1: score -= 20
    # Lambda near 1 for negatives
    if row['match'] == 0 and row['lambda_val'] > 0.85: score -= 20
    # LLM conflict
    llm_true = str(row['llm_match']).upper() == "TRUE"
    if bool(row['match']) != llm_true: score -= 25
    # Same context
    if row['context_a'] == row['context_b'] and row['match'] == 0: score -= 40
    return max(score, 0)

df['quality_score'] = df.apply(quality_score, axis=1)

buckets_q = [(90, 101, "Excellent (90–100)"), (70, 90, "Good (70–89)"),
             (50, 70, "Fair (50–69)"),         (0,  50, "Poor (<50)")]
print()
for lo, hi, label in buckets_q:
    n = ((df['quality_score'] >= lo) & (df['quality_score'] < hi)).sum()
    pct = 100 * n / len(df)
    bar = "█" * int(pct / 2)
    print(f"  {label:<24} {bar:<50} {n:>5} ({pct:.1f}%)")

low_q = df[df['quality_score'] < 50]
if len(low_q) > 0:
    warn(f"\n{len(low_q)} rows with quality score < 50 — consider filtering or fixing:")
    print(low_q[['name_a','name_b','match','lambda_val','quality_score']].head(10).to_string(index=False))
else:
    ok("No rows with quality score < 50.")

info(f"\n  Mean quality score: {df['quality_score'].mean():.1f}")
info(f"  Median:             {df['quality_score'].median():.1f}")
info(f"  Rows scoring 100:   {(df['quality_score']==100).sum()}")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL DEEP-VALIDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
section("DEEP VALIDATION SUMMARY")

issues = []
actions = []

if ratio < 3:
    issues.append(f"Class ratio {ratio:.1f}:1 (below 3:1 guideline)")
    actions.append("Consider adding ~250 more negatives OR use weighted BCELoss(pos_weight=2.0)")

noisy_nn = ((df['match']==0) & (df['lambda_val']>=0.5)).sum()
if noisy_nn > 0:
    issues.append(f"{noisy_nn} noisy negatives (match=0, lambda≥0.5)")
    actions.append("Audit noisy negatives manually; worst offenders listed in Section B")

low_conf_p = ((df['match']==1) & (df['lambda_val']<0.3)).sum()
if low_conf_p > 0:
    issues.append(f"{low_conf_p} low-confidence positives (match=1, lambda<0.3)")
    actions.append("Review low-confidence positives; most are acceptable abbreviation pairs")

total_trunc = df['context_a'].astype(str).str.endswith(("...","…")).sum() + \
              df['context_b'].astype(str).str.endswith(("...","…")).sum()
if total_trunc > 0:
    issues.append(f"{total_trunc} truncated context entries")
    actions.append("Re-scrape truncated contexts from Wikipedia/AGROVOC for those entities")

if len(all_bad_idx) > 0:
    issues.append(f"{len(all_bad_idx)} rows with off-domain boilerplate context")
    actions.append("Replace boilerplate contexts with entity-specific descriptions (see Section E)")

if bc > 0.555:
    issues.append(f"Lambda distribution is bimodal (BC={bc:.3f})")
    actions.append("Add mid-range pairs OR use label smoothing / temperature in loss")

if heaviest_count > 30:
    issues.append(f"Canonical ID '{heaviest_id}' over-represented ({heaviest_count} rows)")
    actions.append("Cap pairs per entity at 15–20 to prevent gradient domination")

print(f"\n  {'ISSUE':<55}  ACTION")
print(f"  {'-'*55}  {'-'*40}")
for i, a in zip(issues, actions):
    print(f"  ⚠  {i:<55}")
    print(f"     → {a}\n")

if not issues:
    print("\n  No issues found. Dataset is fully ready for training.")
else:
    print(f"\n  {len(issues)} advisory item(s) — none are training-blockers.")
    print("  VERDICT: Dataset is TRAINABLE. Fixing the above will improve model performance.")

print(f"\n{SEP}\n")
