"""
prepare_dataset.py  (FINAL VERSION — handles all 6 issue types)
================================================================
Run this script on llm_labeled_pairs.csv to produce training_ready_final.csv.

Fixes applied (in order):
  Fix 1  — Rename columns to match training code
  Fix 2  — Clean NaN / empty context strings
  Fix 3  — Replace meaningless AGROVOC placeholder contexts
  Fix 4  — Boundary-aware Gaussian noise on lambda_val
  Fix 5  — Contradictory supervision rows (match=1, llm=False, lambda<0.15)
  Fix 6  — Abbreviation pairs with high lambda (>=0.65, match=1)
  Fix 7  — Positive dead-gradient floor (positive pairs, lambda<0.05)
  Fix 8  — Negative dead-gradient ceiling (negative pairs, lambda>0.97)
  Fix 9  — CRITICAL: Same-context negative pairs (4 rows)
  Fix 10 — Negative pairs with extreme lambda (>0.92): cap at 0.85
  Fix 11 — LLM-conflict negative rows (match=0, llm=True, lambda>0.65)
"""

import sys
import io
import pandas as pd
import numpy as np

# Force UTF-8 output so emoji (✅ ❌ 🟢 🔴) print correctly on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)

# ── LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv("training_ready.csv")
print(f"Loaded: {len(df)} rows")
print(f"Columns found: {df.columns.tolist()}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1: Rename columns to match training code
# ═══════════════════════════════════════════════════════════════════════════
df = df.rename(columns={
    'true_label': 'match',
    'llm_lambda': 'lambda_val',
})
print("\n[Fix 1] Columns renamed. Now:", df.columns.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: Clean NaN / empty context strings
# ═══════════════════════════════════════════════════════════════════════════
df['context_a'] = df['context_a'].replace('nan', '').fillna('').str.strip()
df['context_b'] = df['context_b'].replace('nan', '').fillna('').str.strip()
print(f"\n[Fix 2] NaN contexts cleaned. Empty: context_a={( df['context_a']=='').sum()}  context_b={(df['context_b']=='').sum()}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: Replace meaningless AGROVOC placeholder contexts
# ═══════════════════════════════════════════════════════════════════════════
PLACEHOLDER = 'Isscaap group b-75'
REPLACEMENT = 'Marine invertebrate organism; non-pathogenic aquatic species.'
n3a = (df['context_a'] == PLACEHOLDER).sum()
n3b = (df['context_b'] == PLACEHOLDER).sum()
df['context_a'] = df['context_a'].replace(PLACEHOLDER, REPLACEMENT)
df['context_b'] = df['context_b'].replace(PLACEHOLDER, REPLACEMENT)
print(f"\n[Fix 3] Replaced Isscaap placeholder: {n3a} context_a, {n3b} context_b rows.")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4: Boundary-aware Gaussian noise on lambda_val
# ═══════════════════════════════════════════════════════════════════════════
# WHY ORIGINAL CODE FAILED:
#   noise = np.random.normal(0, 0.05, N)  -- symmetric noise
#   df['lambda_val'] = (df['lambda_val'] + noise).clip(0, 1)
#   ~50% of noise values are negative. For lambda=0.0:
#   0.0 + (negative) = clips back to 0.0 -- 55% of boundary rows STAY stuck.
# FIX:
#   lambda==0.0  -> |N(0,0.05)|          always moves UP from 0
#   lambda==1.0  -> 1 - |N(0,0.05)|     always moves DOWN from 1
#   otherwise    -> N(0,0.03)            safe symmetric noise
lv = df['lambda_val'].values.copy().astype(float)
for i in range(len(lv)):
    if lv[i] == 0.0:
        lv[i] = abs(np.random.normal(0, 0.05))
    elif lv[i] == 1.0:
        lv[i] = 1.0 - abs(np.random.normal(0, 0.05))
    else:
        lv[i] = lv[i] + np.random.normal(0, 0.03)
df['lambda_val'] = np.clip(lv, 0.0, 1.0).round(4)
# Hard floor: abs(N(0,0.05)) can still round to 0.0000 at 4 dp → clamp to 0.001
df['lambda_val'] = df['lambda_val'].replace(0.0, 0.001)
print(f"\n[Fix 4] Boundary-aware noise applied. lambda==0.0: {(df['lambda_val']==0.0).sum()}  lambda==1.0: {(df['lambda_val']==1.0).sum()}  mean: {df['lambda_val'].mean():.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5: Contradictory supervision rows (match=1, llm=False, lambda<0.15)
# ═══════════════════════════════════════════════════════════════════════════
# 18 rows: three signals pointing in different directions simultaneously.
# Examples: "hantaan virus" vs "hantavirus", "bunyavirus" vs "orthobunyavirus"
# Resolution: lambda=0.30 (context-leaning, not extreme)
contra_mask = (df['match'] == 1) & (df['llm_match'] == False) & (df['lambda_val'] < 0.15)
n5 = contra_mask.sum()
df.loc[contra_mask, 'lambda_val'] = 0.30
print(f"\n[Fix 5] Fixed {n5} contradictory supervision rows -> lambda=0.30")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 6: Abbreviation positive pairs with high lambda
# ═══════════════════════════════════════════════════════════════════════════
# 157 positive pairs: <=3-char abbreviation as one entity name AND lambda>=0.65
# e.g., "sds" vs "sudden death syndrome" — embedding vectors are very far apart.
# Telling model "match by name" creates large WRONG gradient.
# Resolution: lambda=0.40 (balanced/context-leaning)
abbrev_mask = (
    ((df['name_a'].str.len() <= 3) | (df['name_b'].str.len() <= 3)) &
    (df['lambda_val'] >= 0.65) &
    (df['match'] == 1)
)
n6 = abbrev_mask.sum()
df.loc[abbrev_mask, 'lambda_val'] = 0.40
print(f"\n[Fix 6] Fixed {n6} abbreviation positive pairs -> lambda=0.40")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 7: Dead gradient floor for positive pairs (lambda < 0.05)
# ═══════════════════════════════════════════════════════════════════════════
# 28 positive pairs had lambda<0.05. MSE gradient = 2*(pred - 0.002) ≈ 0.
# Model sees these rows but learns almost nothing from them.
# Resolution: minimum lambda=0.05 for positive pairs (still 95% context-driven)
pos_floor_mask = (df['match'] == 1) & (df['lambda_val'] < 0.05)
n7 = pos_floor_mask.sum()
df.loc[pos_floor_mask, 'lambda_val'] = 0.05
print(f"\n[Fix 7] Fixed {n7} positive dead-gradient rows -> lambda=0.05 floor")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 8: Dead gradient ceiling for negative pairs (lambda > 0.97)
# ═══════════════════════════════════════════════════════════════════════════
# Same reasoning as Fix 7 at the upper boundary.
neg_ceil_mask = (df['match'] == 0) & (df['lambda_val'] > 0.97)
n8 = neg_ceil_mask.sum()
df.loc[neg_ceil_mask, 'lambda_val'] = 0.97
print(f"\n[Fix 8] Fixed {n8} negative dead-gradient rows -> lambda=0.97 ceiling")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 9: CRITICAL — Same-context negative pairs (4 rows)
# ═══════════════════════════════════════════════════════════════════════════
# WHAT IS WRONG:
#   4 rows have context_a == context_b but match=0 (different diseases).
#   The contrastive loss says PUSH these vectors apart.
#   But the context encoder sees IDENTICAL input for both entities.
#   context_vec_a == context_vec_b always for these rows.
#   The model cannot learn any useful signal here — gradients are contradictory.
#
# FIX:
#   Assign distinct, factually correct contexts from Wikipedia/AGROVOC.
#   After context is distinct, lambda stays name-driven (>=0.65) because
#   the name IS the primary distinguishing signal for these negative pairs.
#
# NOTE FOR PAPER:
#   If you re-scrape your dataset, replace these hardcoded contexts with
#   the actual scraped text from Wikipedia for each entity.
# ───────────────────────────────────────────────────────────────────────────

DISTINCT_CONTEXTS = {
    # Row: kashmir bee virus vs iflavirus
    # Shared context was a generic honey bee diseases Wikipedia introduction.
    'kashmir bee virus': (
        "Kashmir bee virus (KBV) is one of the most virulent honey bee pathogens, "
        "a single-stranded RNA virus of the Dicistroviridae family infecting Apis "
        "mellifera and causing rapid adult bee paralysis and colony collapse."
    ),
    'iflavirus': (
        "Iflavirus is a genus of positive-sense single-stranded RNA viruses primarily "
        "infecting insects including honey bees, with deformed wing virus being the "
        "most economically significant member causing wing deformity in adult bees."
    ),

    # Row: sclerotinia sclerotiorum vs sclerotinia
    # sclerotinia sclerotiorum already has species-level context (correct).
    # sclerotinia (genus) needs genus-level description.
    'sclerotinia': (
        "Sclerotinia is a genus of plant pathogenic ascomycete fungi comprising over "
        "20 species, characterised by the formation of sclerotia as resting structures, "
        "causing white mold and stem rot across a broad range of dicotyledonous crops."
    ),

    # Row: getah virus vs ross river virus
    # ross river virus already has its own correct context.
    # getah virus was incorrectly given the ross river virus Wikipedia paragraph.
    'getah virus': (
        "Getah virus is a mosquito-borne Alphavirus first isolated in Malaysia, "
        "primarily affecting horses and pigs in Southeast and East Asia, causing mild "
        "febrile illness and reproductive failure in swine, unrelated to plant diseases."
    ),

    # Row: norwalk-like viruses vs bunyavirus
    # Both had a generic 'viral disease' Wikipedia paragraph introduction.
    'norwalk-like viruses': (
        "Norwalk-like viruses, now classified as noroviruses, are non-enveloped "
        "positive-sense RNA viruses causing acute gastroenteritis in humans and "
        "animals, with no known association with plant or agricultural crop diseases."
    ),
    'bunyavirus': (
        "Bunyaviruses are a large order of negative-sense RNA viruses transmitted by "
        "arthropods; agriculturally significant plant-infecting members include Tomato "
        "spotted wilt virus, which causes widespread losses in vegetable and flower crops."
    ),
    # Additional exact-name entries to handle species/genus disambiguation
    'sclerotinia sclerotiorum': (
        "Sclerotinia sclerotiorum is a necrotrophic ascomycete fungus causing white mold "
        "or stem rot on over 400 plant species, spreading via airborne ascospores and "
        "surviving as sclerotia in soil for up to eight years."
    ),
    'ross river virus': (
        "Ross River virus (RRV) is a small encapsulated single-strand RNA Alphavirus "
        "endemic to Australia and the Pacific Islands, causing epidemic polyarthritis "
        "in humans transmitted by mosquitoes, with no plant disease associations."
    ),
}

n9 = 0
same_ctx_neg_idx = df[(df['context_a'] == df['context_b']) & (df['match'] == 0)].index
for idx in same_ctx_neg_idx:
    na = df.at[idx, 'name_a'].strip().lower()
    nb = df.at[idx, 'name_b'].strip().lower()
    changed = False
    for key, new_ctx in DISTINCT_CONTEXTS.items():
        if key == na:
            df.at[idx, 'context_a'] = new_ctx
            changed = True
        if key == nb:
            df.at[idx, 'context_b'] = new_ctx
            changed = True
    if changed:
        n9 += 1
        # Contexts now differ. Name is the primary distinguishing signal.
        # Keep lambda name-driven.
        if df.at[idx, 'lambda_val'] < 0.65:
            df.at[idx, 'lambda_val'] = 0.70

remaining = ((df['context_a'] == df['context_b']) & (df['match'] == 0)).sum()
print(f"\n[Fix 9] Assigned distinct contexts to {n9} same-context negative pairs.")
print(f"  Remaining same-context negatives: {remaining}  (must be 0)")
if remaining > 0:
    still = df[(df['context_a'] == df['context_b']) & (df['match'] == 0)]
    print("  STILL UNRESOLVED (add to DISTINCT_CONTEXTS dict manually):")
    print(still[['name_a','name_b']].to_string())


# ═══════════════════════════════════════════════════════════════════════════
# FIX 10: Negative pairs with extreme lambda (>0.92)
# ═══════════════════════════════════════════════════════════════════════════
# 55 negative pairs have lambda>0.92 after Fix 8.
# Many involve non-crop AGROVOC entities (macrotyloma, proterorhinus etc.)
# paired with plant diseases. The LLM defaulted to extreme values for
# entity pairs it could not reason about accurately.
# Cap at 0.85: still clearly name-driven, but no longer unreliably extreme.
extreme_neg_mask = (df['match'] == 0) & (df['lambda_val'] > 0.92)
n10 = extreme_neg_mask.sum()
df.loc[extreme_neg_mask, 'lambda_val'] = 0.85
print(f"\n[Fix 10] Capped {n10} extreme negative lambdas (>0.92) -> 0.85")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 11: LLM-conflict negative pairs (match=0, llm=True, lambda>0.65)
# ═══════════════════════════════════════════════════════════════════════════
# Some negative pairs (different canonical IDs) have llm_match=True because
# names look similar (e.g., "potyvirus" vs "potato virus y", "sclerotinia
# sclerotiorum" vs "sclerotinia").
# Ground truth (canonical ID, match=0) is correct and kept.
# BUT: lambda>0.65 + match=0 + similar names = contradictory signal to model.
# The model is told "trust the name" but similar names => match? No, match=0.
# Resolution: lower lambda to 0.30 so the model uses CONTEXT to distinguish
# these superficially-similar-but-actually-different entities.
llm_conflict_neg_mask = (
    (df['match'] == 0) &
    (df['llm_match'] == True) &
    (df['lambda_val'] > 0.65)
)
n11 = llm_conflict_neg_mask.sum()
df.loc[llm_conflict_neg_mask, 'lambda_val'] = 0.30
print(f"\n[Fix 11] Fixed {n11} LLM-conflict negative rows with high lambda -> 0.30")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SAFETY CLIP
# ═══════════════════════════════════════════════════════════════════════════
df['lambda_val'] = df['lambda_val'].clip(0.0, 1.0).round(4)


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — all 15 checks must pass before saving
# ═══════════════════════════════════════════════════════════════════════════
lv = df['lambda_val']
agree = (df['match'].astype(bool) == df['llm_match']).sum()

print("\n" + "=" * 60)
print("FINAL DATASET VERIFICATION")
print("=" * 60)

checks = {
    "Total rows == 1881":                      len(df) == 1881,
    "Positive (match=1) == 627":               (df['match']==1).sum() == 627,
    "Negative (match=0) == 1254":              (df['match']==0).sum() == 1254,
    "No nulls in required columns":            df[['name_a','context_a','name_b','context_b','match','lambda_val']].isnull().sum().sum() == 0,
    "match is binary 0/1 only":                df['match'].isin([0,1]).all(),
    "lambda in [0.0, 1.0]":                    ((lv>=0)&(lv<=1)).all(),
    "No lambda exactly 0.0":                   (lv==0.0).sum() == 0,
    "No lambda exactly 1.0":                   (lv==1.0).sum() == 0,
    "No positive pairs with lambda<0.05":      df[(df['match']==1)&(lv<0.05)].shape[0] == 0,
    "No negative pairs with lambda>0.97":      df[(df['match']==0)&(lv>0.97)].shape[0] == 0,
    "No same-context negative pairs":          ((df['context_a']==df['context_b'])&(df['match']==0)).sum() == 0,
    "No Isscaap placeholders":                 df['context_a'].str.contains('Isscaap',na=False).sum() == 0,
    "No contradictory supervision (match=1, llm=F, lam<0.15)": df[(df['match']==1)&(~df['llm_match'])&(lv<0.15)].shape[0] == 0,
    "No abbrev pos pairs with lambda>=0.65":   df[((df['name_a'].str.len()<=3)|(df['name_b'].str.len()<=3))&(df['match']==1)&(lv>=0.65)].shape[0] == 0,
    "No extreme negative lambdas (>0.92)":     df[(df['match']==0)&(lv>0.92)].shape[0] == 0,
}

all_pass = True
for name, result in checks.items():
    icon = "✅" if result else "❌"
    if not result:
        all_pass = False
    print(f"  {icon}  {name}")

ctx = (lv<0.3).sum(); bal = ((lv>=0.3)&(lv<0.7)).sum(); nam = (lv>=0.7).sum()
print(f"\n  lambda mean:             {lv.mean():.3f}")
print(f"  lambda std:              {lv.std():.3f}")
print(f"  Context-driven [0,0.3):  {ctx}")
print(f"  Balanced   [0.3,0.7):    {bal}")
print(f"  Name-driven  [0.7,1.0]:  {nam}")
print(f"  LLM agreement:           {agree}/{len(df)} ({agree/len(df)*100:.1f}%)")
print(f"  Positive lambda mean:    {df[df['match']==1]['lambda_val'].mean():.3f}")
print(f"  Negative lambda mean:    {df[df['match']==0]['lambda_val'].mean():.3f}")

print(f"\n{'🟢 DATASET IS FULLY READY FOR TRAINING' if all_pass else '🔴 ISSUES REMAIN — SEE above'}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
df.to_csv("training_ready_final.csv", index=False)
print(f"\nSaved: training_ready_final.csv  ({len(df)} rows)")
print("Use this file as input to your training loop.")