# Dataset V2 Builder — Agricultural Disease Entity Resolution

## Overview

Build a comprehensive, production-ready entity resolution dataset with **all four pair types** and complete context validation:

- **Type A**: Safe matches (name + context agree)
- **Type B**: Synonym pairs (names differ, same disease) ← needs context
- **Type C**: Polysemy pairs (names similar, different diseases) ← conflict detection
- **Type D**: Clear non-matches (neither signal agrees)

**Target dataset:**

- Training: 2,500 pairs
- External test set: 200 pairs (untouched for evaluation)

## Project Structure

```
dataset_v2_builder/
├── scripts/
│   ├── run_all.py                      # Master runner (runs all steps)
│   ├── step1_context_quality.py        # Score contexts (good/medium/poor)
│   ├── step2_pair_type_classifier.py   # Classify into A/B/C/D types
│   ├── step3_eppo_collector.py         # Collect Type B+C from EPPO API
│   ├── step4_usda_external_test.py     # Collect external test set
│   └── step5_merge_all.py              # Merge and finalize dataset_v2
│
├── data/
│   ├── base_dataset.csv                # Base 1,881 pairs (copy from production)
│   ├── dataset_with_quality.csv        # After step 1 (+ quality scores)
│   ├── dataset_classified.csv          # After step 2 (+ pair types)
│   ├── eppo_pairs_collected.csv        # After step 3
│   ├── usda_external_test_set.csv      # After step 4
│   ├── dataset_v2.csv                  # ⭐ TRAINING SET (2,500 pairs)
│   ├── external_test_set_isolated.csv  # ⭐ EXTERNAL TEST (200 pairs)
│   ├── quality_report.txt              # Context quality analysis
│   └── pair_type_report.txt            # Pair type distribution
│
└── README.md                           # This file
```

## Quick Start

### Option 1: Run All Steps (Recommended)

```bash
cd dataset_v2_builder/scripts
python run_all.py
```

This runs all 5 steps in sequence with error handling.

### Option 2: Run Individual Steps

```bash
cd dataset_v2_builder/scripts

# Step 1: Score every context
python step1_context_quality.py

# Step 2: Classify pairs into types
python step2_pair_type_classifier.py

# Step 3: Collect from EPPO (requires EPPO_API_KEY in .env)
python step3_eppo_collector.py

# Step 4: Collect external test set from USDA
python step4_usda_external_test.py

# Step 5: Merge everything
python step5_merge_all.py
```

## API Keys Required

Edit `../../.env` and add:

```env
# Get from https://data.eppo.int/
EPPO_API_KEY=your_key_here

# USDA is open source (no key needed)
USDA_API_KEY=
```

## Pipeline Details

### Step 1: Context Quality Scoring

**Input:** `base_dataset.csv` (1,881 pairs from training_ready_production.csv)

**What it does:**

- Scores every context as "good", "medium", or "poor"
- "Good": specific disease description (pathogen named, symptoms described)
- "Medium": partial information (some disease data but missing key details)
- "Poor": useless (Wikipedia list pages, taxonomy stubs, generic text)

**Output:** `dataset_with_quality.csv` + quality_report.txt

**Why:**

- Good contexts enable proper entity resolution through semantic signals
- Poor contexts corrupt lambda supervision during training
- You can track which contexts need improvement

### Step 2: Pair Type Classification

**Input:** `dataset_with_quality.csv`

**What it does:**

- Classifies each pair into Type A, B, C, or D using name similarity
- Uses Jaccard similarity (no ML model, fully deterministic)
- Threshold: name_sim ≥ 0.25 → "similar names"

**Classification logic:**

```
match=1 & name_sim ≥ 0.25 → Type A (safe match)
match=1 & name_sim < 0.25 → Type B (synonym pair)
match=0 & name_sim ≥ 0.25 → Type C (polysemy)
match=0 & name_sim < 0.25 → Type D (clear non-match)
```

**Output:** `dataset_classified.csv` + pair_type_report.txt

**Why:**

- Type A: Model sees this easily, don't evaluate mainly on these
- Type B: **Hardest true-matches** — names are completely different
- Type C: **Most valuable for conflict detection** — names seem to match but diseases differ
- Type D: Model should easily reject these

### Step 3: EPPO API Collection

**Input:** EPPO database (via API)

**What it does:**

- Searches EPPO for diseases with multiple names → **Type B synonym pairs**
- Searches for ambiguous common names (rust, blight, wilt, mosaic) → **Type C polysemy pairs**
- Example Type B: "Panama disease" = "Fusarium wilt of banana" = "Mal de Panamá"
- Example Type C: "rust" appears in wheat rust, corn rust, citrus rust (different pathogens)

**Target collection:**

- ~300 Type B pairs (synonyms)
- ~300 Type C pairs (polysemy)

**Output:** `eppo_pairs_collected.csv`

**Why:**

- EPPO is authoritative for crop disease nomenclature
- Synonym pairs test whether context can overcome name differences
- Polysemy pairs directly address the "nitrogen problem" (same name, different meaning)

### Step 4: USDA External Test Set

**Input:** USDA PLANTS database

**What it does:**

- Collects 200 pairs from a completely separate source
- These pairs are **isolated for evaluation only**
- Never trained on, never tuned based on

**Output:** `usda_external_test_set.csv`

**Why:**

- Honest evaluation without data leakage
- USDA data is completely separate from your training sources
- Ensures model generalizes beyond Wikipedia + PlantVillage + EPPO

### Step 5: Final Merge

**Inputs:** All previous outputs

**What it does:**

- Merges classified pairs + EPPO supplements
- Deduplicates
- Separates external test set
- Ensures consistent column order
- Reports pair type distribution vs. targets

**Outputs:**

- `dataset_v2.csv` (2,500 training pairs with full metadata)
- `external_test_set_isolated.csv` (200 test pairs, locked from training)

## Understanding Pair Types

### Type A: Safe Matches (400 pairs)

- Both name similarity AND match label agree
- Name: "late blight" vs "late blight of potato" (very similar)
- Context: Both mention Phytophthora infestans
- **Model impact:** Easy positive examples, important for calibration but don't evaluate mainly on these

### Type B: Synonym Pairs (300 pairs) ⚠️ **CRITICAL FOR CONTEXT**

- Same disease, completely different names
- Name: "Panama disease" vs "Fusarium wilt of banana" (name_sim ≈ 0.0)
- Context: Both specifically mention Fusarium oxysporum f.sp. cubense affecting banana
- **Model impact:** Tests whether your encoder learned to recognize semantic equivalence despite name differences

### Type C: Polysemy Pairs (300 pairs) ⚠️ **CRITICAL FOR CONFLICT DETECTION**

- Different diseases, similar/same names
- Name: "rust (wheat)" vs "rust (corn)" (name_sim ≈ 0.8)
- Context: Wheat rust (Puccinia striiformis) vs Corn rust (Puccinia sorghi) — different pathogens
- **Model impact:** Tests whether your model uses context to disambiguate when names are misleading
- **In training:** Oversample 3x to prevent name-similarity bias

### Type D: Clear Non-Match (600 pairs)

- Both name AND semantic signals disagree
- Name: "late blight" vs "rice blast" (completely different)
- Context: Different pathogens, different crops
- **Model impact:** Easy negative examples, good for precision calibration

## Training with Dataset V2

### Recommended Training Setup

```python
import pandas as pd
from torch.utils.data import WeightedRandomSampler

# Load dataset_v2
df = pd.read_csv("data/dataset_v2.csv")

# Oversample Type C (polysemy) pairs 3x during training
# This prevents name-similarity bias
weights = []
for _, row in df.iterrows():
    if row["pair_type"] == "C":
        weights.append(3.0)  # Oversample Type C
    else:
        weights.append(1.0)

sampler = WeightedRandomSampler(weights, len(df), replacement=True)

# Use sampler in DataLoader
# train_loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### Context Quality Handling

```python
# Exclude "both_poor" pairs from lambda supervision
# (keep for binary match loss though)
both_poor = (df["context_quality_a"] == "poor") & (df["context_quality_b"] == "poor")

for idx, row in df.iterrows():
    if both_poor.iloc[idx]:
        # Use for match/non-match signal only
        use_lambda = False
    else:
        # Use both match label AND lambda signal
        use_lambda = True
```

### Final Evaluation

**Do NOT touch external_test_set_isolated.csv until final model is ready.**

```python
# After training and validation on dataset_v2
test_df = pd.read_csv("external_test_set_isolated.csv")

# Evaluate model on external test set
# This is your honest generalization metric
```

## Output Reports

### quality_report.txt

```
CONTEXT QUALITY REPORT
Total pairs: 1881

── context_a quality ──────────────────────────
  good    : 1652 (87.8%) ████████████████████
  medium  :  156 (8.3%)  ██
  poor    :   73 (3.9%)  █

── Both contexts poor (most dangerous): 15 pairs
   These pairs corrupt your lambda supervision signal most.
```

### pair_type_report.txt

```
PAIR TYPE CLASSIFICATION REPORT
Total pairs: 1881

── Coverage by pair type ──────────────────────────────────
  Type    Count      %   vs Target  Description
  A       410    21.8%   +10 (OK)   Safe match (match=1, names similar)
  B       125    6.6%    -175 (need 175 more) Synonym pair (match=1, names different)
  C       156    8.3%    -144 (need 144 more) Polysemy (match=0, names similar) ← CRITICAL
  D       1190  63.3%   +590 (OK)   Clear non-match (match=0, names different)

── ACTION ITEMS ──────────────────────────────────────────
  URGENT: Only 156 Type C (polysemy) pairs.
  Need ~144 more from EPPO ambiguous common names.
  IMPORTANT: Only 125 Type B (synonym) pairs.
  Need ~175 more from EPPO synonym lists.
```

## Troubleshooting

### EPPO API connection fails

```
Error: EPPO_API_KEY not set in .env
→ Check ../../.env has EPPO_API_KEY=your_key
→ Get key from https://data.eppo.int/
```

### Step X fails, want to skip ahead

```bash
# You can manually run just step5
python step5_merge_all.py
# It will warn about missing files but still try to merge what exists
```

### Want to adjust thresholds

Edit step2_pair_type_classifier.py:

```python
NAME_SIM_THRESHOLD = 0.25  # ← Change this if Type B/C split looks wrong
```

Lower threshold → more pairs classified as "similar names" (A or C)
Higher threshold → fewer pairs classified as "similar names"

## Key Insights for Model Development

1. **Type C is your test case**: If model can't distinguish polysemy pairs, it's relying too much on name similarity

2. **Type B needs good context encoding**: If model fails on synonyms, embeddings aren't capturing semantic meaning

3. **Context quality matters**: Pairs with "poor" contexts on both sides corrupt lambda supervision → consider excluding from lambda loss

4. **External test set is sacred**: Don't peek at results during development. Only final model evaluation.

5. **Balanced pair distribution**: Type A/D are "easy". Don't let them dominate. Oversample B/C during training.

## For Academic Papers

Cite the data sources:

- **Base dataset (1,881 pairs)**: Wikipedia, PlantVillage, AGROVOC (as per training_ready_production.csv)
- **Supplements (700 pairs)**: EPPO API, USDA PLANTS database
- **Evaluation (200 pairs)**: USDA PLANTS database (held-out external test set)

Report distribution:

```
Dataset V2: 2,500 training pairs
  - Type A (same name, same meaning): 400
  - Type B (different names, same disease): 300
  - Type C (same names, different diseases): 300
  - Type D (different names, different disease): 1,600

External evaluation set: 200 pairs from USDA PLANTS (held-out)
```

## Next Steps

1. Run all steps: `python run_all.py`
2. Check quality_report.txt and pair_type_report.txt
3. Load dataset_v2.csv in your training pipeline
4. Oversample Type C pairs (3x) during training
5. Evaluate final model ONLY on external_test_set_isolated.csv
6. Report metrics disaggregated by pair type

Good luck! 🌱
