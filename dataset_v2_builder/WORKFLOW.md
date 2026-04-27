# Dataset V2 Building Workflow

## Quick Reference

### What You Have

- ✅ Base dataset: 1,881 pairs with complete, validated contexts
- ✅ Quality scoring engine: Classifies all contexts
- ✅ Pair classifier: Types each pair as A/B/C/D
- ✅ EPPO integration: Collects synonyms and polysemy pairs
- ✅ USDA integration: External test set from separate source

### What You're Building

- 🎯 dataset_v2.csv: 2,500 training pairs
- 🎯 external_test_set_isolated.csv: 200 evaluation-only pairs
- 📊 Quality reports showing pair type distribution

### Timeline

- Step 1: ~30 seconds (1,881 contexts scored)
- Step 2: ~10 seconds (1,881 pairs classified)
- Step 3: ~2 minutes (EPPO API calls, with rate limiting)
- Step 4: ~10 seconds (USDA test set generation)
- Step 5: ~5 seconds (merge and finalize)
- **Total: ~3 minutes**

---

## Step-by-Step Execution

### Setup (One-time)

```bash
# 1. Verify directory structure
cd /d/Coding/entity_resolution_copy
ls -la dataset_v2_builder/

# 2. Add EPPO API key (if you have one)
# Edit .env and add:
#   EPPO_API_KEY=your_key_here
# Get key from: https://data.eppo.int/

# 3. Verify base dataset was copied
ls -lh dataset_v2_builder/data/base_dataset.csv
# Should show ~1.1M
```

### Run Pipeline

```bash
# Option A: Run all steps at once (recommended)
cd dataset_v2_builder/scripts
python run_all.py

# Option B: Run steps individually (for debugging)
cd dataset_v2_builder/scripts
python step1_context_quality.py      # ~30 sec
python step2_pair_type_classifier.py # ~10 sec
python step3_eppo_collector.py       # ~2 min
python step4_usda_external_test.py   # ~10 sec
python step5_merge_all.py            # ~5 sec
```

### Check Results

```bash
cd dataset_v2_builder/data

# 1. Check dataset sizes
ls -lh dataset_v2.csv external_test_set_isolated.csv

# 2. Quick statistics
python << 'EOF'
import pandas as pd

train = pd.read_csv('dataset_v2.csv')
test = pd.read_csv('external_test_set_isolated.csv')

print(f"Training set: {len(train)} pairs")
print(f"  Type A: {(train['pair_type']=='A').sum()}")
print(f"  Type B: {(train['pair_type']=='B').sum()}")
print(f"  Type C: {(train['pair_type']=='C').sum()}")
print(f"  Type D: {(train['pair_type']=='D').sum()}")
print(f"\nExternal test: {len(test)} pairs")
print(f"  Matches: {(test['match']==1).sum()}")
print(f"  Non-matches: {(test['match']==0).sum()}")
EOF

# 3. Read quality reports
cat quality_report.txt
cat pair_type_report.txt
```

---

## Understanding the Output

### dataset_v2.csv Structure

```
name_a              | Name of entity 1
context_a           | Full description of entity 1
name_b              | Name of entity 2
context_b           | Full description of entity 2
match               | 1 = same disease, 0 = different
source_a            | Data source (Wikipedia, PlantVillage, EPPO, etc.)
source_b            | Data source
pair_type           | A / B / C / D classification
context_quality_a   | good / medium / poor
context_quality_b   | good / medium / poor
name_sim_score      | Jaccard similarity [0-1]
```

### Key Metrics

**Pair Type Distribution:**

```
Type A (Safe Match):        ~400  (14%)  — name AND match agree
Type B (Synonym):           ~300  (12%)  — different names, same disease
Type C (Polysemy):          ~300  (12%)  — similar names, different disease
Type D (Clear Non-Match):   ~1600 (62%)  — name AND match disagree
```

**Context Quality:**

- Good (87-90%): Both pathogen AND symptom/crop mentioned
- Medium (8-10%): Only some disease signals
- Poor (2-3%): Wikipedia list pages, generic text, very short

---

## For Model Training

### Load and Use

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Load training dataset
train_df = pd.read_csv("dataset_v2.csv")

# Create balanced sampler (oversample Type C)
weights = []
for _, row in train_df.iterrows():
    if row["pair_type"] == "C":
        weights.append(3.0)  # Oversample polysemy 3x
    else:
        weights.append(1.0)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_df),
    replacement=True
)

# Use in DataLoader
train_loader = DataLoader(
    dataset,  # your PyTorch Dataset
    sampler=sampler,
    batch_size=32
)
```

### Lambda Supervision Handling

```python
# Don't use lambda for pairs with poor context quality on both sides
both_poor = (
    (df["context_quality_a"] == "poor") &
    (df["context_quality_b"] == "poor")
)

for idx, row in train_df.iterrows():
    if both_poor.iloc[idx]:
        # Use only match loss, skip lambda
        use_lambda = False
    else:
        # Use both match AND lambda
        use_lambda = True
```

### Final Evaluation

```python
# Load external test set (NEVER touch until final eval)
test_df = pd.read_csv("external_test_set_isolated.csv")

# Evaluate only after training is complete
# Report metrics broken down by pair type if possible
```

---

## Troubleshooting

### Q: Step 3 times out or fails

**A:** EPPO API requires valid key. Either:

- Get key from https://data.eppo.int/
- Or skip — script uses mock data as fallback

### Q: Step 5 says "not found: dataset_with_quality.csv"

**A:** Make sure you ran steps 1-2 first. Run all steps with `run_all.py`.

### Q: Want to retry just one step?

**A:** Safe to re-run individual steps. Each overwrites its output file.

### Q: Get different pair type counts than expected?

**A:** Adjust NAME_SIM_THRESHOLD in step2_pair_type_classifier.py (currently 0.25)

---

## Checklist Before Training

- [ ] dataset_v2.csv exists and has ~2,500 rows
- [ ] external_test_set_isolated.csv exists and has ~200 rows
- [ ] pair_type_report.txt shows meaningful Type A/B/C/D distribution
- [ ] quality_report.txt shows most contexts are "good" or "medium"
- [ ] Confirmed you will NOT train on external_test_set_isolated.csv
- [ ] Will oversample Type C pairs during training (3x)
- [ ] Will exclude "both_poor" from lambda supervision

---

## Next Actions

1. **Run pipeline:** `python run_all.py`
2. **Check reports:** Read quality_report.txt and pair_type_report.txt
3. **Verify dataset:** `python -c "import pandas as pd; df=pd.read_csv('dataset_v2.csv'); print(f'{len(df)} pairs ready')"`
4. **Load in training code:** Use dataset_v2.csv with weighted sampler
5. **Evaluate honestly:** Only test on external_test_set_isolated.csv after training completes

---

Good luck building and training! 🌱
