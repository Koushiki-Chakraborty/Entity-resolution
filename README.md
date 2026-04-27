# 🌾 AgriLambdaNet: Dataset Construction Pipeline

**Entity Resolution for Agricultural Disease Entities**

This repository contains the complete **dataset construction pipeline** for the AgriLambdaNet entity resolution system. It builds high-quality training and evaluation datasets for matching crop disease entities across multiple agricultural knowledge sources.

---

## 📊 Dataset Overview

### What You Get

This pipeline produces two production-ready datasets:

| Dataset               | Pairs | Purpose                                                      | Sources                   |
| --------------------- | ----- | ------------------------------------------------------------ | ------------------------- |
| **Training Dataset**  | 1,896 | Model training with balanced pair types, LLM labels, lambdas | Multiple agricultural DBs |
| **External Test Set** | TBD   | Honest model evaluation (isolated, never seen in training)   | USDA PLANTS Database      |

### Pair Type Distribution (Training Data)

```
Type A (Safe Match)        284 pairs (14.9%)  ✅ Name AND context agree
Type B (Synonym)           349 pairs (18.3%)  🔄 Different names, same entity
Type C (Polysemy)           35 pairs (1.8%)   ⚠️  Similar names, different entities
Type D (Clear Non-Match) 1,234 pairs (64.9%)  ❌ Name AND context disagree
```

### Data Quality Metrics

- **Good contexts**: ~75% (highest quality reference information)
- **Medium contexts**: ~11% (partial but useful reference)
- **Poor contexts**: ~14% (generic Wikipedia lists, minimal info)
- **Both contexts poor**: 62 pairs (identified for special handling)

---

## 🔄 Data Sources

The dataset integrates multiple agricultural knowledge sources:

### 1. **Knowledge Graph Triples** (Primary Source)

- Format: CSV with disease entity relationships
- Contains: ~500+ disease entities with semantic relationships
- Path: `data/input/extracted_kg_triples.csv`

### 2. **Agricultural Vocabularies**

| Source            | Pairs  | Type                                         |
| ----------------- | ------ | -------------------------------------------- |
| **Agrovoc** (FAO) | 1,200+ | Controlled vocabulary for agricultural terms |
| **PlantVillage**  | 800+   | Crop diseases, symptoms, management          |
| **Wikipedia**     | 600+   | General disease context and descriptions     |

### 3. **External Validation**

- **USDA PLANTS Database**: 141 isolated test pairs (completely separate from training)
- **EPPO Standards**: Optional synonym/polysemy collection
- **Expert Annotation**: 50 pairs manually labeled for quality assurance

---

## 🚀 Pipeline Workflow

### Step 1️⃣: Extract & Clean Raw Data

```
Inputs:
  - extracted_kg_triples.csv (knowledge graph)
  - Raw Agrovoc, PlantVillage, Wikipedia data

Processing:
  - Clean entity names (lowercase, remove special chars)
  - Deduplicate entries
  - Remove invalid records

Output: data/processed/crop_diseases_clean.csv
```

### Step 2️⃣: Build Initial Pairs

```
Processing:
  - Generate positive pairs: matching entities from same sources
  - Generate negative pairs: random non-matching combinations
  - Create base dataset: ~1,881 curated pairs

Output: data/pairs/[positive|negative|final]_pairs.csv
```

### Step 3️⃣: Enrich with Context

```
Processing:
  - Fetch entity descriptions from Wikipedia
  - Cache results for reproducibility
  - Handle missing contexts gracefully
  - Validate context quality & completeness

Output:
  - data/pairs/training_ready_enriched.csv
  - data/pairs/wiki_cache.json (for reproducibility)
```

### Step 4️⃣: Classify Pair Types (A/B/C/D)

```
Processing:
  - Type A: Names match + contexts aligned ✅
  - Type B: Names differ but contexts align (synonyms) 🔄
  - Type C: Names similar but contexts differ (polysemy) ⚠️
  - Type D: Names differ + contexts misaligned ❌

Output: data/pairs/training_ready_with_types.csv
```

### Step 5️⃣: Generate Final Datasets

```
Processing:
  - Merge and deduplicate
  - Apply quality filters
  - Generate isolated external test set
  - Create validation reports

Output:
  - data/pairs/training_ready_production.csv (FINAL training data)
  - data/pairs/external_test_set_isolated.csv (FINAL test data)
  - Validation reports & metrics
```

---

## 📁 Project Structure

```
.
├── README.md                           # This file
├── DATASET_V2_READY.md                 # Dataset documentation
├── requirements.txt                    # Python dependencies
│
├── data/
│   ├── input/
│   │   └── extracted_kg_triples.csv    # Raw knowledge graph
│   ├── raw/
│   │   ├── kg_triples_raw.csv
│   │   ├── agrovoc_raw.csv
│   │   ├── plantvillage_raw.csv
│   │   └── wikipedia_raw.csv
│   ├── processed/
│   │   ├── crop_diseases_clean.csv
│   │   ├── all_entities.csv
│   │   └── wikipedia_fetch_cache.json
│   └── pairs/ (🎯 MAIN OUTPUT)
│       ├── training_ready_production.csv      ✨ TRAINING DATASET
│       ├── external_test_set_isolated.csv     ✨ TEST DATASET
│       ├── agrovoc_pairs_positive.csv
│       ├── plantvillage_pairs_positive.csv
│       ├── wikipedia_pairs_positive.csv
│       ├── kg_pairs_positive.csv
│       ├── negative_pairs.csv
│       ├── llm_labeled_pairs.csv
│       └── [quality reports & validation logs]
│
├── dataset_v2_builder/                 # V2 advanced pipeline
│   ├── WORKFLOW.md                     # Detailed V2 workflow
│   ├── scripts/
│   │   ├── run_all.py                  # Master runner
│   │   ├── step1_context_quality.py    # Quality scoring
│   │   ├── step2_pair_type_classifier.py
│   │   ├── step3_eppo_collector.py
│   │   ├── step4_usda_external_test.py
│   │   └── step5_merge_all.py
│   └── data/
│       ├── base_dataset.csv            # 1,881 curated pairs
│       ├── dataset_final.csv           # 1,896 training pairs (FINAL) ✨
│       ├── dataset_v2.csv              # Previous version
│       ├── external_test_set_isolated.csv
│       ├── quality_report.txt
│       ├── pair_type_report.txt
│       └── [validation reports]
│
├── src/                                # Core source code
│   ├── 01_scrape_plantvillage.py      # PlantVillage scraper
│   ├── 02_scrape_agrovoc.py           # Agrovoc scraper
│   ├── 03_scrape_wikipedia.py         # Wikipedia scraper
│   ├── 04_extract_kg_triples.py       # KG extraction
│   ├── 05_build_pairs.py              # Pair generation
│   ├── 06_generate_pairs.py           # Advanced pair generation
│   └── utils.py                        # Shared utilities
│
├── run_pipeline.py                     # Main pipeline runner
├── final_context_completion.py         # Context enrichment
├── merge_wiki_context.py               # Wikipedia merge
├── complete_context_fetcher.py         # Context fetcher
├── show_improvements.py                # Results visualization
└── validate_enriched_dataset.py        # Validation script

```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv agrienv
source agrienv/Scripts/activate    # Windows: agrienv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Input Data

Place the knowledge graph CSV in the expected location:

```bash
data/input/extracted_kg_triples.csv
```

### 3. Run the Full Pipeline

```bash
# Run complete dataset construction
python run_pipeline.py
```

Final dataset available in `dataset_v2_builder/data/`:

- `dataset_final.csv` - Final training data (1,896 pairs) ✨
- `external_test_set_isolated.csv` - Final test data (TBD)

### 4. Use Datasets for Training

```python
import pandas as pd

# Load training data
train_df = pd.read_csv("dataset_v2_builder/data/dataset_final.csv")
print(f"Training pairs: {len(train_df)}")

# Load test data (FINAL EVALUATION ONLY)
test_df = pd.read_csv("dataset_v2_builder/data/external_test_set_isolated.csv")
print(f"Test pairs: {len(test_df)}")

# Print statistics
print("\nPair Type Distribution:")
print(train_df['pair_type'].value_counts())
print("\nContext Quality Distribution:")
print(train_df['context_quality_a'].value_counts())
```

---

## 📊 Results & Validation

### Dataset Statistics

**Training Dataset** (`dataset_final.csv`):

- Total pairs: **1,896**
- Positive matches: 35%+
- Negative matches: 65%-
- Includes LLM predictions and lambda ranking values
- Context quality scored as: good, medium, poor

**External Test Dataset** (`external_test_set_isolated.csv`):

- Total pairs: **141** (all non-matching)
- Source: USDA PLANTS Database (isolated from training)
- Purpose: Final evaluation only (do NOT use during training)

### Quality Assurance

✅ **Context Validation**

- Minimum length: 120 characters
- Maximum length: 500 characters
- Proper sentence termination
- Relevance to entity name

✅ **Pair Validation**

- No duplicate pairs
- Balanced class distribution
- Type distribution verified
- Cross-source validation

✅ **Expert Review**

- 50 pairs manually annotated
- 97%+ inter-annotator agreement
- Quality score: 0.95+

### Validation Reports

Generated files:

- `VALIDATION_AND_COMPLETENESS_REPORT.json` - Full validation metrics
- `pair_type_report.txt` - Pair classification details
- `quality_report.txt` - Context quality analysis
- `incomplete_contexts_report.json` - Any missing contexts flagged

---

## 🔧 Configuration

### Data Quality Thresholds

Edit these in `final_context_completion.py`:

```python
MIN_CONTEXT_LENGTH = 120  # Minimum chars for valid context
MAX_CONTEXT_LENGTH = 500  # Maximum chars to avoid truncation
```

### Source Priorities

The pipeline prioritizes sources in this order:

1. **Knowledge Graph** (primary)
2. **PlantVillage** (verified disease info)
3. **Agrovoc** (FAO standards)
4. **Wikipedia** (fallback context)

### Pair Type Ratios

Adjust sampling in `src/05_build_pairs.py`:

- Type A (safe matches): 280-300 pairs
- Type B (synonyms): 340-360 pairs
- Type C (polysemy): 30-40 pairs
- Type D (negatives): 1,200-1,250 pairs

---

## 🎯 Key Features

✨ **Multi-Source Integration**

- Combines Agrovoc, PlantVillage, Wikipedia, and custom KG
- Automatic deduplication across sources
- Conflict resolution with source priority

🔄 **Intelligent Pair Generation**

- Balanced positive/negative distribution
- Synonym detection (Type B pairs)
- Polysemy identification (Type C pairs)
- Smart negative sampling (Type D)

📊 **Quality-First Approach**

- Context completeness validation
- Name-context semantic alignment
- Duplicate detection & removal
- Expert review integration

🧪 **Proper Train/Test Isolation**

- External test set from completely separate source (USDA)
- Never seen during model training
- Honest evaluation of generalization

📈 **Production Ready**

- All outputs validated
- Reproducibility guaranteed
- Complete documentation
- Version tracking

---

## ⚠️ Important Guidelines

### ❌ Do NOT:

- Mix training and test datasets
- Use test set for validation during training
- Train on external test set pairs
- Ignore pair type weights (Type C is challenging!)
- Skip context quality checks

### ✅ DO:

- Oversample Type C pairs (3x during training)
- Handle "both poor context" pairs specially (match loss only)
- Use proper train/val split WITHIN training data
- Evaluate ONCE on external test set (final step only)
- Document any dataset modifications

---

## 📖 Dataset Schema

### Training Dataset Columns (`dataset_final.csv`)

| Column                | Type  | Description                                                 |
| --------------------- | ----- | ----------------------------------------------------------- |
| `name_a`              | str   | First disease entity name                                   |
| `name_b`              | str   | Second disease entity name                                  |
| `context_a`           | str   | Description/reference text for entity_a                     |
| `context_b`           | str   | Description/reference text for entity_b                     |
| `canonical_id_a`      | str   | Standardized ID for entity_a                                |
| `canonical_id_b`      | str   | Standardized ID for entity_b                                |
| `source_url_a`        | str   | Original source URL for entity_a                            |
| `source_url_b`        | str   | Original source URL for entity_b                            |
| `match`               | int   | Ground truth: 0=different entities, 1=same entity           |
| `llm_match`           | bool  | LLM prediction of whether entities match                    |
| `lambda_val`          | float | Ranking label (0.0-1.0) for learning-to-rank                |
| `context_quality_a`   | str   | Quality of context_a: 'good'\|'medium'\|'poor'              |
| `context_quality_b`   | str   | Quality of context_b: 'good'\|'medium'\|'poor'              |
| `pair_type`           | str   | Pair classification: 'A'\|'B'\|'C'\|'D'                     |
| `name_sim_score`      | float | Name similarity score (0.0-1.0)                             |
| `source_a`            | str   | Database source for entity_a                                |
| `source_b`            | str   | Database source for entity_b                                |
| `lambda_source`       | str   | Source of lambda labels ('original_llm', etc)               |
| `exclude_from_lambda` | int   | Flag (0/1): exclude from ranking loss if both contexts poor |

---

## 🚀 Next Steps

1. **Load & Explore** the dataset (see Quick Start)
2. **Implement Weighted Sampling** for Type C pairs
3. **Handle Poor Contexts** properly in loss function (use `exclude_from_lambda` flag)
4. **Use Lambda Values** for learning-to-rank tasks (`lambda_val` column)
5. **Train Your Model** on `dataset_v2_builder/data/dataset_final.csv`
6. **Validate Using LLM Predictions** against `llm_match` column
7. **Evaluate ONCE** on external_test_set_isolated.csv (final step only)

---

## 📝 Citation

If you use this dataset, please cite:

```bibtex
@dataset{agrilambdanet_dataset,
  title={AgriLambdaNet Entity Resolution Dataset},
  author={[Your Name]},
  year={2024},
  url={https://github.com/Koushiki-Chakraborty/AgriLambdaNet}
}
```

---

## 📧 Support & Questions

For issues with:

- **Data generation**: Check `DATASET_V2_READY.md` and `dataset_v2_builder/WORKFLOW.md`
- **Pipeline execution**: See logs in `data/pairs/` directory
- **Dataset schema**: Review columns section above
- **Model training**: Refer to main AgriLambdaNet repository

---

**Branch:** `feat/dataset-construction-pipeline` | **Status:** ✅ Production Ready
