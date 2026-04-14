# Crop Disease Entity Resolution — Dataset Construction Pipeline

A research pipeline for constructing a high-quality, LLM-annotated **entity resolution dataset** for crop and plant diseases. The dataset is designed for training and evaluating entity resolution / record linkage models in the agricultural domain, with dual supervision: binary match labels and continuous _lambda_ weighting signals derived from LLM knowledge distillation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Pipeline Steps](#3-pipeline-steps)
4. [Dataset: `crop_diseases_clean.csv`](#4-dataset-crop_diseases_cleancsv)
5. [Dataset: `llm_labeled_pairs.csv`](#5-dataset-llm_labeled_pairscsv)
6. [Dataset Statistics & Label Analysis](#6-dataset-statistics--label-analysis)
7. [Lambda (λ) Distribution Analysis](#7-lambda-λ-distribution-analysis)
8. [LLM Agreement Analysis](#8-llm-agreement-analysis)
9. [Training-Ready Dataset: `training_ready.csv`](#9-training-ready-dataset-training_readycsv)
10. [Quickstart](#10-quickstart)
11. [Canonical ID Design](#11-canonical-id-design)
12. [Requirements](#12-requirements)
13. [Notes](#13-notes)

---

## 1. Project Overview

This pipeline constructs a labeled dataset of **crop and plant disease entity pairs** for supervised entity resolution. The pipeline proceeds in two major phases:

**Phase 1 — Entity Collection & Curation**
Multi-source disease name variants are scraped, deduplicated, and grouped under unified canonical identifiers. Sources include PlantVillage, AGROVOC (FAO), Wikipedia, and Knowledge Graph (KG) triples.

**Phase 2 — LLM Knowledge Distillation (Step 6)**
A large language model (OpenAI GPT) acts as a "teacher" to annotate each disease name pair with:
- `llm_match` — whether the two names refer to the same crop disease
- `llm_lambda` (λ) — a continuous value in [0, 1] capturing whether the matching decision was name-driven (λ ≈ 1.0) or context-driven (λ ≈ 0.0)

These auxiliary signals are used to train **AgriLambdaNet**, a model that learns to dynamically blend name similarity and contextual semantics when resolving agricultural entity mentions.

---

## 2. Project Structure

```
entity_resolution/
|
|-- run_pipeline.py              # Entry point: runs the full pipeline end-to-end
|-- requirements.txt             # Python dependencies
|-- .env                         # API keys / environment config (not committed)
|
|-- src/                         # Pipeline source scripts (run in order)
|   |-- 01_scrape_plantvillage.py   # Step 1: Extract diseases from PlantVillage
|   |-- 02_scrape_agrovoc.py        # Step 2: Extract disease terms from AGROVOC (FAO)
|   |-- 03_scrape_wikipedia.py      # Step 3: Scrape disease context from Wikipedia
|   |-- 04_extract_kg_triples.py    # Step 4: Extract entities from KG triples
|   |-- 05_build_pairs.py           # Step 5: Build positive + negative training pairs
|   |-- 06_generate_pairs.py        # Step 6: LLM knowledge distillation (dual labeling)
|   `-- utils.py                    # Shared utilities (normalisation, deduplication)
|
|-- data/
|   |-- raw/                     # Raw scraped data (one CSV per source)
|   |   |-- plantvillage_raw.csv
|   |   |-- agrovoc_raw.csv
|   |   |-- wikipedia_raw.csv
|   |   `-- kg_triples_raw.csv
|   |
|   |-- processed/               # Cleaned, deduplicated entity datasets
|   |   |-- all_entities.csv           # Master entity list (all sources merged)
|   |   |-- crop_diseases_clean.csv    # Final curated crop disease dataset (USE THIS)
|   |   `-- wikipedia_fetch_cache.json # Wikipedia API fetch cache (speeds up re-runs)
|   |
|   |-- pairs/                   # Pair files and training datasets
|   |   |-- llm_labeled_pairs.csv      # LLM-annotated pairs with lambda signals (USE THIS)
|   |   |-- training_ready.csv         # Final training dataset (column-renamed, cleaned)
|   |   |-- final_dataset.csv          # Complete ML training dataset (pre-LLM baseline)
|   |   |-- positive_pairs.csv         # Matching pairs only (label=1)
|   |   |-- negative_pairs.csv         # Non-matching pairs (label=0)
|   |   |-- clean_for_training.py      # Script: prepares llm_labeled_pairs → training_ready
|   |   |-- plantvillage_pairs_positive.csv
|   |   |-- agrovoc_pairs_positive.csv
|   |   |-- wikipedia_pairs_positive.csv
|   |   `-- kg_pairs_positive.csv
|   |
|   `-- input/                   # Input files supplied by the user
|       `-- extracted_kg_triples.csv   # KG triples (required to run pipeline)
|
`-- agrienv/                     # Python virtual environment (not committed)
```

---

## 3. Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_scrape_plantvillage.py` | Extracts 275 crop disease entries from the PlantVillage dataset |
| 2 | `02_scrape_agrovoc.py` | Queries AGROVOC for agricultural disease vocabulary terms (FAO) |
| 3 | `03_scrape_wikipedia.py` | Fetches disease context paragraphs from Wikipedia articles |
| 4 | `04_extract_kg_triples.py` | Parses `extracted_kg_triples.csv` and extracts disease entities from KG |
| 5 | `05_build_pairs.py` | Merges sources, deduplicates, generates positive + negative pairs |
| 6 | `06_generate_pairs.py` | **LLM distillation** — labels each pair with `llm_match` and `llm_lambda` via OpenAI |
| — | `clean_for_training.py` | Post-processes `llm_labeled_pairs.csv` → `training_ready.csv` |

### Step 6 — LLM Knowledge Distillation

`06_generate_pairs.py` queries an OpenAI model (default: `gpt-4o-mini`) for every disease name pair using a minimal two-task prompt (~130 tokens per call):

```text
A: "late blight" | caused by Phytophthora infestans, affects potato and tomato
B: "tomato blight" | destructive oomycete disease of solanaceous crops

1. match: true if A and B are the same crop disease (aliases/synonyms count).
2. lambda: 0.0=context drove decision, 1.0=name alone was enough.
JSON only: {"match": true, "lambda": 0.7}
```

**Two execution modes are supported:**

| Mode | Flag | Description | Cost |
|------|------|-------------|------|
| Batch (recommended) | `--mode batch` | Submits all pairs to OpenAI Batch API; retrieve when complete (~1h) | ~50% cheaper |
| Live | `--mode live` | Real-time, pair-by-pair; includes checkpoint/resume and rate-limit handling | 2× cost |

**Estimated cost for 1,881 pairs using Batch API:** ~$0.02–$0.04 (gpt-4o-mini).

```bash
# Submit batch job
python src/06_generate_pairs.py --mode batch

# Retrieve results (run after batch completes)
python src/06_generate_pairs.py --mode retrieve --batch-id batch_abc123

# Test with 10 pairs (no API cost)
python src/06_generate_pairs.py --mode live --dry-run 10
```

---

## 4. Dataset: `crop_diseases_clean.csv`

The primary curated entity dataset. Every row is a **surface form** (name variant) of a crop disease.

| Column | Description |
|--------|-------------|
| `name` | Normalised surface form (lowercase, stripped) |
| `canonical_id` | Slug ID shared by all aliases of the same disease |
| `context` | Descriptive text (Wikipedia extract or PlantVillage description) |
| `source_url` | Source URL for provenance |

**Key statistics (v4):**
- 490 unique disease name variants
- 270 unique canonical disease groups
- 69 synonym groups with 2+ aliases (→ positive training pairs)
- 38 synonym groups with 4+ aliases (→ rich training signal)

**Sources:**
- [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 275 verified crop disease entries
- [Wikipedia](https://en.wikipedia.org) — 34 diseases with full article context
- [AGROVOC](http://aims.fao.org/aos/agrovoc) — FAO agricultural vocabulary (filtered)

---

## 5. Dataset: `llm_labeled_pairs.csv`

The LLM-annotated pair dataset. Each row is a **pair of disease name mentions** with a binary ground-truth label and dual LLM-derived supervision signals.

| Column | Description |
|--------|-------------|
| `name_a`, `name_b` | Disease name surface forms |
| `context_a`, `context_b` | Contextual descriptions (truncated to 120 chars) |
| `canonical_id_a`, `canonical_id_b` | Canonical IDs for each name |
| `source_url_a`, `source_url_b` | Provenance URLs |
| `true_label` | **Ground truth**: `1` = same disease, `0` = different disease |
| `llm_match` | LLM prediction: `True` = same disease, `False` = different |
| `llm_lambda` | λ ∈ [0.0, 1.0] — how name-driven the LLM's decision was |

---

## 6. Dataset Statistics & Label Analysis

The following table summarises the composition of `llm_labeled_pairs.csv`:

| Statistic | Value |
|-----------|-------|
| Total labeled pairs | 1,881 |
| Positive pairs (label = 1) | 627 |
| Negative pairs (label = 0) | 1,254 |
| Positive-to-negative ratio | 1 : 2 |
| LLM label agreement | 1,842 / 1,881 (97.9%) |

The dataset is constructed with a **1:2 positive-to-negative ratio**, reflecting a realistic imbalance encountered in production entity resolution systems while keeping the training signal tractable.

---

## 7. Lambda (λ) Distribution Analysis

The `llm_lambda` field captures the LLM's decision rationale as a continuous score:

| λ Range | Interpretation | Count |
|---------|---------------|-------|
| [0.0, 0.3) | Context-driven match | 1,231 |
| [0.3, 0.7) | Balanced (name + context) | 0 |
| [0.7, 1.0] | Name-driven match | 650 |
| **Mean λ** | — | **0.252** |

The distribution reveals a **striking bimodal pattern**: 1,231 pairs are context-driven (λ ∈ [0.0, 0.3)) and 650 are name-driven (λ ∈ [0.7, 1.0]), with **no pairs** in the balanced range (λ ∈ [0.3, 0.7)). This suggests that for agricultural disease entities, matching signals are rarely ambiguous — they are driven either strongly by contextual semantics or strongly by surface name similarity.

This bimodal structure motivates the design of **AgriLambdaNet**, which uses λ as a dynamic gating signal to weight the contribution of name-level and context-level encoders, rather than treating them as fixed or equally weighted.

> **Note on noise injection:** The `clean_for_training.py` script optionally adds small Gaussian noise (σ = 0.05) to λ values before training. This is a regularisation measure intended to prevent the model from over-fitting to the hard 0/1 LLM λ outputs and to provide a more continuous training signal.

---

## 8. LLM Agreement Analysis

Of the 1,881 pairs, the LLM agreed with the curated ground-truth labels in **1,842 cases**, yielding an agreement rate of **97.9%**.

The 39 disagreements arise primarily from differences in semantic interpretation: certain pairs correspond to closely related but biologically distinct entities (e.g., different species within the same genus or genus-level groupings versus specific pathovar designations), where the LLM may infer similarity while the curated dataset treats them as distinct.

**Label authority:** In this work, the **curated labels are treated as ground truth**, as they originate from authoritative data sources including [AGROVOC](http://aims.fao.org/aos/agrovoc) (FAO). The LLM outputs (`llm_match`) are used solely to provide auxiliary supervision in the form of weighting signals (λ), not to override the primary binary labels.

**Scope note:** The dataset intentionally includes a small number of non-crop-related biological entities (e.g., animal or viral diseases), which were retained to improve model robustness and generalization across a broader range of agricultural and biological entity types.

---

## 9. Training-Ready Dataset: `training_ready.csv`

`data/pairs/clean_for_training.py` post-processes `llm_labeled_pairs.csv` into a training-ready format:

| Transformation | Detail |
|----------------|--------|
| Column rename | `name_a/b` → `name_1/2`; `context_a/b` → `context_1/2`; `true_label` → `label`; `llm_lambda` → `lambda_label` |
| Context cleaning | Removes `"nan"` string artefacts, fills empty contexts with `""` |
| Lambda noise | Optional Gaussian noise (σ = 0.05, seed = 42) clipped to [0, 1] |
| Output | `data/pairs/training_ready.csv` |

```python
import pandas as pd

# Load training dataset
df = pd.read_csv("data/pairs/training_ready.csv")

X = df[["name_1", "name_2", "context_1", "context_2", "lambda_label"]]
y = df["label"]   # 1 = match, 0 = non-match

print(f"Pairs: {len(df)} | Positives: {y.sum()} | Negatives: {(y==0).sum()}")
print(f"Lambda mean: {df['lambda_label'].mean():.3f}")
```

---

## 10. Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv agrienv
agrienv\Scripts\activate          # Windows
# source agrienv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key in .env
#    OPENAI_API_KEY=sk-...

# 4. Place KG triples input file
#    data/input/extracted_kg_triples.csv

# 5. Run Phases 1–5 (entity collection + pair generation)
python run_pipeline.py

# 6. Run LLM distillation (Step 6) — submit batch job
python src/06_generate_pairs.py --mode batch
#    Then, after batch completes (~1h):
python src/06_generate_pairs.py --mode retrieve --batch-id <BATCH_ID>

# 7. Prepare the training-ready CSV
python data/pairs/clean_for_training.py
```

**Primary output files:**
- `data/processed/crop_diseases_clean.csv` — curated entity dataset
- `data/pairs/llm_labeled_pairs.csv` — LLM-annotated pairs with λ signals
- `data/pairs/training_ready.csv` — final training dataset for AgriLambdaNet

---

## 11. Canonical ID Design

Every disease alias shares a `canonical_id` with all other aliases of the same disease. This enables supervised entity resolution without requiring manual pairwise annotation:

```
canonical_id: late_blight
  -> "late blight", "lb", "tomato late blight", "late blight of tomato",
     "phytophthora blight", "tomato blight", "potato late blight" (12 aliases)

canonical_id: banana_fusarium_wilt
  -> "panama disease", "fusarium wilt", "tr4", "foc", "banana wilt",
     "fusarium wilt of banana", "banana panama disease" (8 aliases)

canonical_id: citrus_greening_hlb
  -> "citrus greening", "huanglongbing", "hlb", "yellow dragon disease",
     "citrus hlb", "citrus haunglongbing" (10 aliases)
```

Pairs with the **same** `canonical_id` → `label=1` (positive match)
Pairs with **different** `canonical_id` → `label=0` (non-match)

---

## 12. Requirements

See `requirements.txt`. Core dependencies:

```
pandas
requests
beautifulsoup4
numpy
scikit-learn
openai
python-dotenv
```

---

## 13. Notes

- The `agrienv/` virtual environment folder is excluded from version control (see `.gitignore`)
- The Wikipedia fetch cache (`wikipedia_fetch_cache.json`) is kept in `data/processed/` to avoid re-fetching on pipeline re-runs
- Raw data files in `data/raw/` are the immutable source of truth — never edit them manually
- The OpenAI Batch API is strongly recommended over live mode for cost efficiency (~$0.02–$0.04 for the full 1,881-pair job)
- Batch job metadata (`batch_input.jsonl`, `pairs_metadata.json`, `batch_id.txt`) is persisted in `data/pairs/batch_files/` to support retrieval from any machine
