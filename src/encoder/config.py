# encoder/config.py
# =============================================================================
# Single source of truth for all paths, model names, and magic numbers.
# Nothing else in the project should contain hardcoded paths or constants.
# All paths are relative to the PROJECT ROOT (entity_resolution_copy/).
# =============================================================================

import os

# ── Model ─────────────────────────────────────────────────────────────────────

# After fine-tuning (finetune.py), the model is saved here.
# PairAwareAgriLambdaNet uses this as its frozen backbone.
FINETUNED_MODEL_DIR = "plant-disease-encoder"

# Base model — only used if the fine-tuned model does not exist yet.
BASE_MODEL = "all-MiniLM-L6-v2"

# Output dimension of all-MiniLM-L6-v2.
# Every vector produced by the encoder has exactly this many numbers.
# PairAwareAgriLambdaNet input = 4 x EMB_DIM + 2 sim scores = 1538 dims.
EMB_DIM = 384

# ── Data ──────────────────────────────────────────────────────────────────────

# The final cleaned dataset with type_a and type_b fully filled.
CLEAN_DATA_PATH = "dataset_v2_builder/data/dataset_clean.csv"

# Hold-out test set — NEVER train on this.
EXTERNAL_TEST_PATH = "dataset_v2_builder/data/usda_external_test_set.csv"

# Expert-annotated 50-pair validation set.
EXPERT_ANNOTATED_PATH = "dataset_v2_builder/data/expert_validated_50.csv"

# Where encoded vectors are cached so we don't re-encode every run.
# Delete this file to force a fresh encode (e.g. after changing serializer).
VECTORS_PATH = "outputs/encoded_vectors.pt"

# Where the fine-tuned test split is saved after running finetune.py.
TEST_SET_PATH = "outputs/test_set.csv"

# Threshold sweep results from evaluate.py.
THRESHOLD_RESULTS_PATH = "outputs/threshold_results.csv"

# FAISS index files.
FAISS_INDEX_PATH  = "outputs/disease_index.faiss"
FAISS_META_PATH   = "outputs/disease_index_metadata.csv"

# ── Encoding ──────────────────────────────────────────────────────────────────

# Number of rows to encode at once. Lower if you run out of CPU/GPU memory.
ENCODE_BATCH_SIZE = 64

# ── Training ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
TEST_SIZE  = 0.15   # fraction held out as test
VAL_SIZE   = 0.15   # fraction of remaining used for validation

# Drop rows where BOTH context_a and context_b are "poor" quality before training.
FILTER_BOTH_POOR_CONTEXT = True

# ── Evaluation ────────────────────────────────────────────────────────────────

# Threshold values swept during evaluation to find the best F1 cut-off.
import numpy as np
EVAL_THRESHOLDS = np.arange(0.05, 0.95, 0.05).round(3).tolist()

# Top-K candidates returned by the FAISS blocking layer.
FAISS_TOP_K = 10

# ── Entity type normalisation ──────────────────────────────────────────────────

# When type_a or type_b is one of these values, the type_detector is called.
UNKNOWN_TYPE_VALUES = {"unknown", "", "nan", "none"}

# Fallback tag when type is truly unresolvable.
FALLBACK_TAG = "ENTITY"
