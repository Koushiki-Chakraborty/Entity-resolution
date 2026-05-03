"""
Step 4: Fine-tune the Domain-Adapted Encoder
Agricultural Entity Resolution -- AgriLambdaNet

Run from project root:
    python src/encoder/finetune.py

Dataset columns used:
    name_a, context_a, type_a        → entity A
    name_b, context_b, type_b        → entity B
    match                            → hard label (0/1)
    lambda_val                       → soft label (0.0–1.0)
    exclude_from_lambda              → if 1, use hard match label instead
    context_quality_a/b              → filter poor quality pairs if needed
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL   = "all-MiniLM-L6-v2"
# BASE_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"   # biomedical (better)

CSV_PATH     = "dataset_v2_builder/data/dataset_clean.csv"
OUTPUT_DIR   = "plant-disease-encoder"
EPOCHS       = 5
BATCH_SIZE   = 16
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_SEED  = 42

# Set to True to drop rows where both contexts are poor quality
FILTER_POOR_CONTEXT = True


# ─── SERIALIZATION ───────────────────────────────────────────────────────────

def serialize_entity(name: str, context: str, entity_type: str = "DISEASE") -> str:
    """
    Ditto-style domain tag injection.
    Uses actual entity type from dataset (DISEASE, PEST, VIRUS, FUNGUS etc.).
    Falls back to type_detector for unknown/null types.
    """
    from encoder.serializer import serialize_context
    return serialize_context(name, context, entity_type)


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pairs")
    print(f"  Positives (match=1) : {df['match'].sum()}")
    print(f"  Negatives (match=0) : {(df['match']==0).sum()}")
    print(f"  Lambda mean         : {df['lambda_val'].mean():.3f}")
    print(f"  Entity types (a)    : {df['type_a'].value_counts().to_dict()}")
    print(f"  Exclude from lambda : {df['exclude_from_lambda'].sum()} rows")

    if FILTER_POOR_CONTEXT:
        before = len(df)
        df = df[~((df['context_quality_a'] == 'poor') &
                  (df['context_quality_b'] == 'poor'))]
        print(f"  Filtered both-poor context rows: {before - len(df)} removed")
        print(f"  Remaining: {len(df)} pairs")

    return df


def get_label(row) -> float:
    """
    Smart label selection:
    - If exclude_from_lambda=1, use hard binary match label
    - Otherwise use soft lambda_val
    """
    if row["exclude_from_lambda"] == 1:
        return float(row["match"])
    return float(row["lambda_val"])


def make_examples(df: pd.DataFrame) -> list:
    examples = []
    for _, row in df.iterrows():
        # Use actual entity type from dataset — no hardcoding
        text_a = serialize_entity(
            row["name_a"], row["context_a"],
            row.get("type_a", "DISEASE")
        )
        text_b = serialize_entity(
            row["name_b"], row["context_b"],
            row.get("type_b", "DISEASE")
        )
        label = get_label(row)
        examples.append(InputExample(texts=[text_a, text_b], label=label))
    return examples


def split_data(df: pd.DataFrame):
    train_val, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["match"], random_state=RANDOM_SEED
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val, test_size=val_relative,
        stratify=train_val["match"], random_state=RANDOM_SEED
    )
    print(f"\nSplit -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    return train_df, val_df, test_df


# ─── TRAINING ────────────────────────────────────────────────────────────────

def build_evaluator(val_df: pd.DataFrame):
    texts_a, texts_b, labels = [], [], []
    for _, row in val_df.iterrows():
        texts_a.append(serialize_entity(row["name_a"], row["context_a"], row.get("type_a", "DISEASE")))
        texts_b.append(serialize_entity(row["name_b"], row["context_b"], row.get("type_b", "DISEASE")))
        labels.append(get_label(row))
    return EmbeddingSimilarityEvaluator(texts_a, texts_b, labels, name="val")


def train_model(train_df, val_df):
    print(f"\nLoading base model: {BASE_MODEL}")
    model    = SentenceTransformer(BASE_MODEL)
    examples = make_examples(train_df)
    loader   = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn  = losses.CosineSimilarityLoss(model)
    evaluator = build_evaluator(val_df)
    warmup   = int(0.1 * len(loader) * EPOCHS)

    print(f"Training {len(examples)} pairs | {EPOCHS} epochs | warmup {warmup} steps")

    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup,
        output_path=OUTPUT_DIR,
        save_best_model=True,
        show_progress_bar=True,
        evaluation_steps=len(loader),
    )
    print(f"\nBest model saved -> {OUTPUT_DIR}/")
    return model


# ─── SANITY CHECK ────────────────────────────────────────────────────────────

def sanity_check(model: SentenceTransformer, df: pd.DataFrame):
    """Pull real matched and non-matched pairs from train set for sanity check."""
    print("\n--- Sanity check (real pairs from dataset) ---")

    def cos(a, b):
        return F.cosine_similarity(
            torch.tensor(a).float().unsqueeze(0),
            torch.tensor(b).float().unsqueeze(0)
        ).item()

    matched    = df[df["match"] == 1].sample(n=2, random_state=1)
    unmatched  = df[df["match"] == 0].sample(n=2, random_state=1)

    for label, subset in [("MATCH — expect HIGH", matched),
                           ("NON-MATCH — expect LOW", unmatched)]:
        for _, row in subset.iterrows():
            s1  = serialize_entity(row["name_a"], row["context_a"], row.get("type_a", "DISEASE"))
            s2  = serialize_entity(row["name_b"], row["context_b"], row.get("type_b", "DISEASE"))
            sim = cos(model.encode(s1), model.encode(s2))
            print(f"  [{label}]")
            print(f"  '{row['name_a']}'  vs  '{row['name_b']}'")
            print(f"  type: {row.get('type_a','?')} / {row.get('type_b','?')}  →  sim: {sim:.4f}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pathlib
    # Add src/ to path so 'from encoder.xxx import ...' works
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    # Ensure outputs/ directory exists
    pathlib.Path("outputs").mkdir(exist_ok=True)

    df = load_data(CSV_PATH)
    train_df, val_df, test_df = split_data(df)

    test_df.to_csv("outputs/test_set.csv", index=False)
    print("outputs/test_set.csv saved (use in Step 5)\n")

    model = train_model(train_df, val_df)
    sanity_check(model, train_df)

    print("Step 4 complete.")
    print(f"  Model    -> {OUTPUT_DIR}/")
    print("  Test set -> outputs/test_set.csv")
    print("\nNext: run step2_encode_and_verify.py")