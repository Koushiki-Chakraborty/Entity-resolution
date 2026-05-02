"""
Step 5: Threshold Sweep + F1 Evaluation
Agricultural Entity Resolution — updated for production dataset

Run:
    python step5_evaluate.py

Requires:
    - ./plant-disease-encoder/   (from Step 4)
    - test_set.csv               (from Step 4)

Output:
    - threshold_results.csv
    - Best threshold + full classification report printed
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR    = "./plant-disease-encoder"
TEST_CSV     = "test_set.csv"
RESULTS_CSV  = "threshold_results.csv"
THRESHOLDS   = np.arange(0.1, 0.9, 0.05).round(2)


# ─── SERIALIZATION (must match Step 4 exactly) ───────────────────────────────

def serialize_entity(name: str, context: str, entity_type: str = "DISEASE") -> str:
    
    tag = entity_type.upper().strip()
    return (
        f"[{tag}] {name} [/{tag}] "
        f"[CONTEXT] {context} [/CONTEXT]"
    )


# ─── EMBED TEST SET ──────────────────────────────────────────────────────────

def embed_test_set(model: SentenceTransformer, df: pd.DataFrame):
    print(f"Encoding {len(df)} test pairs...")

    texts_a = [
        serialize_entity(r["name_a"], r["context_a"], r.get("type_a", "DISEASE"))
        for _, r in df.iterrows()
    ]
    texts_b = [
        serialize_entity(r["name_b"], r["context_b"], r.get("type_b", "DISEASE"))
        for _, r in df.iterrows()
    ]

    vecs_a = model.encode(texts_a, batch_size=32, show_progress_bar=True,
                          convert_to_tensor=True)
    vecs_b = model.encode(texts_b, batch_size=32, show_progress_bar=True,
                          convert_to_tensor=True)

    return F.cosine_similarity(vecs_a, vecs_b).cpu().numpy()


# ─── THRESHOLD SWEEP ─────────────────────────────────────────────────────────

def sweep_thresholds(similarities, true_labels):
    records = []
    for t in THRESHOLDS:
        preds = (similarities >= t).astype(int)
        f1    = f1_score(true_labels, preds, zero_division=0)
        prec  = precision_score(true_labels, preds, zero_division=0)
        rec   = recall_score(true_labels, preds, zero_division=0)
        tp    = int(((preds==1) & (true_labels==1)).sum())
        fp    = int(((preds==1) & (true_labels==0)).sum())
        fn    = int(((preds==0) & (true_labels==1)).sum())
        tn    = int(((preds==0) & (true_labels==0)).sum())
        records.append({
            "threshold": t, "f1": round(f1,4),
            "precision": round(prec,4), "recall": round(rec,4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    results_df = pd.DataFrame(records)
    best_row   = results_df.loc[results_df["f1"].idxmax()]
    return results_df, best_row


# ─── PRINT RESULTS ───────────────────────────────────────────────────────────

def print_sweep_table(results_df, best_threshold):
    print("\n── Threshold Sweep ──────────────────────────────────────")
    print(f"{'Threshold':>10}  {'F1':>6}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 46)
    for _, row in results_df.iterrows():
        marker = "  ← BEST" if row["threshold"] == best_threshold else ""
        print(f"  {row['threshold']:>8.2f}  {row['f1']:>6.4f}  "
              f"{row['precision']:>10.4f}  {row['recall']:>8.4f}{marker}")


def print_final_report(df, similarities, best_threshold):
    true_labels = df["match"].values
    preds       = (similarities >= best_threshold).astype(int)

    print(f"\n── Final Report at threshold = {best_threshold} ──────────────")
    print(classification_report(true_labels, preds,
                                target_names=["no match", "match"]))

    cm = confusion_matrix(true_labels, preds)
    print("Confusion matrix:")
    print(f"                 Predicted NO   Predicted YES")
    print(f"  Actual NO      {cm[0][0]:>12}   {cm[0][1]:>12}")
    print(f"  Actual YES     {cm[1][0]:>12}   {cm[1][1]:>12}")

    # Per entity type breakdown
    print(f"\nBreakdown by entity type:")
    for etype in df["type_a"].dropna().unique():
        mask  = df["type_a"] == etype
        if mask.sum() < 5:
            continue
        sub_true  = true_labels[mask]
        sub_preds = preds[mask]
        sub_f1    = f1_score(sub_true, sub_preds, zero_division=0)
        print(f"  {etype:15s}  n={mask.sum():4d}  F1={sub_f1:.4f}")

    # Similarity distribution
    pos_sims = similarities[true_labels == 1]
    neg_sims = similarities[true_labels == 0]
    print(f"\nSimilarity distribution:")
    print(f"  Matches     (n={len(pos_sims):3d})  "
          f"mean={pos_sims.mean():.4f}  min={pos_sims.min():.4f}  max={pos_sims.max():.4f}")
    print(f"  Non-matches (n={len(neg_sims):3d})  "
          f"mean={neg_sims.mean():.4f}  min={neg_sims.min():.4f}  max={neg_sims.max():.4f}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading model: {MODEL_DIR}")
    model = SentenceTransformer(MODEL_DIR)

    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} pairs  |  "
          f"positives={df['match'].sum()}  negatives={(df['match']==0).sum()}")

    similarities = embed_test_set(model, df)
    true_labels  = df["match"].values

    results_df, best_row = sweep_thresholds(similarities, true_labels)
    best_threshold = float(best_row["threshold"])

    print_sweep_table(results_df, best_threshold)

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nFull results saved → {RESULTS_CSV}")

    print_final_report(df, similarities, best_threshold)

    print(f"\n{'='*50}")
    print(f"  Best threshold : {best_threshold}")
    print(f"  Best F1        : {best_row['f1']}")
    print(f"  Precision      : {best_row['precision']}")
    print(f"  Recall         : {best_row['recall']}")
    print(f"  TP={int(best_row['tp'])}  FP={int(best_row['fp'])}  "
          f"FN={int(best_row['fn'])}  TN={int(best_row['tn'])}")
    print(f"{'='*50}")
    print("\nStep 5 complete.")
    print(f"  Use threshold={best_threshold} in Step 6 (loaded automatically).")
    print("  Next: run step6_blocking.py")