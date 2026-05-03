"""
Step 6: FAISS Blocking Layer + Interactive Entity Resolution
Agricultural Entity Resolution — updated for production dataset

Run:
    pip install faiss-cpu
    python step6_blocking.py

Requires:
    - ./plant-disease-encoder/
    - Dataset/dataset_production_ready.csv
    - test_set.csv
    - threshold_results.csv      (from Step 5)
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR      = "plant-disease-encoder"
CSV_PATH       = "dataset_v2_builder/data/dataset_clean.csv"
TEST_CSV       = "outputs/test_set.csv"
THRESHOLD_CSV  = "outputs/threshold_results.csv"
INDEX_PATH     = "outputs/disease_index.faiss"
METADATA_CSV   = "outputs/disease_index_metadata.csv"
TOP_K          = 10


# ─── LOAD BEST THRESHOLD FROM STEP 5 ─────────────────────────────────────────

def load_best_threshold() -> float:
    if not os.path.exists(THRESHOLD_CSV):
        raise FileNotFoundError(
            f"{THRESHOLD_CSV} not found. Run step5_evaluate.py first."
        )
    df  = pd.read_csv(THRESHOLD_CSV)
    row = df.loc[df["f1"].idxmax()]
    threshold = float(row["threshold"])
    print(f"Threshold loaded from {THRESHOLD_CSV}")
    print(f"  Best F1        : {row['f1']:.4f}")
    print(f"  Best threshold : {threshold}")
    return threshold


# ─── SERIALIZATION (must match Steps 4 & 5 exactly) ──────────────────────────

def serialize_entity(name: str, context: str, entity_type: str = "DISEASE") -> str:
    from encoder.serializer import serialize_context
    return serialize_context(name, context, entity_type)


# ─── KNOWLEDGE BASE ──────────────────────────────────────────────────────────

def build_knowledge_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all unique canonical entities.
    Now includes entity type (disease, pest, virus, fungus etc.)
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            "canonical_id": row["canonical_id_a"],
            "name":         row["name_a"],
            "context":      row["context_a"],
            "entity_type":  row.get("type_a", "disease"),
        })
    for _, row in df.iterrows():
        records.append({
            "canonical_id": row["canonical_id_b"],
            "name":         row["name_b"],
            "context":      row["context_b"],
            "entity_type":  row.get("type_b", "disease"),
        })

    kb = pd.DataFrame(records)
    kb["ctx_len"] = kb["context"].str.len()
    kb = (kb.sort_values("ctx_len", ascending=False)
            .drop_duplicates(subset="canonical_id")
            .drop(columns="ctx_len")
            .reset_index(drop=True))

    print(f"Knowledge base: {len(kb)} unique canonical entities")
    print(f"  Type breakdown: {kb['entity_type'].value_counts().to_dict()}")
    return kb


# ─── FAISS INDEX ─────────────────────────────────────────────────────────────

def build_index(model, kb):
    print(f"Encoding {len(kb)} entities...")
    texts = [
        serialize_entity(r["name"], r["context"], r["entity_type"])
        for _, r in kb.iterrows()
    ]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors")
    return index


def save_index(index, kb):
    faiss.write_index(index, INDEX_PATH)
    kb.to_csv(METADATA_CSV, index=False)
    print(f"Saved → {INDEX_PATH}, {METADATA_CSV}")


def load_index():
    index = faiss.read_index(INDEX_PATH)
    kb    = pd.read_csv(METADATA_CSV)
    print(f"Index loaded: {index.ntotal} vectors")
    return index, kb


# ─── RESOLVE ─────────────────────────────────────────────────────────────────

def resolve_entity(name, context, model, index, kb, threshold,
                   entity_type="DISEASE"):
    query_vec = model.encode(
        serialize_entity(name, context, entity_type),
        normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32").reshape(1, -1)

    scores, indices = index.search(query_vec, TOP_K)
    candidates = [
        {
            "canonical_id": kb.iloc[idx]["canonical_id"],
            "name":         kb.iloc[idx]["name"],
            "entity_type":  kb.iloc[idx]["entity_type"],
            "score":        round(float(score), 4),
        }
        for score, idx in zip(scores[0], indices[0]) if idx != -1
    ]

    best = candidates[0] if candidates else None
    if best and best["score"] >= threshold:
        return {"matched": True, "canonical_id": best["canonical_id"],
                "matched_name": best["name"], "entity_type": best["entity_type"],
                "score": best["score"], "candidates": candidates}
    else:
        return {"matched": False, "canonical_id": None,
                "matched_name": None, "entity_type": None,
                "score": best["score"] if best else 0.0,
                "candidates": candidates}


# ─── EVALUATE ────────────────────────────────────────────────────────────────

def evaluate_pipeline(model, index, kb, threshold):
    print("\n── Pipeline evaluation on test_set.csv ──────────────────")
    df      = pd.read_csv(TEST_CSV)
    pos     = df[df["match"] == 1]
    correct = sum(
        1 for _, row in pos.iterrows()
        if (r := resolve_entity(
                row["name_a"], row["context_a"], model, index, kb,
                threshold, row.get("type_a", "DISEASE")
            ))["matched"] and r["canonical_id"] == row["canonical_id_a"]
    )
    recall = correct / len(pos) if len(pos) else 0
    print(f"  Positive pairs     : {len(pos)}")
    print(f"  Correctly resolved : {correct}")
    print(f"  Pipeline recall    : {recall:.4f}")

    # Per entity type breakdown
    print(f"\n  Breakdown by entity type:")
    for etype in df["type_a"].dropna().unique():
        sub = df[(df["match"] == 1) & (df["type_a"] == etype)]
        if len(sub) < 3:
            continue
        sub_correct = sum(
            1 for _, row in sub.iterrows()
            if (r := resolve_entity(
                    row["name_a"], row["context_a"], model, index, kb,
                    threshold, row.get("type_a", "DISEASE")
                ))["matched"] and r["canonical_id"] == row["canonical_id_a"]
        )
        print(f"    {etype:15s}  n={len(sub):3d}  correct={sub_correct:3d}  "
              f"recall={sub_correct/len(sub):.4f}")


# ─── INTERACTIVE LOOP ────────────────────────────────────────────────────────

def interactive_loop(model, index, kb, threshold):
    print("\n── Interactive Entity Resolution ────────────────────────")
    print("  Supported types: DISEASE, PEST, VIRUS, FUNGUS, BACTERIA, PLANT")
    print("  Press Enter with empty name to quit.\n")

    while True:
        name = input("  Entity name     : ").strip()
        if not name:
            print("  Exiting.")
            break

        context = input("  Context         : ").strip()
        if not context:
            print("  Context cannot be empty. Try again.\n")
            continue

        # NEW
        entity_type = input("  Entity type (DISEASE/PEST/VIRUS/FUNGUS/BACTERIA) [DISEASE] : ").strip().upper()
        if not entity_type:
            entity_type = "DISEASE"

        result = resolve_entity(name, context, model, index, kb,
                                threshold, entity_type)

        print()
        if result["matched"]:
            print(f"  Status         : MATCHED")
            print(f"  canonical_id   : {result['canonical_id']}")
            print(f"  Matched name   : {result['matched_name']}")
            print(f"  Matched type   : {result['entity_type']}")
            print(f"  Score          : {result['score']}")
        else:
            print(f"  Status         : NEW ENTITY")
            print(f"  Best score     : {result['score']} (below threshold {threshold})")
            print(f"  Action         : Flag for review / add to knowledge base")

        print(f"\n  Top candidates:")
        for i, c in enumerate(result["candidates"][:5], 1):
            print(f"    {i}. [{c['entity_type']:10s}] "
                  f"{c['name']!r:35s}  score={c['score']}")
        print()


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    pathlib.Path("outputs").mkdir(exist_ok=True)

    # 1. Load threshold from Step 5 -- no hardcoding
    threshold = load_best_threshold()

    # 2. Load model
    print(f"\nLoading model: {MODEL_DIR}")
    model = SentenceTransformer(MODEL_DIR)

    # 3. Build knowledge base with entity types
    df = pd.read_csv(CSV_PATH)
    kb = build_knowledge_base(df)

    # 4. Build or load FAISS index
    if os.path.exists(INDEX_PATH):
        print(f"Loading existing index from {INDEX_PATH}")
        index, kb = load_index()
    else:
        index = build_index(model, kb)
        save_index(index, kb)

    # 5. Evaluate pipeline
    evaluate_pipeline(model, index, kb, threshold)

    # 6. Interactive loop — accepts any entity type
    interactive_loop(model, index, kb, threshold)

    print("\nStep 6 complete.")
    print(f"  FAISS index → {INDEX_PATH}  ({index.ntotal} entities)")
    print(f"  Threshold   → {threshold}  (from {THRESHOLD_CSV})")