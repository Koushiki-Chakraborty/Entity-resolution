"""
=============================================================================
Step 5 — Final Dataset Merger
Combine all sources into dataset_v2.csv (training) and external_test_set.csv

WHAT THIS DOES:
    Merges:
      1. dataset_classified.csv (base 1,881 pairs + quality scores + pair types)
      2. eppo_pairs_collected.csv (Type B + C supplements from EPPO)
      3. usda_external_test_set.csv (200 external test pairs)
    
    Creates two outputs:
      - dataset_v2.csv: Training set (2,500 pairs)
        Includes Type A, B, C, D with quality and pair_type columns
      - external_test_set.csv: Test set (200 pairs, isolated)
        Marked as "external_test" for evaluation only

TARGET DISTRIBUTION:
    Training: 2,500 pairs
      - Type A: ~400 pairs (safe matches)
      - Type B: ~300 pairs (synonyms - names mislead)
      - Type C: ~300 pairs (polysemy - must use context)
      - Type D: ~600 pairs (clear non-matches)
    
    External Test: 200 pairs (separate, never trained on)

HOW TO RUN:
    1. Ensure all four previous steps completed successfully
    2. Run: python step5_merge_all.py
    3. Outputs:
         - ../data/dataset_v2.csv (training, 2,500 pairs)
         - ../data/external_test_set.csv (test, 200 pairs)

=============================================================================
"""

import pandas as pd
import os
from pathlib import Path
import sys

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = "../data"
INPUTS = {
    "classified": "dataset_classified.csv",
    "eppo": "eppo_pairs_collected.csv",
    "usda": "usda_external_test_set.csv"
}
TRAINING_OUTPUT = "dataset_v2.csv"
TEST_OUTPUT = "external_test_set_isolated.csv"

TARGET_TRAINING = 2500
TARGET_TEST = 200

def load_file(name):
    """Safely load a CSV, with error handling"""
    path = os.path.join(DATA_DIR, INPUTS[name])
    if not os.path.exists(path):
        print(f"  ⚠ {name} not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        print(f"  ✓ {name}: {len(df)} pairs")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {name}: {e}")
        return pd.DataFrame()

def main():
    print("=" * 70)
    print(" Final Dataset Merger — dataset_v2.csv")
    print("=" * 70)
    
    print(f"\n[1/5] Loading source datasets...")
    
    df_classified = load_file("classified")
    df_eppo = load_file("eppo")
    df_usda = load_file("usda")
    
    # Separate external test set
    print(f"\n[2/5] Separating external test set...")
    
    # Filter USDA external test set
    if "dataset_split" in df_usda.columns:
        external_test = df_usda[df_usda["dataset_split"] == "external_test"].copy()
    else:
        external_test = df_usda.copy()
    
    print(f"  External test pairs: {len(external_test)}")
    
    # Combine training sources (all except USDA external test)
    print(f"\n[3/5] Combining training datasets...")
    
    training_dfs = [df_classified]
    if len(df_eppo) > 0:
        training_dfs.append(df_eppo)
    
    df_training = pd.concat(training_dfs, ignore_index=True)
    print(f"  Combined training pairs: {len(df_training)}")
    
    # Remove duplicates
    print(f"\n[4/5] Deduplicating...")
    
    # Create pair signature for deduplication
    df_training["_pair_sig"] = (
        df_training["name_a"].str.lower() + " | " + 
        df_training["name_b"].str.lower()
    )
    df_training = df_training.drop_duplicates(subset=["_pair_sig"])
    df_training = df_training.drop(columns=["_pair_sig"])
    
    external_test["_pair_sig"] = (
        external_test["name_a"].str.lower() + " | " + 
        external_test["name_b"].str.lower()
    )
    external_test = external_test.drop_duplicates(subset=["_pair_sig"])
    external_test = external_test.drop(columns=["_pair_sig"])
    
    print(f"  After dedup: {len(df_training)} training pairs, {len(external_test)} test pairs")
    
    # Ensure required columns exist
    print(f"\n[5/5] Finalizing and saving...")
    
    # Add missing columns if needed
    required_train_cols = [
        "name_a", "context_a", "name_b", "context_b", "match", 
        "source_a", "source_b", "pair_type", 
        "context_quality_a", "context_quality_b", "name_sim_score"
    ]
    
    for col in required_train_cols:
        if col not in df_training.columns:
            if col == "source_a":
                df_training[col] = df_training.get("source_a", "Unknown")
            elif col == "source_b":
                df_training[col] = df_training.get("source_b", "Unknown")
            else:
                df_training[col] = None
    
    required_test_cols = [
        "name_a", "context_a", "name_b", "context_b", "match",
        "source_a", "source_b", "context_quality_a", "context_quality_b"
    ]
    
    for col in required_test_cols:
        if col not in external_test.columns:
            external_test[col] = None
    
    # Save training set
    training_path = os.path.join(DATA_DIR, TRAINING_OUTPUT)
    df_training.to_csv(training_path, index=False)
    print(f"  ✅ Training set: {training_path}")
    print(f"     {len(df_training)} pairs")
    
    # Save external test set
    test_path = os.path.join(DATA_DIR, TEST_OUTPUT)
    external_test.to_csv(test_path, index=False)
    print(f"  ✅ External test set: {test_path}")
    print(f"     {len(external_test)} pairs")
    
    # Report pair type distribution
    if "pair_type" in df_training.columns:
        print(f"\n── Training Set Pair Type Distribution ──────")
        counts = df_training["pair_type"].value_counts().sort_index()
        targets = {"A": 400, "B": 300, "C": 300, "D": 600}
        for pt in ["A", "B", "C", "D"]:
            n = counts.get(pt, 0)
            tgt = targets.get(pt, 0)
            status = "✓" if n >= tgt * 0.8 else "⚠"
            print(f"  {status} Type {pt}: {n:4d} / ~{tgt} target")
    
    # Report context quality distribution
    if "context_quality_a" in df_training.columns:
        print(f"\n── Context Quality Distribution ────────────")
        q_a = df_training["context_quality_a"].value_counts()
        q_b = df_training["context_quality_b"].value_counts()
        print(f"  context_a: Good={q_a.get('good',0)}, Medium={q_a.get('medium',0)}, Poor={q_a.get('poor',0)}")
        print(f"  context_b: Good={q_b.get('good',0)}, Medium={q_b.get('medium',0)}, Poor={q_b.get('poor',0)}")
    
    print(f"\n" + "=" * 70)
    print(f" ✅ Dataset v2 Complete!")
    print(f" Training: {training_path}")
    print(f" Testing:  {test_path}")
    print(f"=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Check data/quality_report.txt and data/pair_type_report.txt")
    print(f"  2. Train AgriLambdaNet on dataset_v2.csv")
    print(f"  3. Evaluate ONLY on external_test_set_isolated.csv")
    print(f"  4. Do NOT touch the external test set during development")


if __name__ == "__main__":
    main()
