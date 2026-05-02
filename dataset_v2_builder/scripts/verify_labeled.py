import pandas as pd
import sys, io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_csv("../data/dataset_augmented.csv")

print("=== FINAL DATASET STATUS ===")
print(f"Total rows        : {len(df)}")
print()
print("lambda_source breakdown:")
print(df["lambda_source"].value_counts().to_string())
print()
print(f"lambda_val NaN    : {df['lambda_val'].isna().sum()}")
print(f"llm_match  NaN    : {df['llm_match'].isna().sum()}")
print()
print("pair_type distribution:")
print(df["pair_type"].value_counts().to_string())
print()

labeled = df[df["lambda_source"] == "llm_labeled_type_c"]
print(f"Newly labeled rows : {len(labeled)}")
print(f"  llm_match=0 (correct no-match) : {(labeled['llm_match']==0).sum()}")
print(f"  llm_match=1 (LLM said match!)  : {(labeled['llm_match']==1).sum()}")
print()

print("Lambda distribution (full dataset, no NaN):")
lam = df["lambda_val"].dropna()
for lo, hi in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)]:
    n = ((lam >= lo) & (lam < hi)).sum()
    bar = "#" * (n // 20)
    print(f"  [{lo:.1f}-{hi:.1f}): {n:5d} ({n/len(lam)*100:5.1f}%)  {bar}")
print()

print("llm_labeled_type_c lambda distribution:")
lam2 = labeled["lambda_val"].dropna()
for lo, hi in [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,1.01)]:
    n = ((lam2 >= lo) & (lam2 < hi)).sum()
    print(f"  [{lo:.2f}-{hi:.2f}): {n:4d}")
