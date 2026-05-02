import pandas as pd
import sys, io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_csv("../data/dataset_augmented.csv")

# 7 rows where LLM said match=1 but ground truth is match=0
conflicts = df[
    (df["lambda_source"] == "llm_labeled_type_c") &
    (df["llm_match"] == 1)
]

print(f"=== LLM DISAGREEMENTS ({len(conflicts)} rows) ===")
print("These are rows where match=0 (ground truth) but llm_match=1")
print("Review: are these genuine polysemy errors or real aliases?\n")

for i, (_, row) in enumerate(conflicts.iterrows(), 1):
    print(f"[{i}] name_a    : {row['name_a']}")
    print(f"    name_b    : {row['name_b']}")
    print(f"    lambda    : {row['lambda_val']:.3f}")
    print(f"    ctx_a     : {str(row['context_a'])[:100]}...")
    print(f"    ctx_b     : {str(row['context_b'])[:100]}...")
    print()
