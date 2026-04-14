import pandas as pd
import numpy as np

# Load your file
df = pd.read_csv("llm_labeled_pairs.csv")
print(f"Loaded: {len(df)} rows")

# ── FIX 1: Rename columns to match training code ─────────────────────────
df = df.rename(columns={
    'name_a':      'name_1',
    'context_a':   'context_1',
    'name_b':      'name_2',
    'context_b':   'context_2',
    'true_label':  'label',
    'llm_lambda':  'lambda_label',
    # keep: canonical_id_a, canonical_id_b, source_url_a, source_url_b, llm_match
})

# ── FIX 2: Clean up nan strings in context ───────────────────────────────
df['context_1'] = df['context_1'].replace('nan', '').fillna('')
df['context_2'] = df['context_2'].replace('nan', '').fillna('')

# ── FIX 3 (optional): Add small noise to lambda to spread the distribution
# This makes the LambdaEstimator learn finer-grained weights
# Comment this out if you want to keep the original LLM values exactly
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(df))
df['lambda_label'] = (df['lambda_label'] + noise).clip(0.0, 1.0).round(3)

# ── FIX 4: Drop pure-metadata columns training does not need ─────────────
# (optional — keeping them does not hurt, but cleaner without)
# df = df.drop(columns=['source_url_a','source_url_b','llm_match'])

# ── VERIFY ────────────────────────────────────────────────────────────────
print("\nFinal columns:", df.columns.tolist())
print("Label distribution:")
print(df['label'].value_counts())
print("\nLambda stats:")
print(df['lambda_label'].describe())
print("\nMissing values:")
print(df[['name_1', 'context_1', 'name_2', 'context_2',
          'label', 'lambda_label']].isnull().sum())

df.to_csv("training_ready.csv", index=False)
print(f"\nSaved: training_ready.csv ({len(df)} rows)")
print("Your dataset is now ready to train AgriLambdaNet!")
