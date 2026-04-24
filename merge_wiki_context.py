import pandas as pd

# Load the file
df = pd.read_csv(r"data\pairs\training_ready_with_wikipedia.csv")

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print()

# --- Merge wiki context into context_a ---
# Replace context_a with wiki_context_a where has_wiki_a is True
mask_a = df["has_wiki_a"] == True
df.loc[mask_a, "context_a"] = df.loc[mask_a, "wiki_context_a"]

# --- Merge wiki context into context_b ---
# Replace context_b with wiki_context_b where has_wiki_b is True
mask_b = df["has_wiki_b"] == True
df.loc[mask_b, "context_b"] = df.loc[mask_b, "wiki_context_b"]

print(f"Replaced context_a for {mask_a.sum()} rows (has_wiki_a=True)")
print(f"Replaced context_b for {mask_b.sum()} rows (has_wiki_b=True)")
print()

# --- Report rows where wiki data was NOT found ---
no_wiki_a = df[df["has_wiki_a"] == False][["name_a", "context_a"]].drop_duplicates(subset="name_a")
no_wiki_b = df[df["has_wiki_b"] == False][["name_b", "context_b"]].drop_duplicates(subset="name_b")

print("=" * 60)
print(f"Entities with NO Wikipedia data found for name_a ({len(no_wiki_a)} unique):")
print("=" * 60)
for _, row in no_wiki_a.iterrows():
    print(f"  - {row['name_a']}")

print()
print("=" * 60)
print(f"Entities with NO Wikipedia data found for name_b ({len(no_wiki_b)} unique):")
print("=" * 60)
for _, row in no_wiki_b.iterrows():
    print(f"  - {row['name_b']}")

# --- Drop the 4 extra wiki columns ---
df = df.drop(columns=["wiki_context_a", "has_wiki_a", "wiki_context_b", "has_wiki_b"])

# --- Verify final columns ---
print()
print("=" * 60)
print(f"Final columns ({len(df.columns)}): {list(df.columns)}")
print(f"Final row count: {len(df)}")

# --- Save the result ---
output_path = r"data\pairs\training_ready_with_wikipedia.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
