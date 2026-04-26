"""
Before & After Examples - Context Enhancement
==============================================================
This file shows real examples of how contexts were improved
"""

import pandas as pd

df_before = pd.read_csv("data/pairs/training_ready_final.csv")
df_after = pd.read_csv("data/pairs/training_ready_enriched.csv")

print("\n" + "="*100)
print("BEFORE & AFTER CONTEXT EXAMPLES")
print("="*100)

examples = []

# Find rows with significant improvements
for idx in range(len(df_before)):
    name_a = df_before['name_a'].iloc[idx]
    ctx_a_before = df_before['context_a'].iloc[idx]
    ctx_a_after = df_after['context_a'].iloc[idx]
    
    if ctx_a_before != ctx_a_after:
        improvement = len(ctx_a_after) - len(ctx_a_before)
        if improvement > 30:  # Significant improvement
            examples.append({
                'name': name_a,
                'before': ctx_a_before,
                'after': ctx_a_after,
                'improvement': improvement,
                'type': 'context_a'
            })

# Sort by improvement
examples.sort(key=lambda x: x['improvement'], reverse=True)

# Show top 15 examples
for i, ex in enumerate(examples[:15], 1):
    print(f"\n{'─'*100}")
    print(f"Example {i}: {ex['name'].upper()}")
    print(f"{'─'*100}")
    
    print(f"\n📝 BEFORE ({len(ex['before'])} chars):")
    print(f"   {ex['before']}")
    
    print(f"\n✨ AFTER ({len(ex['after'])} chars, +{ex['improvement']} chars):")
    print(f"   {ex['after']}")
    
    # Highlight what was added
    if len(ex['after']) > len(ex['before']):
        added = ex['after'][len(ex['before']):].strip()
        if added:
            print(f"\n💡 NEW INFORMATION ADDED:")
            print(f"   {added}")

print(f"\n{'='*100}")
print(f"EXAMPLES WITH WIKIPEDIA ENRICHMENT")
print(f"{'='*100}")

# Find examples that were enriched via Wikipedia
wiki_examples = []
for idx in range(min(50, len(df_before))):
    name = df_before['name_a'].iloc[idx]
    before = df_before['context_a'].iloc[idx]
    after = df_after['context_a'].iloc[idx]
    
    if before != after and len(after) > 150 and 'http' in str(df_before['source_url_a'].iloc[idx]):
        wiki_examples.append({
            'name': name,
            'before': before,
            'after': after,
            'improvement': len(after) - len(before)
        })

wiki_examples.sort(key=lambda x: x['improvement'], reverse=True)

for i, ex in enumerate(wiki_examples[:10], 1):
    print(f"\n{'─'*100}")
    print(f"Entity {i}: {ex['name'].upper()}")
    print(f"{'─'*100}")
    
    print(f"\n❌ ORIGINAL (Truncated, {len(ex['before'])} chars):")
    print(f"   \"{ex['before'][:100]}{'...' if len(ex['before']) > 100 else ''}\"")
    
    print(f"\n✅ ENHANCED (Complete, {len(ex['after'])} chars):")
    print(f"   \"{ex['after']}\"")

print(f"\n{'='*100}")
print(f"CONTEXT LENGTH DISTRIBUTION")
print(f"{'='*100}")

import numpy as np

before_lens = df_before['context_a'].astype(str).str.len()
after_lens = df_after['context_a'].astype(str).str.len()

print(f"\nBEFORE Enhancement:")
print(f"  Mean:     {before_lens.mean():.1f} chars")
print(f"  Median:   {before_lens.median():.1f} chars")
print(f"  Std Dev:  {before_lens.std():.1f} chars")
print(f"  Min:      {before_lens.min():.0f} chars")
print(f"  Max:      {before_lens.max():.0f} chars")
print(f"  Q1 (25%): {before_lens.quantile(0.25):.1f} chars")
print(f"  Q3 (75%): {before_lens.quantile(0.75):.1f} chars")

print(f"\nAFTER Enhancement:")
print(f"  Mean:     {after_lens.mean():.1f} chars")
print(f"  Median:   {after_lens.median():.1f} chars")
print(f"  Std Dev:  {after_lens.std():.1f} chars")
print(f"  Min:      {after_lens.min():.0f} chars")
print(f"  Max:      {after_lens.max():.0f} chars")
print(f"  Q1 (25%): {after_lens.quantile(0.25):.1f} chars")
print(f"  Q3 (75%): {after_lens.quantile(0.75):.1f} chars")

improvement_pct = ((after_lens.mean() - before_lens.mean()) / before_lens.mean() * 100)
print(f"\n📈 Average Improvement: {improvement_pct:+.1f}%")

print(f"\n{'='*100}\n")
