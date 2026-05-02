import pandas as pd
import numpy as np

print("\n" + "="*80)
print("ANALYZING AUGMENTED DATASET FOR THE 5 KEY PROBLEMS")
print("="*80)

df = pd.read_csv('dataset_v2_builder/data/dataset_augmented.csv')

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# PROBLEM 1: Type C (hard negatives) is critically small
print("\n" + "="*80)
print("PROBLEM 1: TYPE C DISTRIBUTION (Hard Negatives)")
print("="*80)
type_counts = df['pair_type'].value_counts()
print("\nPair Type Distribution:")
print(type_counts)
print(f"\nType C rows: {type_counts.get('C', 0)}")
print(f"Target: 300-500+, Current: {type_counts.get('C', 0)}")
if type_counts.get('C', 0) < 100:
    print("❌ CRITICAL: Type C is still too small!")
elif type_counts.get('C', 0) < 300:
    print("⚠️  WARNING: Type C is improving but still below target")
else:
    print("✅ GOOD: Type C is at target level")

# PROBLEM 2: Lambda values skewed toward 0
print("\n" + "="*80)
print("PROBLEM 2: LAMBDA VALUE DISTRIBUTION")
print("="*80)

lambda_stats = df['lambda_val'].describe()
print(f"\nLambda Statistics:")
print(lambda_stats)

# Bin distribution
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
bin_counts = pd.cut(df['lambda_val'], bins=bins).value_counts().sort_index()
print(f"\nLambda Distribution by Range:")
for interval, count in bin_counts.items():
    pct = 100 * count / len(df)
    print(f"  {interval}: {count} ({pct:.1f}%)")

low_lambda = len(df[df['lambda_val'] < 0.1])
mid_lambda = len(df[(df['lambda_val'] >= 0.3) & (df['lambda_val'] <= 0.7)])
print(f"\nLambda < 0.1: {low_lambda} ({100*low_lambda/len(df):.1f}%) [was 52%, target <40%]")
print(f"Lambda 0.3-0.7: {mid_lambda} ({100*mid_lambda/len(df):.1f}%) [was 19%, target >30%]")

if 100*mid_lambda/len(df) > 30:
    print("✅ GOOD: Mid-range lambda values are at target")
else:
    print("⚠️  WARNING: Need more ambiguous pairs in 0.3-0.7 range")

# PROBLEM 3: Match label imbalance
print("\n" + "="*80)
print("PROBLEM 3: MATCH LABEL BALANCE")
print("="*80)

match_counts = df['match'].value_counts()
print(f"\nMatch Distribution:")
for label, count in match_counts.items():
    pct = 100 * count / len(df)
    print(f"  match={label}: {count} ({pct:.1f}%)")

ratio = match_counts[0] / match_counts[1] if 1 in match_counts.index else np.inf
print(f"\nNon-match:Match ratio: {ratio:.2f}:1")
print(f"[Original was 2:1, target is <2:1]")

if ratio < 2.0:
    print("✅ GOOD: Balance is improved")
else:
    print("⚠️  WARNING: Still imbalanced but acceptable")

# PROBLEM 4: Entity type distribution
print("\n" + "="*80)
print("PROBLEM 4: ENTITY TYPE DISTRIBUTION")
print("="*80)

print(f"\ntype_a Distribution:")
type_a_counts = df['type_a'].value_counts()
for entity_type, count in type_a_counts.items():
    pct = 100 * count / len(df)
    print(f"  {entity_type}: {count} ({pct:.1f}%)")

print(f"\ntype_b Distribution:")
type_b_counts = df['type_b'].value_counts()
for entity_type, count in type_b_counts.items():
    pct = 100 * count / len(df)
    print(f"  {entity_type}: {count} ({pct:.1f}%)")

# Focus on underrepresented types
underrep_types = ['bacteria', 'pest', 'virus']
print(f"\nUnderrepresented types combined:")
for etype in underrep_types:
    count = len(df[df['type_a'] == etype]) + len(df[df['type_b'] == etype])
    pct = 100 * count / (2 * len(df))  # Since both type_a and type_b count
    print(f"  {etype}: {count} instances ({pct:.1f}%)")
    
disease_count = len(df[df['type_a'] == 'disease']) + len(df[df['type_b'] == 'disease'])
print(f"  disease: {disease_count} instances ({100*disease_count/(2*len(df)):.1f}%)")

underrep_total = sum(len(df[df[col] == etype]) for col in ['type_a', 'type_b'] for etype in underrep_types)
print(f"\nCombined (bacteria+pest+virus): {underrep_total} vs disease alone: {disease_count}")
if underrep_total > disease_count * 0.5:
    print("✅ GOOD: Underrepresented types improved significantly")
else:
    print("⚠️  WARNING: Underrepresented types still need attention")

# PROBLEM 5: LLM vs manual disagreements
print("\n" + "="*80)
print("PROBLEM 5: LLM vs MANUAL LABEL DISAGREEMENTS")
print("="*80)

# Convert llm_match to int for comparison
df['llm_match_int'] = pd.to_numeric(df['llm_match'], errors='coerce')
disagreements = df[df['match'] != df['llm_match_int']]

print(f"\nTotal disagreements: {len(disagreements)} ({100*len(disagreements)/len(df):.2f}%)")
print(f"Original: 33 disagreements")

if len(disagreements) < 33:
    print("✅ GOOD: Disagreements reduced!")
else:
    print("⚠️  WARNING: Disagreements not reduced")

case_A = disagreements[disagreements['match'] == 1]  # GT=match, LLM=no match
case_B = disagreements[disagreements['match'] == 0]  # GT=no match, LLM=match

print(f"\nCase A (GT=MATCH, LLM=NO MATCH): {len(case_A)}")
print(f"Case B (GT=NO MATCH, LLM=MATCH): {len(case_B)}")

print("\n" + "="*80)
print("CASE A DETAILS (GT says MATCH, LLM says NO MATCH):")
print("="*80)
for i, (_, row) in enumerate(case_A.iterrows(), 1):
    print(f"\n[A{i}] {row['name_a']} <-> {row['name_b']}")
    print(f"     canonical_a: {row['canonical_id_a']}, canonical_b: {row['canonical_id_b']}")
    print(f"     lambda_val: {row['lambda_val']:.4f}")
    print(f"     context_a[:80]: {str(row['context_a'])[:80]}")
    print(f"     context_b[:80]: {str(row['context_b'])[:80]}")

print("\n" + "="*80)
print("CASE B DETAILS (GT says NO MATCH, LLM says MATCH):")
print("="*80)
for i, (_, row) in enumerate(case_B.iterrows(), 1):
    print(f"\n[B{i}] {row['name_a']} <-> {row['name_b']}")
    print(f"     canonical_a: {row['canonical_id_a']}, canonical_b: {row['canonical_id_b']}")
    print(f"     lambda_val: {row['lambda_val']:.4f}")
    print(f"     context_a[:80]: {str(row['context_a'])[:80]}")
    print(f"     context_b[:80]: {str(row['context_b'])[:80]}")

# Summary
print("\n" + "="*80)
print("SUMMARY: ADDRESSING THE 5 PROBLEMS")
print("="*80)

problems_fixed = 0
print("\n1. Type C Hard Negatives:")
if type_counts.get('C', 0) >= 300:
    print("   ✅ FIXED")
    problems_fixed += 1
else:
    print(f"   ⚠️  PARTIAL ({type_counts.get('C', 0)}/300)")

print("\n2. Lambda Mid-Range Distribution:")
if 100*mid_lambda/len(df) >= 30:
    print("   ✅ FIXED")
    problems_fixed += 1
else:
    print(f"   ⚠️  PARTIAL ({100*mid_lambda/len(df):.1f}%/30%)")

print("\n3. Match Label Balance:")
if ratio < 2.0:
    print("   ✅ FIXED")
    problems_fixed += 1
else:
    print(f"   ⚠️  ACCEPTABLE ({ratio:.2f}:1)")

print("\n4. Entity Type Representation:")
if underrep_total >= disease_count * 0.5:
    print("   ✅ GOOD")
    problems_fixed += 1
else:
    print(f"   ⚠️  NEEDS WORK")

print("\n5. LLM Disagreements:")
if len(disagreements) < 33:
    print("   ✅ FIXED")
    problems_fixed += 1
else:
    print(f"   ⚠️  {len(disagreements)} remaining")

print(f"\nOVERALL: {problems_fixed}/5 problems addressed")
print("="*80 + "\n")
