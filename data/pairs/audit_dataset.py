import pandas as pd

df = pd.read_csv('data/pairs/training_ready_final.csv')
df['llm_match_int'] = df['llm_match'].astype(int)
disagreements = df[df['match'] != df['llm_match_int']].copy()

case_A = disagreements[disagreements['match'] == 1]  # GT=match, LLM=no match
case_B = disagreements[disagreements['match'] == 0]  # GT=no match, LLM=match

print(f'Total disagreements: {len(disagreements)}')
print(f'Case A (GT=MATCH, LLM=NO MATCH): {len(case_A)}')
print(f'Case B (GT=NO MATCH, LLM=MATCH): {len(case_B)}')

print()
print('=' * 70)
print('CASE A DETAILS (GT says MATCH, LLM says NO MATCH):')
print('=' * 70)
for i, (_, row) in enumerate(case_A.iterrows()):
    print(f'[A{i+1}]')
    print(f'  name_a     : {row["name_a"]}')
    print(f'  name_b     : {row["name_b"]}')
    print(f'  ctx_a      : {str(row["context_a"])[:130]}')
    print(f'  ctx_b      : {str(row["context_b"])[:130]}')
    print(f'  canonical_a: {row["canonical_id_a"]}')
    print(f'  canonical_b: {row["canonical_id_b"]}')
    print(f'  lambda_val : {row["lambda_val"]:.4f}')
    print()
