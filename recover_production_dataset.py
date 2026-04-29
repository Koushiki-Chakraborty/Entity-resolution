#!/usr/bin/env python3
"""
Recover dataset_production_ready.csv from dataset_final.csv
This script regenerates the cleaned production dataset
"""

import pandas as pd
import re
import os

def classify_entity_type(name, context):
    """
    Classify entity into agricultural types based on keywords in name and context
    """
    text = f"{name} {context}".lower()
    
    # Define keyword patterns for each type (priority order matters!)
    patterns = {
        'disease': r'\b(disease|blight|rot|wilt|scab|spot|canker|necrosis|damping|damp off)\b',
        'fungus': r'\b(fungus|fungal|mycete|oomycete|mold|mildew|rust|smut|phytophthora|alternaria|botrytis|fusarium|cladosporium|cercospora|septoria)\b',
        'virus': r'\b(virus|viral|virion|begomovirus|potyvirus|tobamovirus|comovirus|nepovirus|tymovirus|reovirus|lentivirus|potexvirus)\b',
        'bacteria': r'\b(bacteria|bacterial|bacillus|xanthomonas|pseudomonas|ralstonia|erwinia|agrobacterium|corynebacterium)\b',
        'pest': r'\b(pest|insect|mite|aphid|whitefly|psyllid|beetle|fly|weevil|nematode|caterpillar|moth|scale)\b',
        'plant': r'\b(plant|crop|cultivar|variety|species|genus|family|flora|botanical)\b',
        'organism': r'\b(organism|pathogen|microorganism|agent|vector|host)\b'
    }
    
    scores = {}
    for entity_type, pattern in patterns.items():
        matches = len(re.findall(pattern, text))
        scores[entity_type] = matches
    
    # Return highest scoring type, or 'unknown' if no keywords found
    if max(scores.values()) == 0:
        return 'unknown'
    return max(scores, key=scores.get)


def check_context_match(entity_name, context, name_keywords=None):
    """
    Check if entity name matches its context description
    Returns: (is_match: bool, confidence: float)
    """
    if pd.isna(context) or context == '':
        return False, 0.0
    
    context_lower = str(context).lower()
    name_lower = str(entity_name).lower()
    
    # Perfect match: name appears in context
    if name_lower in context_lower:
        return True, 0.9
    
    # Keyword overlap: split name and check if parts appear in context
    name_words = set(name_lower.split())
    context_words = set(context_lower.split())
    
    if len(name_words) > 0:
        overlap = len(name_words & context_words) / len(name_words)
        if overlap >= 0.5:
            return True, 0.7 + (0.2 * overlap)
    
    # No match
    return False, max(0.0, 0.2 * overlap) if 'overlap' in locals() else 0.0


def is_non_agricultural(context_a, context_b):
    """
    Check if row contains non-agricultural content
    """
    non_agri_keywords = [
        'sports car', 'motor company', 'railway', 'train', 'music', 'album',
        'film', 'movie', 'novel', 'book', 'character', 'nasa', 'mission',
        'act of congress', 'government', 'legislation', 'law', 'language',
        'linguistics', 'geography', 'river', 'country', 'city', 'location',
        'architecture', 'building', 'rock band', 'musician', 'artist',
        'mercedes', 'triumph', 'automobile', 'car model', 'suv', 'luxury'
    ]
    
    contexts = [str(c).lower() for c in [context_a, context_b] if pd.notna(c)]
    
    for ctx in contexts:
        for keyword in non_agri_keywords:
            if keyword in ctx:
                return True
    return False


def main():
    print("=" * 100)
    print("RECOVERING dataset_production_ready.csv")
    print("=" * 100)
    
    # Load original dataset
    print("\n[1] Loading dataset_final.csv...")
    df = pd.read_csv('dataset_v2_builder/data/dataset_final.csv')
    print(f"    Loaded {len(df)} rows")
    print(f"    Columns: {len(df.columns)}")
    
    original_count = len(df)
    
    # Add entity types
    print("\n[2] Classifying entity types...")
    df['type_a'] = df.apply(lambda row: classify_entity_type(row['name_a'], row['context_a']), axis=1)
    df['type_b'] = df.apply(lambda row: classify_entity_type(row['name_b'], row['context_b']), axis=1)
    
    type_a_dist = df['type_a'].value_counts()
    type_b_dist = df['type_b'].value_counts()
    print(f"    Type A distribution:\n{type_a_dist}")
    print(f"    Type B distribution:\n{type_b_dist}")
    
    # Identify context mismatches
    print("\n[3] Identifying context mismatches...")
    mismatch_rows = []
    
    for idx, row in df.iterrows():
        match_a, conf_a = check_context_match(row['name_a'], row['context_a'])
        match_b, conf_b = check_context_match(row['name_b'], row['context_b'])
        
        # Flag rows with severe mismatches (both sides score low)
        if conf_a < 0.2 and conf_b < 0.2:
            mismatch_rows.append(idx)
        # Also flag if match column is 0 (confirmed non-match) and both confidence low
        elif row['match'] == 0 and conf_a < 0.3 and conf_b < 0.3:
            mismatch_rows.append(idx)
    
    print(f"    Found {len(mismatch_rows)} rows with severe mismatches")
    
    # Remove mismatches
    df = df.drop(mismatch_rows).reset_index(drop=True)
    print(f"    After removing mismatches: {len(df)} rows")
    
    # Identify non-agricultural rows
    print("\n[4] Identifying non-agricultural data...")
    non_agri_rows = []
    
    for idx, row in df.iterrows():
        if is_non_agricultural(row['context_a'], row['context_b']):
            non_agri_rows.append(idx)
    
    print(f"    Found {len(non_agri_rows)} non-agricultural rows")
    
    # Remove non-agricultural
    df = df.drop(non_agri_rows).reset_index(drop=True)
    print(f"    After removing non-agricultural: {len(df)} rows")
    
    # Final statistics
    print("\n[5] FINAL DATASET STATISTICS")
    print("=" * 100)
    
    removed_count = len(mismatch_rows) + len(non_agri_rows)
    
    print(f"\nOriginal rows:              {original_count}")
    print(f"Removed (mismatches):       {len(mismatch_rows)}")
    print(f"Removed (non-agricultural): {len(non_agri_rows)}")
    print(f"Final rows:                 {len(df)}")
    print(f"Total rows removed:         {removed_count} ({100*removed_count/original_count:.1f}%)")
    
    total_slots = len(df) * 2
    unknown_slots = ((df['type_a'] == 'unknown').sum() + (df['type_b'] == 'unknown').sum())
    classified_slots = total_slots - unknown_slots
    
    print(f"\nTotal entity slots:         {total_slots}")
    print(f"Classified:                 {classified_slots}")
    print(f"Unknown:                    {unknown_slots}")
    print(f"Classification rate:        {100*classified_slots/total_slots:.1f}%")
    
    print("\n[6] TYPE DISTRIBUTION")
    print("=" * 100)
    
    type_dist = pd.DataFrame({
        'Type': df['type_a'].unique(),
        'Count_A': [len(df[df['type_a'] == t]) for t in df['type_a'].unique()],
    })
    type_dist['Count_B'] = type_dist['Type'].apply(lambda t: len(df[df['type_b'] == t]))
    type_dist['Total'] = type_dist['Count_A'] + type_dist['Count_B']
    type_dist = type_dist.sort_values('Total', ascending=False)
    
    print(type_dist.to_string(index=False))
    
    # Save dataset
    output_path = 'dataset_v2_builder/data/dataset_production_ready.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    print(f"     Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"     Rows: {len(df)}")
    print(f"     Columns: {len(df.columns)}")
    
    # Create summary report
    summary_data = {
        'Metric': [
            'Original rows',
            'Context mismatches removed',
            'Non-agricultural removed',
            'Final production rows',
            'Total entity slots',
            'Classification rate (%)'
        ],
        'Value': [
            original_count,
            len(mismatch_rows),
            len(non_agri_rows),
            len(df),
            total_slots,
            f"{100*classified_slots/total_slots:.1f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('dataset_v2_builder/data/dataset_recovery_report.csv', index=False)
    print(f"[OK] Saved: dataset_v2_builder/data/dataset_recovery_report.csv")
    
    print("\n" + "=" * 100)
    print("RECOVERY COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
