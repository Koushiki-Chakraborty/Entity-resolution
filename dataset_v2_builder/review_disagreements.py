"""
Interactive Disagreement Reviewer
==================================
Shows each disagreement row with full context so you can decide if it's a labeling error.
Press Enter to go to next row. Type 'help' to see commands.
"""

import pandas as pd
import sys


def display_row(df, idx, row_num, total):
    """Display a single disagreement row with context."""
    row = df.iloc[idx]
    
    print("\n" + "=" * 80)
    print(f"ROW {row_num}/{total}  |  Pair Type: {row['pair_type']}  |  "
          f"Lambda: {row['lambda_val']:.4f}")
    print("=" * 80)
    
    print(f"\nEntity A:  {row['name_a']}")
    print(f"  Type: {row['type_a']}")
    print(f"  Context: {row['context_a'][:150]}...")
    
    print(f"\nEntity B:  {row['name_b']}")
    print(f"  Type: {row['type_b']}")
    print(f"  Context: {row['context_b'][:150]}...")
    
    print(f"\n[DISAGREEMENT]")
    print(f"  Your label (match):  {row['match']}")
    print(f"  LLM label (llm_match): {row['llm_match']}")
    print(f"  Pairs should match? (0=different, 1=same)")
    
    print(f"\nQUESTION: Is the label '{int(row['match'])}' correct, or should it be '{int(row['llm_match'])}'?")
    print(f"Options: [c]orrect (keep), [w]rong (fix), [?]skip, [q]uit")


def main():
    # Load the disagreements
    df_full = pd.read_csv("data/dataset_fixed.csv")
    disagreements = df_full[df_full["match"] != df_full["llm_match"]].reset_index(drop=True)
    
    print(f"\nDisagreement Reviewer")
    print(f"Found {len(disagreements)} rows to review\n")
    
    # Track decisions
    fixes = {}  # idx -> new_match_value
    
    current_idx = 0
    while current_idx < len(disagreements):
        display_row(disagreements, current_idx, current_idx + 1, len(disagreements))
        
        response = input("\n➜ ").strip().lower()
        
        if response == 'c':
            print("  ✓ Keeping label as-is (match={})".format(
                int(disagreements.iloc[current_idx]['match'])))
            current_idx += 1
            
        elif response == 'w':
            new_val = int(disagreements.iloc[current_idx]['llm_match'])
            fixes[current_idx] = new_val
            print(f"  ✓ Marked for fix: match={new_val}")
            current_idx += 1
            
        elif response == '?':
            print("  ⊘ Skipping for now")
            current_idx += 1
            
        elif response == 'q':
            print("\nQuitting...")
            break
            
        elif response == 'help':
            print("\n[COMMANDS]")
            print("  c     - Correct (keep your label)")
            print("  w     - Wrong (fix to LLM label)")
            print("  ?     - Skip for now")
            print("  show  - Show full contexts")
            print("  q     - Quit and save")
            
        elif response == 'show':
            row = disagreements.iloc[current_idx]
            print(f"\n[FULL CONTEXT A]\n{row['context_a']}")
            print(f"\n[FULL CONTEXT B]\n{row['context_b']}")
            
        else:
            print("  ? Unknown command. Type 'help'")
            continue
    
    # Apply fixes
    if fixes:
        print(f"\n\nApplying {len(fixes)} fixes...")
        for idx, new_val in fixes.items():
            # Get original row index in full dataset
            orig_idx = disagreements.iloc[idx].name
            df_full.at[orig_idx, "match"] = new_val
            print(f"  Row {idx}: match = {new_val}")
        
        # Save
        output = "data/dataset_fixed_reviewed.csv"
        df_full.to_csv(output, index=False)
        print(f"\nSaved {len(fixes)} fixes → {output}")
        print("\nNow compare:")
        print(f"  diff data/dataset_fixed.csv {output}")
    else:
        print("\nNo fixes applied.")


if __name__ == "__main__":
    main()
