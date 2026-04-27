"""
=============================================================================
Master Runner — Build Dataset V2 Complete Pipeline
Run all 5 steps in sequence with error handling

USAGE:
    cd dataset_v2_builder/scripts
    python run_all.py
    
    Or run individual steps:
    python step1_context_quality.py
    python step2_pair_type_classifier.py
    python step3_eppo_collector.py
    python step4_usda_external_test.py
    python step5_merge_all.py

OUTPUT:
    - ../data/dataset_v2.csv (2,500 training pairs)
    - ../data/external_test_set_isolated.csv (200 test pairs)
    - ../data/quality_report.txt
    - ../data/pair_type_report.txt

=============================================================================
"""

import subprocess
import sys
import os
from pathlib import Path
import io

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPTS_DIR = Path(__file__).parent
STEPS = [
    ("step1_context_quality.py", "Context Quality Scoring"),
    ("step2_pair_type_classifier.py", "Pair Type Classification"),
    ("step3_eppo_collector.py", "EPPO API Collection"),
    ("step4_usda_external_test.py", "USDA External Test Set"),
    ("step5_merge_all.py", "Final Merge")
]

def run_step(script_path, step_name):
    """Run a single step, with error handling"""
    print("\n" + "=" * 70)
    print(f" Step: {step_name}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SCRIPTS_DIR,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in {step_name}: {e}")
        print(f"  To debug, run manually:")
        print(f"  cd {SCRIPTS_DIR}")
        print(f"  python {script_path.name}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    print("=" * 70)
    print(" Dataset V2 Builder — Complete Pipeline")
    print("=" * 70)
    print(f"\nWill run {len(STEPS)} steps to build your dataset_v2.csv\n")
    
    for script_name, step_name in STEPS:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            print(f"\n✗ Script not found: {script_path}")
            sys.exit(1)
    
    # Run steps
    completed = 0
    for script_name, step_name in STEPS:
        script_path = SCRIPTS_DIR / script_name
        if run_step(script_path, step_name):
            completed += 1
        else:
            print(f"\nStopping due to error. {completed}/{len(STEPS)} steps completed.")
            sys.exit(1)
    
    # Success!
    print("\n" + "=" * 70)
    print(f" ✅ All {len(STEPS)} steps completed successfully!")
    print("=" * 70)
    
    data_dir = SCRIPTS_DIR.parent / "data"
    print(f"\nOutput files:")
    print(f"  Training: {data_dir / 'dataset_v2.csv'}")
    print(f"  External Test: {data_dir / 'external_test_set_isolated.csv'}")
    print(f"  Reports: {data_dir / 'quality_report.txt'}")
    print(f"           {data_dir / 'pair_type_report.txt'}")
    
    print(f"\nNext: Train your model on dataset_v2.csv")
    print(f"      Evaluate on external_test_set_isolated.csv")
    print(f"      Read the reports for insights on pair distribution")


if __name__ == "__main__":
    main()
