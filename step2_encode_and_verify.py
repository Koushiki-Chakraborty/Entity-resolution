"""
step2_encode_and_verify.py
==========================
Loads dataset_clean.csv, runs the frozen fine-tuned encoder, saves the
four vector tensors, and prints a verification report.

Run from the PROJECT ROOT (entity_resolution_copy/):
    python step2_encode_and_verify.py

Requires:
    - dataset_v2_builder/data/dataset_clean.csv
    - plant-disease-encoder/     (fine-tuned model from finetune.py)

Outputs:
    - outputs/encoded_vectors.pt    (4 tensors of shape [N, 384])
    - Verification report on console

Force re-encode (ignore cache):
    python step2_encode_and_verify.py --force

Ablation (use base model, not fine-tuned):
    python step2_encode_and_verify.py --base
"""

import sys
import pathlib

# Make src/ importable so 'from encoder.xxx import ...' resolves correctly.
ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from encoder.config import CLEAN_DATA_PATH, VECTORS_PATH
from encoder.frozen_encoder import FrozenEncoder


def main(force_recompute: bool = False, force_base: bool = False):
    # ── Step 1: Load the cleaned dataset ─────────────────────────────────────
    print(f"\nLoading dataset from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"  {len(df)} rows loaded")
    print(f"  match=1 : {df['match'].sum()}   match=0 : {(df['match']==0).sum()}")

    if "pair_type" in df.columns:
        pt_counts = df["pair_type"].value_counts().to_dict()
        print(f"  pair_type breakdown : {pt_counts}")

    if "type_a" in df.columns:
        print(f"  type_a distribution : {df['type_a'].value_counts().to_dict()}")

    # ── Step 2: Create the frozen encoder ────────────────────────────────────
    # Loads the fine-tuned model and freezes all weights.
    # If fine-tuned model doesn't exist, falls back to base model with warning.
    encoder = FrozenEncoder(force_base=force_base)

    # ── Step 3: Encode all entity pairs -> 4 tensors ─────────────────────────
    # Each tensor: shape [N, 384].
    # Reads type_a / type_b from dataframe for entity tags.
    vectors = encoder.encode_dataset(
        df,
        cache_path=VECTORS_PATH,
        force_recompute=force_recompute,
    )

    # ── Step 4: Print verification report ────────────────────────────────────
    # Checks that encoded vectors separate match=1 from match=0.
    # Positive separation = encoder is working correctly.
    encoder.verification_report(vectors, df)

    # ── Step 5: Confirm output ────────────────────────────────────────────────
    n_input = 4 * 384 + 2   # 1538

    print(f"\n{'='*60}")
    print(f"  Step 2 complete.")
    print(f"  Vectors saved -> {VECTORS_PATH}")
    print(f"\n  Tensor shapes:")
    for key, tensor in vectors.items():
        print(f"    {key:<14} : {tuple(tensor.shape)}")
    print(f"\n  PairAwareAgriLambdaNet input size = {n_input} dims")
    print(f"    = 4 x 384 (vecs) + 2 (sim_name + sim_ctx)")
    print(f"\n  These tensors are ready as input for PairAwareAgriLambdaNet.")
    print(f"{'='*60}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    base  = "--base"  in sys.argv

    if force:
        print("--force: ignoring cache, re-encoding from scratch.")
    if base:
        print("--base: using base model (ablation mode).")

    main(force_recompute=force, force_base=base)
