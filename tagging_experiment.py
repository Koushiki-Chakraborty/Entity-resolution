"""
Step 5a: Tagging Experiment (Ablation Study)
Agricultural Disease Entity Resolution

PURPOSE:
    Prove that [DISEASE] / [CONTEXT] tags improve embedding separation
    BEFORE any fine-tuning. This is the ablation study for your paper/report.

    Compares 3 conditions on the same base model (all-MiniLM-L6-v2):
        1. Name only          →  no tags, no context
        2. Name + context     →  no tags, but context included
        3. Ditto tagged       →  [DISEASE] + [CONTEXT] tags (your novelty)

    Then also compares base model vs your fine-tuned model (Step 4).

Run:
    python step5a_tagging_experiment.py

Requires:
    - test_set.csv                (from Step 4)
    - ./plant-disease-encoder/    (from Step 4)

Output:
    - Console table showing separation scores per condition
    - tagging_experiment_results.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL     = "all-MiniLM-L6-v2"
FINETUNED_DIR  = "./plant-disease-encoder"
TEST_CSV       = "test_set.csv"
RESULTS_CSV    = "tagging_experiment_results.csv"


# ─── SERIALIZATION VARIANTS ──────────────────────────────────────────────────

def encode_name_only(name: str, context: str, entity_type: str = "DISEASE") -> str:
    """Condition 1: just the raw name, no context, no tags"""
    return name


def encode_name_context(name: str, context: str, entity_type: str = "DISEASE") -> str:
    """Condition 2: name + context concatenated, no tags"""
    return f"{name} {context}"


def encode_ditto_tagged(name: str, context: str, entity_type: str = "DISEASE") -> str:
    """Condition 3: Ditto-style domain tags (your novelty)"""
    return (
        f"[{entity_type}] {name} [/{entity_type}] "
        f"[CONTEXT] {context} [/CONTEXT]"
    )


CONDITIONS = {
    "1_name_only":      encode_name_only,
    "2_name_context":   encode_name_context,
    "3_ditto_tagged":   encode_ditto_tagged,
}


# ─── CORE EXPERIMENT ─────────────────────────────────────────────────────────

def cosine_similarities(model, texts_a, texts_b):
    vecs_a = model.encode(texts_a, batch_size=32, show_progress_bar=False,
                          convert_to_tensor=True)
    vecs_b = model.encode(texts_b, batch_size=32, show_progress_bar=False,
                          convert_to_tensor=True)
    return F.cosine_similarity(vecs_a, vecs_b).cpu().numpy()


def run_condition(model, df, serialize_fn, condition_name):
    texts_a = [serialize_fn(r["name_1"], r["context_1"]) for _, r in df.iterrows()]
    texts_b = [serialize_fn(r["name_2"], r["context_2"]) for _, r in df.iterrows()]
    sims    = cosine_similarities(model, texts_a, texts_b)

    pos_sims = sims[df["label"].values == 1]
    neg_sims = sims[df["label"].values == 0]

    separation = pos_sims.mean() - neg_sims.mean()

    return {
        "condition":       condition_name,
        "match_mean":      round(float(pos_sims.mean()),  4),
        "match_min":       round(float(pos_sims.min()),   4),
        "match_max":       round(float(pos_sims.max()),   4),
        "nonmatch_mean":   round(float(neg_sims.mean()),  4),
        "nonmatch_max":    round(float(neg_sims.max()),   4),
        "separation":      round(float(separation),       4),  # KEY METRIC
    }


# ─── QUALITATIVE TEST CASES ──────────────────────────────────────────────────

def run_qualitative(model, serialize_fn):
    """
    Hand-picked pairs to show tag effect on specific examples.
    Same pairs used in Step 4 sanity check + a cross-type pest example.
    """
    cases = [
        # Same disease, different name → should be HIGH
        ("sudden death syndrome",
         "Fungal disease caused by Fusarium virguliforme.",
         "DISEASE",
         "sds",
         "Fungal disease caused by Fusarium virguliforme.",
         "DISEASE",
         "MATCH — abbreviation"),

        # Completely different diseases → should be LOW
        ("banana fusarium wilt",
         "Vascular wilt caused by Fusarium oxysporum f. sp. cubense.",
         "DISEASE",
         "black rot of grape",
         "Caused by Guignardia bidwellii. Produces dark lesions on fruit.",
         "DISEASE",
         "NON-MATCH — different diseases"),

        # Disease vs pest → should be LOW with type-aware tags
        ("leaf blight",
         "Fungal infection causing brown necrotic spots on leaves.",
         "DISEASE",
         "aphid infestation",
         "Small sap-sucking insects that colonise stems and undersides of leaves.",
         "PEST",
         "NON-MATCH — disease vs pest"),
    ]

    results = []
    for n1, c1, t1, n2, c2, t2, label in cases:
        s1  = serialize_fn(n1, c1, t1)
        s2  = serialize_fn(n2, c2, t2)
        v1  = model.encode(s1, convert_to_tensor=True)
        v2  = model.encode(s2, convert_to_tensor=True)
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        results.append((label, n1, n2, round(sim, 4)))

    return results


# ─── PRETTY PRINT ────────────────────────────────────────────────────────────

def print_results_table(all_results):
    print("\n── Tagging Experiment Results ───────────────────────────────────────────")
    print(f"{'Condition':<35} {'Match↑':>8} {'Non-match↓':>12} {'Separation↑':>13}")
    print("-" * 72)
    for r in all_results:
        marker = "  ← BEST" if r["separation"] == max(x["separation"] for x in all_results) else ""
        print(f"  {r['condition']:<33} {r['match_mean']:>8.4f} {r['nonmatch_mean']:>12.4f} "
              f"{r['separation']:>13.4f}{marker}")


def print_qualitative_table(base_results, finetuned_results):
    print("\n── Qualitative Examples ─────────────────────────────────────────────────")
    print(f"{'Pair':<40} {'Base (tagged)':>14} {'Fine-tuned':>12} {'Expected':>12}")
    print("-" * 82)
    for (label, n1, n2, base_sim), (_, _, _, ft_sim) in zip(base_results, finetuned_results):
        pair = f"{n1[:18]}.. vs {n2[:14]}.." if len(n1) > 18 else f"{n1} vs {n2}"
        expected = "HIGH" if "MATCH —" in label and "NON" not in label else "LOW"
        print(f"  {pair:<38} {base_sim:>14.4f} {ft_sim:>12.4f} {expected:>12}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} pairs  |  "
          f"positives={df['label'].sum()}  negatives={(df['label']==0).sum()}")

    # ── Part 1: Base model, 3 serialization conditions ───────────────────────
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = SentenceTransformer(BASE_MODEL)

    print("Running 3 conditions on base model...")
    all_results = []
    for cname, fn in CONDITIONS.items():
        print(f"  → {cname}")
        result = run_condition(base_model, df, fn, f"{cname} [base]")
        all_results.append(result)

    # ── Part 2: Fine-tuned model with Ditto tags ──────────────────────────────
    print(f"\nLoading fine-tuned model: {FINETUNED_DIR}")
    ft_model = SentenceTransformer(FINETUNED_DIR)

    print("  → 3_ditto_tagged [fine-tuned]")
    ft_result = run_condition(ft_model, df, encode_ditto_tagged,
                              "3_ditto_tagged [fine-tuned]")
    all_results.append(ft_result)

    # ── Print results table ───────────────────────────────────────────────────
    print_results_table(all_results)

    # ── Part 3: Qualitative examples ─────────────────────────────────────────
    base_qual = run_qualitative(base_model, encode_ditto_tagged)
    ft_qual   = run_qualitative(ft_model,   encode_ditto_tagged)
    print_qualitative_table(base_qual, ft_qual)

    # ── Save ─────────────────────────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved → {RESULTS_CSV}")

    # ── Summary ──────────────────────────────────────────────────────────────
    base_name_only = all_results[0]["separation"]
    base_tagged    = all_results[2]["separation"]
    ft_tagged      = all_results[3]["separation"]

    tag_improvement = round(base_tagged - base_name_only, 4)
    ft_improvement  = round(ft_tagged - base_tagged, 4)

    print(f"\n── What the numbers prove ───────────────────────────────────────────────")
    print(f"  Name only (base)     separation: {base_name_only:.4f}")
    print(f"  Ditto tags (base)    separation: {base_tagged:.4f}  "
          f"(+{tag_improvement} from tagging alone)")
    print(f"  Ditto tags (tuned)   separation: {ft_tagged:.4f}  "
          f"(+{ft_improvement} from fine-tuning on top of tags)")
    print(f"\n  → Tagging contribution  : +{tag_improvement:.4f}")
    print(f"  → Fine-tuning contribution: +{ft_improvement:.4f}")
    print(f"\nStep 5a complete. These numbers are your ablation study.")