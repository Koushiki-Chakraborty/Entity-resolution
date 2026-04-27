"""
=============================================================================
Step 2 of 2 — Pair Type Classifier
Agricultural Disease Entity Resolution

WHAT THIS DOES:
    Reads dataset_with_quality.csv (output of step1_context_quality.py)
    and adds a pair_type column classifying every pair as:

        Type A — Both signals agree: match
                 High name similarity + high context similarity + match=1
                 These are "easy" true matches. Your model needs them
                 but should not be evaluated mainly on them.

        Type B — Synonym pairs (the "name misleads" case)
                 Low name similarity + match=1
                 Same disease, completely different names.
                 Example: "Panama disease" vs "Fusarium wilt of banana"
                 Critical for testing whether context can override names.

        Type C — Polysemy pairs (the "nitrogen problem")
                 High name similarity + match=0
                 Different diseases with the same or very similar name.
                 Example: "leaf blight of wheat" vs "leaf blight of rice"
                 Critical for testing the conflict-aware lambda fix.

        Type D — Clear non-match
                 Low name similarity + match=0
                 Both signals agree: not the same disease.
                 These are "easy" negatives.

HOW NAME SIMILARITY IS COMPUTED:
    We use token Jaccard similarity — no ML model needed, no internet,
    runs in seconds. Jaccard(A,B) = |words in common| / |total unique words|.
    This is a reliable proxy for lexical name similarity.

    Threshold: name_sim >= 0.25 → "similar names" (A or C)
               name_sim <  0.25 → "different names" (B or D)

    You can adjust NAME_SIM_THRESHOLD below if the split looks wrong.

HOW TO RUN:
    1. Run step1_context_quality.py first
    2. Run: python step2_pair_type_classifier.py
    3. Check pair_type_report.txt and dataset_final.csv

OUTPUT:
    dataset_final.csv        — complete dataset with pair_type + quality columns
    pair_type_report.txt     — coverage statistics + examples of each type

=============================================================================
"""

import pandas as pd
import re
from collections import Counter
import sys

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_CSV  = "../data/dataset_with_quality.csv"   # output from step 1
OUTPUT_CSV = "../data/dataset_classified.csv"
REPORT_TXT = "../data/pair_type_report.txt"

# Adjust this threshold if Type B or C counts look wrong.
# Lower = more pairs classified as "similar names" (A or C)
# Higher = fewer pairs classified as "similar names"
NAME_SIM_THRESHOLD = 0.25


# ── NAME SIMILARITY ───────────────────────────────────────────────────────────

def tokenise(name):
    """
    Convert a disease name into a set of meaningful tokens.
    Removes stopwords and very short words that add noise.
    Example: "Fusarium wilt of banana" → {"fusarium", "wilt", "banana"}
    """
    STOPWORDS = {
        "of", "the", "a", "an", "and", "or", "in", "on", "by",
        "from", "with", "to", "for", "as", "at", "its", "their",
        "this", "that", "is", "are", "was", "be", "caused", "disease",
        "infection", "blight", "plant", "crop",   # these are so common
    }                                              # they don't help distinguish names

    tokens = re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def jaccard_similarity(name_a, name_b):
    """
    Jaccard similarity between two name strings.
    Returns a float in [0, 1]:
        0.0 = completely different words
        1.0 = identical words (order doesn't matter)
    """
    if not isinstance(name_a, str) or not isinstance(name_b, str):
        return 0.0
    set_a = tokenise(name_a)
    set_b = tokenise(name_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


def contains_substring(name_a, name_b):
    """
    True if one name is contained inside the other.
    Catches cases like "banana wilt" inside "Fusarium wilt of banana".
    """
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    return a in b or b in a


def name_similarity(name_a, name_b):
    """
    Combined name similarity: max of Jaccard and substring containment.
    Returns float in [0, 1].
    """
    j = jaccard_similarity(name_a, name_b)
    s = 1.0 if contains_substring(name_a, name_b) else 0.0
    return max(j, s)


# ── PAIR TYPE CLASSIFIER ──────────────────────────────────────────────────────

def classify_pair(row, threshold=NAME_SIM_THRESHOLD):
    """
    Classify a single row into pair type A, B, C, or D.

    Logic:
        match=1 AND name_sim >= threshold → Type A (safe match, names agree)
        match=1 AND name_sim <  threshold → Type B (synonym, names differ)
        match=0 AND name_sim >= threshold → Type C (polysemy, names similar but different)
        match=0 AND name_sim <  threshold → Type D (clear non-match)
    """
    is_match  = int(row["match"]) == 1
    name_sim  = name_similarity(str(row["name_a"]), str(row["name_b"]))

    if is_match and name_sim >= threshold:
        return "A", name_sim, "match + similar names"
    elif is_match and name_sim < threshold:
        return "B", name_sim, "match + different names (synonym pair)"
    elif not is_match and name_sim >= threshold:
        return "C", name_sim, "no-match + similar names (polysemy)"
    else:
        return "D", name_sim, "no-match + different names (clear non-match)"


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Pair Type Classifier")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n  ERROR: {INPUT_CSV} not found.")
        print("  Run step1_context_quality.py first.")
        return
    print(f"  {len(df)} rows loaded")

    # ── Classify ─────────────────────────────────────────────────────────────
    print("\n[2/4] Classifying pairs...")
    results = [classify_pair(row) for _, row in df.iterrows()]

    df["pair_type"]       = [r[0] for r in results]
    df["name_sim_score"]  = [round(r[1], 4) for r in results]
    df["pair_type_reason"]= [r[2] for r in results]

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/4] Generating report...")

    counts = Counter(df["pair_type"])
    total  = len(df)

    descriptions = {
        "A": "Safe match        (match=1, names similar)",
        "B": "Synonym pair      (match=1, names different) ← needs more",
        "C": "Polysemy          (match=0, names similar)   ← CRITICAL GAP",
        "D": "Clear non-match   (match=0, names different)",
    }

    targets = {"A": 400, "B": 300, "C": 300, "D": 600}

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append(" PAIR TYPE CLASSIFICATION REPORT")
    report_lines.append("=" * 65)
    report_lines.append(f"\nTotal pairs: {total}")
    report_lines.append(f"Name similarity threshold: {NAME_SIM_THRESHOLD}")
    report_lines.append("\n── Coverage by pair type ─────────────────────────────────")
    report_lines.append(f"  {'Type':<6} {'Count':>6} {'%':>7}  {'vs Target':>10}  Description")
    report_lines.append("  " + "-" * 60)

    for pt in ["A", "B", "C", "D"]:
        n   = counts.get(pt, 0)
        pct = n / total * 100
        tgt = targets[pt]
        gap = n - tgt
        gap_str = f"+{gap}" if gap >= 0 else str(gap)
        status  = "OK" if gap >= 0 else f"need {abs(gap)} more"
        report_lines.append(
            f"  {pt:<6} {n:>6} ({pct:5.1f}%)  {gap_str:>6} ({status:<16})  {descriptions[pt]}"
        )

    # Name similarity distribution
    report_lines.append("\n── Name similarity score distribution ────────────────────")
    bins = [(0.0,0.1),(0.1,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)]
    for lo, hi in bins:
        mask = (df["name_sim_score"] >= lo) & (df["name_sim_score"] < hi)
        n = mask.sum()
        bar = "█" * (n // 25)
        report_lines.append(f"  [{lo:.2f}–{hi:.2f}): {n:5d}  {bar}")

    # ── Examples of each type ─────────────────────────────────────────────────
    for pt, label in [("A","Safe match"), ("B","Synonym pair"),
                       ("C","Polysemy"), ("D","Clear non-match")]:
        subset = df[df["pair_type"] == pt][
            ["name_a", "name_b", "match", "name_sim_score",
             "context_a", "context_b"]
        ].head(5)

        report_lines.append(f"\n── Type {pt}: {label} — sample pairs ──────────────────")
        for _, row in subset.iterrows():
            report_lines.append(
                f"  name_a : {str(row['name_a'])[:55]}"
            )
            report_lines.append(
                f"  name_b : {str(row['name_b'])[:55]}"
            )
            report_lines.append(
                f"  match={int(row['match'])}  name_sim={row['name_sim_score']:.3f}"
            )
            report_lines.append(
                f"  ctx_a  : {str(row['context_a'])[:80]}"
            )
            report_lines.append(
                f"  ctx_b  : {str(row['context_b'])[:80]}"
            )
            report_lines.append("")

    # ── Danger zone: Type C with good contexts ────────────────────────────────
    if "context_quality_a" in df.columns:
        c_good = df[
            (df["pair_type"] == "C") &
            (df["context_quality_a"] == "good") &
            (df["context_quality_b"] == "good")
        ]
        report_lines.append(
            f"── Type C pairs with GOOD contexts (genuine polysemy): {len(c_good)}"
        )
        report_lines.append(
            "   These are the most valuable training examples for the conflict detector."
        )
        for _, row in c_good.head(5).iterrows():
            report_lines.append(f"  {row['name_a']!r:40s} vs {row['name_b']!r}")

    # ── Action items ──────────────────────────────────────────────────────────
    c_count = counts.get("C", 0)
    b_count = counts.get("B", 0)

    report_lines.append("\n── ACTION ITEMS ─────────────────────────────────────────")
    if c_count < 100:
        report_lines.append(
            f"  URGENT: Only {c_count} Type C (polysemy) pairs."
            f" Need ~{300-c_count} more from EPPO ambiguous common names."
        )
    if b_count < 200:
        report_lines.append(
            f"  IMPORTANT: Only {b_count} Type B (synonym) pairs."
            f" Need ~{300-b_count} more from EPPO synonym lists."
        )
    report_lines.append(
        "  During training, oversample Type C pairs 3x using WeightedRandomSampler."
    )
    report_lines.append(
        "  Exclude 'both_poor' pairs from lambda supervision (keep for match loss)."
    )

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved → {REPORT_TXT}")

    # ── Save final dataset ────────────────────────────────────────────────────
    print(f"\n[4/4] Saving {OUTPUT_CSV}...")
    # Drop debug reason columns for the clean final file
    clean_df = df.drop(columns=["quality_reason_a", "quality_reason_b",
                                  "pair_type_reason"], errors="ignore")
    clean_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}")
    print(f"\n  New columns added:")
    print(f"    pair_type          — A / B / C / D")
    print(f"    name_sim_score     — Jaccard name similarity [0–1]")
    print(f"    context_quality_a  — good / medium / poor")
    print(f"    context_quality_b  — good / medium / poor")
    print(f"\n  Total columns: {len(clean_df.columns)}")
    print(f"  Columns: {list(clean_df.columns)}")

    print("\n" + "=" * 60)
    print(" Done. dataset_classified.csv is your enriched training dataset.")
    print(" Next: run step3_eppo_collector.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
