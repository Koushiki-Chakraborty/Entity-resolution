"""
=============================================================================
Step 1 of 2 — Context Quality Scorer
Agricultural Disease Entity Resolution

WHAT THIS DOES:
    Reads your training_ready_final.csv and scores every context string
    (both context_a and context_b) as:

        "good"   — specific disease description: pathogen named,
                   symptoms described, crop mentioned
        "medium" — partial information: some disease info but missing
                   key details, or a generic taxon description
        "poor"   — useless for matching: Wikipedia list pages,
                   taxonomy family overviews, generic definitions,
                   very short stubs

    Adds two new columns: context_quality_a and context_quality_b.
    Saves as dataset_with_quality.csv.

WHY NO ML MODEL IS USED:
    The patterns that make a context poor are deterministic and keyword-
    driven. A rules-based scorer is faster, fully explainable, and does
    not need any internet or GPU. You can read every rule and understand
    exactly why each context got its label.

HOW TO RUN:
    1. Put this file in the same folder as training_ready_final.csv
    2. Run:  python step1_context_quality.py
    3. Check the printed report and dataset_with_quality.csv

OUTPUT:
    dataset_with_quality.csv   — your CSV with two new quality columns
    quality_report.txt         — summary statistics

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

INPUT_CSV  = "../data/base_dataset.csv"
OUTPUT_CSV = "../data/dataset_with_quality.csv"
REPORT_TXT = "../data/quality_report.txt"


# ── QUALITY RULES ─────────────────────────────────────────────────────────────
#
# These rules are applied IN ORDER. The first rule that matches wins.
# Order matters: POOR rules are checked first (they are the most specific),
# then GOOD rules (they require positive evidence), then MEDIUM is the default.

# ── POOR signals ─────────────────────────────────────────────────────────────
# These phrases appear in contexts that are useless for entity resolution.
# If ANY of these are found, the context is immediately POOR.

POOR_PHRASES = [
    # Wikipedia list pages
    "this article is a list",
    "is a list of",
    "list of diseases",
    "list of plant diseases",
    "list of crop",
    "abnormal hive conditions include",
    "diseases of the honey bee",
    "following is a list",
    "this is a list",
    "this page lists",
    "the following lists",
    "examples of the following",

    # Generic biological family / class definitions
    "rhabdoviridae is a family",
    "iflavirus is a genus",
    "is a genus of",
    "is a family of",
    "is an order of",
    "is a class of",
    "biosafety level",

    # Generic disease class overviews
    "plant viruses are viruses that",
    "a viral disease occurs when",
    "an rna virus is a virus",
    "plant diseases are diseases in plants",
    "organisms that cause infectious",

    # Taxonomy / agriculture product pages (not disease-specific)
    "is a root vegetable",
    "is an agricultural product affected by many",
    "most of the information here concerns",
    "durians are an agricultural",
    "parsnip is a root vegetable",

    # Zoo / aquarium pages that crept in
    "freshwater species have successfully adapted to live in aquariums",
    "live in aquariums",

    # Overly generic pathogen family pages
    "vertebrates, invertebrates, plant",
    "obligate intracellular parasites",
    "characterized by a ribonucleic acid",

    # Animal disease (not crop)
    "myxoma virus",
    "leporipoxvirus",
    "natural hosts are tapeti",
]

# ── POOR length threshold ─────────────────────────────────────────────────────
# Contexts shorter than this many characters cannot contain enough information.
POOR_LENGTH = 40

# ── GOOD signals — pathogen keywords ─────────────────────────────────────────
# A GOOD context names the pathogen AND describes the disease.
# We look for BOTH a pathogen indicator AND a symptom/impact indicator.

PATHOGEN_KEYWORDS = [
    # Fungal
    "caused by", "fusarium", "phytophthora", "alternaria", "puccinia",
    "botrytis", "magnaporthe", "colletotrichum", "sclerotinia", "pythium",
    "verticillium", "erysiphe", "blumeria", "plasmopara", "bremia",
    "cercospora", "septoria", "rhizoctonia", "claviceps", "ustilago",
    "gymnosporangium", "guignardia", "cochliobolus", "exserohilum",
    "passalora", "corynespora", "pectobacterium", "xanthomonas",
    "pseudomonas", "erwinia", "ralstonia", "agrobacterium",
    # Viral
    "tylcv", "tomato yellow leaf curl", "cucumber mosaic", "tobacco mosaic",
    "potyvirus", "geminivirus", "caulimovirus", "luteovirus", "begomovirus",
    "transmitted by", "whitefly", "aphid", "bemisia",
    # Oomycete
    "phytophthora infestans", "oomycete", "downy mildew",
    # Scientific name pattern: two capitalised words (Genus species)
]

SYMPTOM_KEYWORDS = [
    "lesion", "lesions", "blight", "wilt", "rot", "spot", "spots",
    "pustule", "pustules", "chlorosis", "necrosis", "yellowing",
    "discolouration", "discoloration", "canker", "gall", "scab",
    "mildew", "rust", "smut", "damping", "stippling", "bronzing",
    "mosaic", "streak", "mottle", "curl", "stunt", "dwarf",
    "symptom", "symptoms", "infect", "infection", "pathogen",
    "damage", "yield loss", "devastating", "severe",
    "water-soaked", "dark brown", "orange", "yellow-green", "grey",
]

CROP_KEYWORDS = [
    "wheat", "rice", "maize", "corn", "potato", "tomato", "soybean",
    "cotton", "sugarcane", "barley", "oat", "sorghum", "cassava",
    "banana", "grape", "citrus", "apple", "coffee", "cocoa",
    "pepper", "cucumber", "lettuce", "onion", "carrot", "beet",
    "sunflower", "canola", "rapeseed", "lentil", "chickpea",
    "groundnut", "peanut", "mango", "avocado", "strawberry",
    "turfgrass", "grapevine", "durum", "spelt", "triticale",
]


# ── MEDIUM signals — partial information ──────────────────────────────────────
# A context is MEDIUM if it mentions a pathogen/taxon but lacks symptoms/crop,
# or mentions a crop but not the pathogen.

MEDIUM_PHRASES = [
    "is a bacterium",
    "is a fungus",
    "is a virus",
    "is a species of",
    "is a plant pathogen",
    "is a pathogen",
    "formerly known as",
    "used to be a member",
    "reclassified",
    "in the family",
]


# ── SCORER ────────────────────────────────────────────────────────────────────

def has_scientific_name(text):
    """
    Detects patterns like 'Fusarium oxysporum' — capitalised genus + lowercase species.
    This is a strong signal that the context is specific enough to be useful.
    """
    return bool(re.search(r'\b[A-Z][a-z]+\s+[a-z]+\b', text))


def score_context(text):
    """
    Score a single context string.
    Returns: ("good" | "medium" | "poor", reason_string)
    """
    if not isinstance(text, str) or text.strip() == "":
        return "poor", "empty or missing"

    t = text.lower().strip()

    # ── Rule 1: Too short → always poor ──────────────────────────────────────
    if len(t) < POOR_LENGTH:
        return "poor", f"too short ({len(t)} chars)"

    # ── Rule 2: Poor phrase match → immediately poor ──────────────────────────
    for phrase in POOR_PHRASES:
        if phrase in t:
            return "poor", f"poor phrase: '{phrase}'"

    # ── Rule 3: Good — needs BOTH pathogen evidence AND symptom/crop evidence ─
    has_pathogen = (
        any(kw in t for kw in PATHOGEN_KEYWORDS)
        or has_scientific_name(text)
    )
    has_symptom = any(kw in t for kw in SYMPTOM_KEYWORDS)
    has_crop    = any(kw in t for kw in CROP_KEYWORDS)

    if has_pathogen and (has_symptom or has_crop):
        return "good", "pathogen + symptom/crop found"

    # ── Rule 4: Medium — partial evidence ────────────────────────────────────
    if has_pathogen or has_symptom or has_crop:
        return "medium", "partial: only some disease signals present"

    for phrase in MEDIUM_PHRASES:
        if phrase in t:
            return "medium", f"medium phrase: '{phrase}'"

    # ── Rule 5: Default — not enough information ──────────────────────────────
    return "poor", "no disease-specific information detected"


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Context Quality Scorer")
    print("=" * 60)

    # Load
    print(f"\n[1/4] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Score every context
    print("\n[2/4] Scoring context_a...")
    results_a = [score_context(t) for t in df["context_a"]]
    df["context_quality_a"] = [r[0] for r in results_a]
    df["quality_reason_a"]  = [r[1] for r in results_a]

    print("[2/4] Scoring context_b...")
    results_b = [score_context(t) for t in df["context_b"]]
    df["context_quality_b"] = [r[0] for r in results_b]
    df["quality_reason_b"]  = [r[1] for r in results_b]

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/4] Generating report...")

    def dist(col):
        c = Counter(df[col])
        total = len(df)
        return {k: (c[k], f"{c[k]/total*100:.1f}%") for k in ["good", "medium", "poor"]}

    dist_a = dist("context_quality_a")
    dist_b = dist("context_quality_b")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(" CONTEXT QUALITY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nTotal pairs: {len(df)}")
    report_lines.append("\n── context_a quality ──────────────────────────")
    for q in ["good", "medium", "poor"]:
        n, pct = dist_a[q]
        bar = "█" * (n // 20)
        report_lines.append(f"  {q:8s}: {n:5d} ({pct:6s}) {bar}")
    report_lines.append("\n── context_b quality ──────────────────────────")
    for q in ["good", "medium", "poor"]:
        n, pct = dist_b[q]
        bar = "█" * (n // 20)
        report_lines.append(f"  {q:8s}: {n:5d} ({pct:6s}) {bar}")

    # Pairs where BOTH contexts are poor — most dangerous for training
    both_poor = ((df["context_quality_a"] == "poor") &
                 (df["context_quality_b"] == "poor"))
    report_lines.append(f"\n── Both contexts poor (most dangerous): {both_poor.sum()} pairs")
    report_lines.append("   These pairs corrupt your lambda supervision signal most.")
    report_lines.append("   Consider excluding them from lambda training (keep for match training).")

    # Examples of poor contexts
    report_lines.append("\n── Sample POOR context_a reasons ──────────────")
    poor_a = df[df["context_quality_a"] == "poor"][["name_a", "context_a", "quality_reason_a"]].head(10)
    for _, row in poor_a.iterrows():
        report_lines.append(f"  Name : {row['name_a']}")
        report_lines.append(f"  Text : {str(row['context_a'])[:90]}")
        report_lines.append(f"  Why  : {row['quality_reason_a']}")
        report_lines.append("")

    # Examples of good contexts
    report_lines.append("── Sample GOOD context_a ───────────────────────")
    good_a = df[df["context_quality_a"] == "good"][["name_a", "context_a"]].head(5)
    for _, row in good_a.iterrows():
        report_lines.append(f"  Name : {row['name_a']}")
        report_lines.append(f"  Text : {str(row['context_a'])[:90]}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Saved → {REPORT_TXT}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving {OUTPUT_CSV}...")
    # Keep reason columns for inspection but note they are debug-only
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}")
    print(f"  New columns: context_quality_a, context_quality_b, quality_reason_a, quality_reason_b")

    print("\n" + "=" * 60)
    print(" Next step: run step2_pair_type_classifier.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
