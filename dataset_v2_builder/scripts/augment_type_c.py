"""
augment_type_c.py
=================
Systematically generates Type C (polysemy / hard-negative) pairs for the
AgriLambdaNet entity resolution dataset.

Strategy
--------
Type C = match=0, name_sim >= 0.25  (similar names, DIFFERENT diseases)

We mine pairs by cross-pairing entities that share a keyword (rust, blight,
wilt, spot, rot, mosaic, mildew, streak, scab, smut) but belong to DIFFERENT
canonical IDs (i.e. genuinely different diseases).

Input : dataset_production_ready.csv   (existing 1,505 rows)
        all_entities.csv               (entity index)
Output: dataset_augmented.csv          (existing rows + new Type C rows)
        augmentation_report.txt
"""

import pandas as pd
import re
import random
import hashlib
from itertools import combinations
from collections import defaultdict

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_IN   = "../data/dataset_production_ready.csv"
ENTITIES_CSV = "../../data/processed/all_entities.csv"
DATASET_OUT  = "../data/dataset_augmented.csv"
REPORT_OUT   = "../data/augmentation_report.txt"

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_NEW_C   = 350          # desired new Type C pairs
MAX_PER_SEED   = 40           # cap pairs sharing the exact same keyword seed
NAME_SIM_FLOOR = 0.25         # minimum Jaccard to qualify as Type C

KEYWORDS = [
    "rust", "blight", "wilt", "spot", "rot", "mosaic",
    "mildew", "streak", "scab", "smut", "blast", "canker",
    "necrosis", "yellowing", "curl", "leaf",
]

STOPWORDS = {
    "of", "the", "a", "an", "and", "or", "in", "on", "by",
    "from", "with", "to", "for", "as", "at", "its", "their",
    "this", "that", "is", "are", "was", "be", "caused",
    "infection", "plant", "crop",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def tokenise(name: str) -> set:
    tokens = re.sub(r"[^a-z0-9\s]", " ", str(name).lower()).split()
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    sa, sb = tokenise(a), tokenise(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def contains_sub(a: str, b: str) -> bool:
    return a.lower().strip() in b.lower().strip() or b.lower().strip() in a.lower().strip()


def name_sim(a: str, b: str) -> float:
    return max(jaccard(a, b), 1.0 if contains_sub(a, b) else 0.0)


def pair_id(a: str, b: str) -> str:
    key = "_".join(sorted([a.lower().strip(), b.lower().strip()]))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def lambda_for_type_c(sim: float) -> float:
    """Type C pairs: context must override names -> low lambda (0.05–0.25)."""
    base = 0.05 + sim * 0.15
    base += random.uniform(-0.03, 0.03)
    return round(max(0.04, min(0.28, base)), 4)


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    df  = pd.read_csv(DATASET_IN)
    ent = pd.read_csv(ENTITIES_CSV)
    return df, ent


# ── Build existing pair index ──────────────────────────────────────────────────

def existing_pair_keys(df: pd.DataFrame) -> set:
    keys = set()
    for _, row in df.iterrows():
        keys.add(pair_id(str(row["name_a"]), str(row["name_b"])))
    return keys


# ── Filter entities to agricultural diseases ───────────────────────────────────

def get_disease_entities(ent: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with non-empty context and agricultural relevance
    mask = (
        ent["entity_type"].str.lower().isin(["disease"]) &
        ent["name"].notna() &
        ent["context"].notna() &
        (ent["context"].str.len() > 50)
    )
    return ent[mask].copy().reset_index(drop=True)


# ── Keyword index: keyword -> list of (entity_id, name, canonical, context) ────

def build_keyword_index(diseases: pd.DataFrame) -> dict:
    idx = defaultdict(list)
    for _, row in diseases.iterrows():
        tokens = tokenise(str(row["name"]))
        for kw in KEYWORDS:
            if kw in tokens:
                idx[kw].append({
                    "entity_id": row["entity_id"],
                    "name":      row["name"],
                    "canonical": row["canonical"],
                    "context":   str(row["context"])[:500],
                })
    return idx


# ── Generate candidate Type C pairs ───────────────────────────────────────────

def generate_candidates(kw_idx: dict, existing_keys: set) -> list:
    candidates = []
    per_seed_count = defaultdict(int)

    for kw, entities in kw_idx.items():
        # Shuffle to get variety
        entities = list(entities)
        random.shuffle(entities)

        for e1, e2 in combinations(entities, 2):
            # Must be DIFFERENT canonicals (different diseases)
            if e1["canonical"] == e2["canonical"]:
                continue

            na, nb = e1["name"], e2["name"]
            sim = name_sim(na, nb)

            # Must qualify as Type C (similar names)
            if sim < NAME_SIM_FLOOR:
                continue

            key = pair_id(na, nb)
            if key in existing_keys:
                continue

            if per_seed_count[kw] >= MAX_PER_SEED:
                continue

            lam = lambda_for_type_c(sim)

            candidates.append({
                "name_a":         na,
                "context_a":      e1["context"],
                "canonical_id_a": e1["canonical"],
                "name_b":         nb,
                "context_b":      e2["context"],
                "canonical_id_b": e2["canonical"],
                "match":          0,
                "llm_match":      0,
                "lambda_val":     lam,
                "lambda_source":  f"augmented_type_c_{kw}",
                "pair_type":      "C",
                "name_sim_score": round(sim, 4),
                "context_quality_a": "medium",
                "context_quality_b": "medium",
                "exclude_from_lambda": False,
                "pair_key":       key,
                "seed_keyword":   kw,
            })

            per_seed_count[kw] += 1
            existing_keys.add(key)

    return candidates


# ── Deduplicate and select best candidates ─────────────────────────────────────

def select_candidates(candidates: list, target: int) -> list:
    if not candidates:
        return []

    # Score: prefer higher name_sim (harder for the model) and diverse keywords
    candidates.sort(key=lambda x: -x["name_sim_score"])

    # Ensure keyword diversity: max 30 per keyword in final selection
    kw_counts = defaultdict(int)
    selected = []
    KW_CAP = 30

    for c in candidates:
        kw = c["seed_keyword"]
        if kw_counts[kw] < KW_CAP and len(selected) < target:
            selected.append(c)
            kw_counts[kw] += 1

    # If still under target, fill remaining without cap
    if len(selected) < target:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:target - len(selected)])

    return selected[:target]


# ── Build final rows matching dataset schema ───────────────────────────────────

def build_rows(selected: list, existing_df: pd.DataFrame) -> pd.DataFrame:
    cols = list(existing_df.columns)

    rows = []
    for c in selected:
        row = {col: None for col in cols}
        row["name_a"]              = c["name_a"]
        row["context_a"]           = c["context_a"]
        row["canonical_id_a"]      = c["canonical_id_a"]
        row["name_b"]              = c["name_b"]
        row["context_b"]           = c["context_b"]
        row["canonical_id_b"]      = c["canonical_id_b"]
        row["match"]               = 0
        row["llm_match"]           = 0
        row["lambda_val"]          = c["lambda_val"]
        row["lambda_source"]       = c["lambda_source"]
        row["pair_type"]           = "C"
        row["name_sim_score"]      = c["name_sim_score"]
        row["exclude_from_lambda"] = False

        # Fill optional quality columns if they exist
        if "context_quality_a" in cols:
            row["context_quality_a"] = "medium"
        if "context_quality_b" in cols:
            row["context_quality_b"] = "medium"
        if "pair_type_reason" in cols:
            row["pair_type_reason"] = f"no-match + similar names (polysemy) — keyword: {c['seed_keyword']}"

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(existing_df, new_df, selected, report_path):
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    pt_before = existing_df["pair_type"].value_counts()
    pt_after  = combined["pair_type"].value_counts()

    from collections import Counter
    kw_dist = Counter(c["seed_keyword"] for c in selected)
    sim_buckets = [(0.25,0.4),(0.4,0.55),(0.55,0.7),(0.7,0.85),(0.85,1.01)]

    lines = []
    lines.append("=" * 65)
    lines.append(" AUGMENTATION REPORT — Type C Expansion")
    lines.append("=" * 65)
    lines.append(f"\nOriginal rows  : {len(existing_df)}")
    lines.append(f"New Type C rows: {len(new_df)}")
    lines.append(f"Total rows     : {len(combined)}")

    lines.append("\n── Pair type counts (before -> after) ──────────────────────")
    for t in ["A","B","C","D"]:
        b = pt_before.get(t, 0)
        a = pt_after.get(t, 0)
        lines.append(f"  Type {t}: {b:5d}  ->  {a:5d}  (+{a-b})")

    lines.append("\n── New Type C: keyword distribution ───────────────────────")
    for kw, cnt in kw_dist.most_common():
        bar = "█" * (cnt // 2)
        lines.append(f"  {kw:<12s}: {cnt:4d}  {bar}")

    lines.append("\n── New Type C: name_sim_score distribution ─────────────────")
    sims = [c["name_sim_score"] for c in selected]
    for lo, hi in sim_buckets:
        n = sum(1 for s in sims if lo <= s < hi)
        bar = "█" * (n // 3)
        lines.append(f"  [{lo:.2f}–{hi:.2f}): {n:4d}  {bar}")

    lines.append("\n── New Type C: lambda_val distribution ─────────────────────")
    lams = [c["lambda_val"] for c in selected]
    lam_buckets = [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4)]
    for lo, hi in lam_buckets:
        n = sum(1 for l in lams if lo <= l < hi)
        bar = "█" * (n // 3)
        lines.append(f"  [{lo:.2f}–{hi:.2f}): {n:4d}  {bar}")

    lines.append("\n── Sample new Type C pairs ─────────────────────────────────")
    for c in selected[:10]:
        lines.append(f"  name_a : {c['name_a'][:55]}")
        lines.append(f"  name_b : {c['name_b'][:55]}")
        lines.append(f"  sim={c['name_sim_score']:.3f}  lambda={c['lambda_val']:.3f}  kw={c['seed_keyword']}")
        lines.append("")

    lines.append("\n── Lambda distribution of FULL augmented dataset ────────────")
    all_lam = combined["lambda_val"].dropna()
    for lo, hi in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)]:
        n = ((all_lam >= lo) & (all_lam < hi)).sum()
        pct = n / len(all_lam) * 100
        bar = "█" * (n // 20)
        lines.append(f"  [{lo:.1f}–{hi:.1f}): {n:5d} ({pct:5.1f}%)  {bar}")

    report = "\n".join(lines)
    print(report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved -> {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Type C Augmentation -- AgriLambdaNet Entity Resolution")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df, ent = load_data()
    print(f"  Dataset rows   : {len(df)}")
    print(f"  Entity rows    : {len(ent)}")

    current_c = (df["pair_type"] == "C").sum() if "pair_type" in df.columns else 0
    print(f"  Current Type C : {current_c}")

    print("\n[2/6] Building existing pair index...")
    existing_keys = existing_pair_keys(df)
    print(f"  Indexed {len(existing_keys)} existing pairs")

    print("\n[3/6] Filtering disease entities...")
    diseases = get_disease_entities(ent)
    print(f"  Disease entities with context: {len(diseases)}")

    print("\n[4/6] Building keyword index...")
    kw_idx = build_keyword_index(diseases)
    for kw, ents in sorted(kw_idx.items(), key=lambda x: -len(x[1])):
        print(f"  {kw:<14s}: {len(ents)} entities")

    print("\n[5/6] Generating Type C candidates...")
    candidates = generate_candidates(kw_idx, existing_keys)
    print(f"  Raw candidates : {len(candidates)}")

    selected = select_candidates(candidates, TARGET_NEW_C)
    print(f"  Selected       : {len(selected)}")

    if not selected:
        print("\n  WARNING: No new Type C pairs generated.")
        print("  This may mean all cross-keyword pairs already exist.")
        return

    print("\n[6/6] Building augmented dataset...")
    new_rows_df = build_rows(selected, df)
    augmented   = pd.concat([df, new_rows_df], ignore_index=True)
    augmented.to_csv(DATASET_OUT, index=False)
    print(f"  Saved -> {DATASET_OUT}")
    print(f"  Total rows: {len(augmented)}")

    write_report(df, new_rows_df, selected, REPORT_OUT)

    print("\n" + "=" * 60)
    print(f" Done. {len(selected)} new Type C pairs added.")
    print(f" Type C total: {current_c} -> {current_c + len(selected)}")
    print(f" Output: {DATASET_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
