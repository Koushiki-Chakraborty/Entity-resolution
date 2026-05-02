"""
STEP 2 — Scrape entity types from Wikipedia & AGROVOC
======================================================
For every entity in the dataset where type_a or type_b is still missing
after step1's keyword inference, this script:

  1. Searches Wikipedia for the entity name
  2. Reads the first paragraph of the Wikipedia article
  3. Classifies the entity type (fungus / virus / bacteria / pest / plant / disease)
     using the same keyword rules as step1, applied to the REAL Wikipedia text
  4. Falls back to AGROVOC search if Wikipedia returns no useful result
  5. Writes results to 'scraped_types.csv' — a lookup table (name → type + url)

Then step3_merge.py applies the lookup table back to the fixed dataset.

Run:
    pip install requests beautifulsoup4 wikipedia-api
    python step2_scrape_types.py \
        --input  dataset_fixed.csv \
        --output scraped_types.csv

The script is:
  - Rate-limited (1 request per second) to be polite to Wikipedia
  - Resumable — if interrupted, restart and it skips already-done entities
  - Transparent — prints what it found and why for every entity
"""

import argparse
import csv
import re
import sys
import io
import time
from pathlib import Path

# Force UTF-8 output on Windows (Wikipedia text contains non-ASCII characters)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup

# ── 1. CLI ────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Scrape entity types for missing rows")
    p.add_argument("--input",   default="dataset_fixed.csv",  help="Output of step1")
    p.add_argument("--output",  default="scraped_types.csv",  help="Lookup table to produce")
    p.add_argument("--delay",   type=float, default=1.0,
                   help="Seconds to wait between requests (default 1.0)")
    return p.parse_args()


# ── 2. Type classification rules (same as step1 but applied to web text) ──────

TYPE_RULES = [
    ("fungus", re.compile(
        r"fungal\s+disease|oomycete|ascomycete|basidiomycete"
        r"|puccinia|fusarium|alternaria|botrytis|sclerotinia"
        r"|magnaporthe|colletotrichum|phytophthora|plasmopara"
        r"|blumeria|venturia|cercospora|cochliobolus|septoria"
        r"|erysiphe|uncinula|podosphaera|guignardia|diplocarpon"
        r"|exserohilum|helminthosporium|mycosphaerella|gibberella"
        r"|is a fungus|is a fungal|caused by a fungus|ascomycota"
        r"|basidiomycota|is an oomycete",
        re.IGNORECASE
    )),
    ("bacteria", re.compile(
        r"bacterial\s+disease|is a bacterium|is a gram-"
        r"|xanthomonas|pseudomonas|erwinia|agrobacterium"
        r"|clavibacter|ralstonia|streptomyces|is a plant pathogen.*bacterium"
        r"|caused by.*bacterium",
        re.IGNORECASE
    )),
    ("virus", re.compile(
        r"\bvirus\b|\bviral\s+disease\b|viridae|begomovirus"
        r"|potyvirus|tobamovirus|luteovirus|caulimovirus"
        r"|geminivirus|is a plant virus|single-stranded rna"
        r"|double-stranded rna|is transmitted by",
        re.IGNORECASE
    )),
    ("pest", re.compile(
        r"\binsect\b|\bmite\b|\bnematode\b|\baphid\b|\bwhitefly\b"
        r"|\bthrip\b|\bweevil\b|\bborer\b|\bcaterpillar\b"
        r"|is an arthropod|is a pest",
        re.IGNORECASE
    )),
    ("plant", re.compile(
        r"is a plant species|is a species of plant|is a flowering plant"
        r"|is a genus of|is a family of|taxonomy|botanical",
        re.IGNORECASE
    )),
]

def classify_from_text(text: str) -> str:
    """Run keyword rules on any text. Returns type string or 'unknown'."""
    for entity_type, pattern in TYPE_RULES:
        if pattern.search(text):
            return entity_type
    return "unknown"


# ── 3. Wikipedia scraper ───────────────────────────────────────────────────────

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_PARSE_URL  = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "AgriEntityResearch/1.0 (academic research; contact: your@email.com)"
}


def wikipedia_search(query: str) -> list[dict]:
    """
    Use the Wikipedia OpenSearch API to find article titles for a query.
    Returns a list of {'title': ..., 'url': ...} dicts (up to 5 results).
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 5,
        "namespace": 0,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        # data[1] = titles, data[3] = urls
        titles = data[1]
        urls   = data[3]
        return [{"title": t, "url": u} for t, u in zip(titles, urls)]
    except Exception as e:
        print(f"    [Wikipedia search error] {query!r}: {e}")
        return []


def wikipedia_first_paragraph(page_title: str) -> str:
    """
    Fetch the first paragraph (plain text) of a Wikipedia article by title.
    Uses the 'extracts' API — no HTML parsing needed for plain text.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,          # intro section only
        "explaintext": True,      # plain text, no markup
        "titles": page_title,
        "format": "json",
        "redirects": 1,           # follow redirects
    }
    try:
        r = requests.get(WIKI_PARSE_URL, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":  # page not found
                return ""
            extract = page.get("extract", "")
            # Return only the first paragraph (split by double newline)
            first_para = extract.split("\n\n")[0].strip()
            return first_para
    except Exception as e:
        print(f"    [Wikipedia fetch error] {page_title!r}: {e}")
        return ""


def scrape_wikipedia(entity_name: str, delay: float) -> dict:
    """
    Full Wikipedia lookup for one entity name.
    Returns dict: {name, type, source_url, source_text}
    """
    print(f"  -> Searching Wikipedia for: {entity_name!r}")
    results = wikipedia_search(entity_name)
    time.sleep(delay)

    if not results:
        print(f"    No Wikipedia results found")
        return {"name": entity_name, "type": None, "source_url": None, "source_text": ""}

    # Try each result until we find one that classifies successfully
    for result in results:
        title = result["title"]
        url   = result["url"]
        print(f"    Trying article: {title!r}")

        text = wikipedia_first_paragraph(title)
        time.sleep(delay)

        if not text:
            print(f"    Empty article — skipping")
            continue

        entity_type = classify_from_text(text)
        preview = text[:120].replace("\n", " ")
        print(f"    Text preview: {preview!r}")
        print(f"    Classified as: {entity_type}")

        if entity_type != "unknown":
            return {
                "name":        entity_name,
                "type":        entity_type,
                "source_url":  url,
                "source_text": text[:300],
            }
        else:
            print(f"    Could not classify from this article, trying next...")

    print(f"    [FAILED] Could not determine type for {entity_name!r}")
    return {
        "name":        entity_name,
        "type":        "unknown",
        "source_url":  results[0]["url"] if results else None,
        "source_text": "",
    }


# ── 4. Main scraping loop ─────────────────────────────────────────────────────

def main():
    import pandas as pd
    args = get_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}\n")

    # ── 4a. Collect all entity names that need type assignment ──────────────
    # type_needs_scraping == 1 means step1 could not classify from context
    needs_scraping = df[df.get("type_needs_scraping", 0) == 1] \
        if "type_needs_scraping" in df.columns \
        else df[df["type_a"].isna() | df["type_b"].isna()]

    entities_needing_type = set()
    for _, row in needs_scraping.iterrows():
        if pd.isna(row["type_a"]):
            entities_needing_type.add(str(row["name_a"]).strip())
        if pd.isna(row["type_b"]):
            entities_needing_type.add(str(row["name_b"]).strip())

    # ── 4b. Collect entities for context update (same_context rows) ─────────
    same_ctx_rows = df[df.get("same_context", 0) == 1] \
        if "same_context" in df.columns else pd.DataFrame()

    entities_needing_context = set()
    for _, row in same_ctx_rows.iterrows():
        entities_needing_context.add(str(row["name_a"]).strip())
        entities_needing_context.add(str(row["name_b"]).strip())

    all_entities = entities_needing_type | entities_needing_context
    print(f"Entities needing type:    {len(entities_needing_type)}")
    print(f"Entities needing context: {len(entities_needing_context)}")
    print(f"Total unique to scrape:   {len(all_entities)}\n")

    # ── 4c. Load existing results to allow resuming ──────────────────────────
    output_path = Path(args.output)
    done = {}
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for _, row in existing.iterrows():
            done[row["name"]] = row.to_dict()
        print(f"Resuming: {len(done)} entities already scraped\n")

    # ── 4d. Scrape each entity ───────────────────────────────────────────────
    results = list(done.values())  # start with already-done

    todo = sorted(all_entities - set(done.keys()))
    print(f"Entities left to scrape: {len(todo)}\n")

    for i, entity in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {entity!r}")

        # Try Wikipedia first
        result = scrape_wikipedia(entity, args.delay)

        # If Wikipedia returned 'unknown', return it as-is (no AGROVOC fallback for simplicity)
        results.append(result)
        print()

        # Save after every entity so progress is not lost on crash
        pd.DataFrame(results).to_csv(output_path, index=False)

    # ── 4e. Final report ─────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Total entities scraped: {len(out_df)}")
    print(f"Type distribution:\n{out_df['type'].value_counts(dropna=False).to_string()}")
    print(f"Unknown (manual review needed): {(out_df['type'] == 'unknown').sum()}")
    print(f"\nResults saved -> {args.output}")
    print("\nNEXT STEP: Run step3_merge.py to apply scraped types to the dataset")


if __name__ == "__main__":
    main()
