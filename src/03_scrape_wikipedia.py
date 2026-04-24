"""
03_scrape_wikipedia.py  -  Scrape disease descriptions from Wikipedia
AgriNet Entity Resolution Pipeline

===========================================================================
BEGINNER EXPLANATION  -  What is this script doing?
===========================================================================

Wikipedia has a page for almost every crop disease. Each page has:
  - The disease name as the page title
  - Alternative names listed in the intro paragraph (often in bold)
  - A description in the first paragraph
  - The pathogen name, affected crops, symptoms

We use the Wikipedia API (not scraping HTML directly) because:
  1. Wikipedia allows it  -  it's legal and ethical
  2. The API returns clean JSON, much easier to parse
  3. The API is fast and reliable

API endpoint we use:
  https://en.wikipedia.org/w/api.php
  action=query
  titles=Late_blight
  prop=extracts
  exintro=1   ← only the intro paragraph (not the whole article)
  format=json

We also search for bold text ('''like this''') in Wikipedia markup
because that's how Wikipedia indicates ALTERNATIVE NAMES.
E.g., "'''Late blight''', also called '''potato blight''' ..."
These become positive pairs for training.
===========================================================================
"""

import sys
import re
import time
import itertools
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import build_entity_record, save_raw, normalise_name, PAIRS_DIR

SOURCE_NAME = "wikipedia"
WIKI_API    = "https://en.wikipedia.org/w/api.php"

# Wikipedia blocks requests that don't identify themselves.
# This header tells Wikipedia who we are  -  it is required by their API policy.
# See: https://www.mediawiki.org/wiki/API:Etiquette
HEADERS = {
    "User-Agent": "AgriLambdaNet-DatasetBuilder/1.0 (academic research; crop disease entity resolution) python-requests"
}

# ─────────────────────────────────────────────────────────────────────────────
# List of Wikipedia page titles to scrape
# These are the 50 most important crop disease pages
# ─────────────────────────────────────────────────────────────────────────────

WIKIPEDIA_PAGES = [
    # Potato & Tomato
    "Late_blight",
    "Early_blight_(plant_disease)",
    "Potato_virus_Y",

    # Wheat diseases
    "Wheat_streak_mosaic_virus",
    "Fusarium_head_blight",
    "Septoria_tritici_blotch",
    "Tan_spot",
    "Powdery_mildew_of_cereals_and_grasses",
    "Yellow_rust",
    "Leaf_rust_of_wheat",
    "Stem_rust",

    # Rice diseases
    "Rice_blast",
    "Rice_brown_spot",
    "Bacterial_leaf_blight_of_rice",
    "Sheath_blight",

    # Maize/Corn
    "Northern_corn_leaf_blight",
    "Gray_leaf_spot_of_maize",
    "Corn_smut",
    "Maize_streak_virus",
    "Goss%27s_wilt",

    # Soybean
    "Asian_soybean_rust",
    "Sudden_death_syndrome_of_soybean",
    "Soybean_mosaic_virus",
    "Phytophthora_root_rot",
    "White_mold",

    # Fruit diseases
    "Apple_scab",
    "Fire_blight",
    "Grape_powdery_mildew",
    "Grape_downy_mildew",
    "Botrytis_cinerea",
    "Citrus_greening_disease",
    "Citrus_canker",
    "Peach_leaf_curl",
    "Brown_rot_(fruit_disease)",
    "Cedar-apple_rust",

    # Banana
    "Panama_disease",
    "Black_Sigatoka",
    "Banana_bunchy_top_virus",

    # Coffee
    "Coffee_leaf_rust",

    # General
    "Powdery_mildew",
    "Downy_mildew",
    "Fusarium_wilt",
    "Verticillium_wilt",
    "Anthracnose",
    "Damping_off",
    "Crown_gall",
    "Clubroot",
    "Sclerotinia_stem_rot",
    "Bacterial_blight",
]


def fetch_wikipedia_intro(page_title: str) -> dict:
    """
    WHAT THIS DOES:
      Calls the Wikipedia API to get full article intro and first sections.
      Uses HTML format to get more content without strict truncation.

    RETURNS:
      dict with keys: title, extract, wikitext (for alt name parsing)
      Returns None if page doesn't exist.
    """
    # First attempt: get more content with different params
    params = {
        "action":      "query",
        "titles":      page_title.replace("_", " "),
        "prop":        "extracts|revisions",
        "exlimit":     "max",        # Allow maximum extraction
        "explaintext": 0,            # Get HTML (allows more content)
        "exchars":     3000,         # Request up to 3000 chars
        "rvprop":      "content",    # Also get wikitext for bold extraction
        "rvsection":   0,            # Only intro section wikitext
        "format":      "json",
        "formatversion": 2,
    }

    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print(f"    Warning: Timeout: {page_title}")
        return None
    except Exception as e:
        print(f"    Warning: Error fetching {page_title}: {e}")
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        print(f"    Warning: Page not found: {page_title}")
        return None

    # Get HTML and strip tags to get plain text
    extract_html = page.get("extract", "")
    # Simple HTML tag removal
    import re as regex_module
    extract_text = regex_module.sub(r'<[^>]+>', '', extract_html)
    extract_text = regex_module.sub(r'&nbsp;', ' ', extract_text)
    extract_text = regex_module.sub(r'&amp;', '&', extract_text)
    extract_text = regex_module.sub(r'&quot;', '"', extract_text)
    extract_text = extract_text.strip()

    result = {
        "title":    page.get("title", page_title),
        "extract":  extract_text,
        "wikitext": "",
    }

    revisions = page.get("revisions", [])
    if revisions:
        result["wikitext"] = revisions[0].get("content", "")

    return result


def extract_bold_names(wikitext: str) -> list:
    """
    WHAT THIS DOES:
      Wikipedia uses '''triple quotes''' to mark bold text, which often
      marks alternative names for the disease in the intro paragraph.

      Example wikitext:
        "'''Late blight''', also known as '''potato blight''', is a disease..."

      This function returns: ["Late blight", "potato blight"]

    WHY THIS MATTERS:
      These bold names are Wikipedia's way of saying "these are all names
      for the same thing"  -  perfect positive pairs for entity resolution!
    """
    # Pattern: text surrounded by ''' ... '''
    bold_pattern = re.compile(r"'''(.*?)'''")
    matches = bold_pattern.findall(wikitext[:3000])  # Only look in intro

    # Clean up: remove wikilinks [[...]], HTML tags, etc.
    clean_names = []
    for match in matches:
        # Remove [[link|text]] → keep "text" part
        match = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", match)
        # Remove remaining wiki markup
        match = re.sub(r"\[.*?\]", "", match)
        match = re.sub(r"<[^>]+>", "", match)
        match = match.strip()

        if match and len(match) > 2 and len(match) < 100:
            clean_names.append(match)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for name in clean_names:
        n = name.lower()
        if n not in seen:
            seen.add(n)
            result.append(name)

    return result


def ensure_complete_sentence(text: str, target_length: int = 600) -> str:
    """
    WHAT THIS DOES:
      Wikipedia API limits output to ~300 chars.  
      This function finds the best sentence boundary closest to target_length.
      Since Wikipedia cuts mid-sentence, we find the LAST period before the cut.
      
    RETURNS:
      Text ending with a period or complete phrase (best effort).
    """
    text = text.strip()
    if not text:
        return text
    
    # If it's all we got, return it
    if len(text) <= 150:
        return text
    
    # Look for last period BEFORE position that might be cut
    # Typical Wikipedia cuts at 300-310 chars
    search_limit = min(len(text), 350)
    
    # Find the LAST period in the text (best complete sentence)
    last_period = text[:search_limit].rfind('.')
    
    if last_period > len(text) * 0.5:  # Found period in second half
        return text[:last_period + 1]
    
    # No good period found, find last sentence-like boundary
    for punct in ['. ', '!\n', '?\n', '. ', '!\t', '?\t']:
        idx = text[:search_limit].rfind(punct)
        if idx > 0:
            return text[:idx + 1]
    
    # Last resort: return what we have (Wikipedia's limit)
    return text


def scrape_all_pages() -> tuple:
    """
    WHAT THIS DOES:
      Loops through all Wikipedia pages, fetches their content,
      extracts the disease name, alt names, and description.
      Ensures contexts contain complete sentences.

    RETURNS:
      (entity_records, synonym_groups)
      entity_records: list of entity dicts for the CSV
      synonym_groups: list of (page_title, [all surface forms]) for pairs
    """
    entity_records = []
    synonym_groups = []

    print(f"  Fetching {len(WIKIPEDIA_PAGES)} Wikipedia pages...")
    print(f"  (1 second delay between requests  -  respecting Wikipedia's API)\n")

    for i, page_title in enumerate(WIKIPEDIA_PAGES, 1):
        print(f"  [{i:2d}/{len(WIKIPEDIA_PAGES)}] {page_title}", end="... ", flush=True)

        page_data = fetch_wikipedia_intro(page_title)
        if page_data is None:
            print("SKIP")
            continue

        title   = page_data["title"]
        full_text = page_data["extract"]
        # Get around 600 chars but ensure complete sentence
        extract = ensure_complete_sentence(full_text, target_length=600)
        wikitext= page_data["wikitext"]

        # Extract alternative names from bold text in intro
        alt_names = extract_bold_names(wikitext)
        print(f"OK  -  {len(alt_names)} alt names")

        # The page title becomes the canonical entity
        canonical_record = build_entity_record(
            name=title,
            context=extract,
            source=SOURCE_NAME,
            source_url=f"https://en.wikipedia.org/wiki/{page_title}",
            entity_type="Disease",
        )
        entity_records.append(canonical_record)

        # Each alt name also gets a record (linked to same context)
        for alt in alt_names:
            if alt.lower() != title.lower():
                alt_record = build_entity_record(
                    name=alt,
                    context=extract,   # Same context as canonical
                    source=SOURCE_NAME,
                    source_url=f"https://en.wikipedia.org/wiki/{page_title}",
                    entity_type="Disease",
                )
                entity_records.append(alt_record)

        # Store synonym group for pair building
        if alt_names:
            all_forms = [title] + [a for a in alt_names if a.lower() != title.lower()]
            synonym_groups.append({
                "page":  page_title,
                "forms": all_forms,
                "context": extract,
            })

        # ── Be polite to Wikipedia  -  don't hammer their API ─────────────────
        time.sleep(1.0)

    return entity_records, synonym_groups


def build_wikipedia_pairs(synonym_groups: list) -> list:
    """
    WHAT THIS DOES:
      For each Wikipedia page, pair all its bold-named alt forms together.
      All pairs get label=1 (same entity).
    """
    pairs = []
    pair_id = 1

    for group in synonym_groups:
        forms = group["forms"]
        if len(forms) < 2:
            continue

        for name_1, name_2 in itertools.combinations(forms, 2):
            pairs.append({
                "pair_id":     f"WP_POS_{pair_id:04d}",
                "name_1":      name_1,
                "name_2":      name_2,
                "canonical_1": normalise_name(name_1),
                "canonical_2": normalise_name(name_2),
                "entity_type": "Disease",
                "label":       1,
                "pair_source": f"wikipedia_{group['page']}",
                "confidence":  0.9,  # Slightly less than 1.0  -  Wikipedia isn't perfect
                "note":        f"Bold-text synonyms from Wikipedia:{group['page']}",
            })
            pair_id += 1

    return pairs


def main():
    print("\n" + "="*60)
    print("  SCRIPT 03  -  Wikipedia Disease Scraper")
    print("  AgriNet Entity Resolution Pipeline")
    print("="*60)

    print("\n[1/3] Scraping Wikipedia pages...")
    entity_records, synonym_groups = scrape_all_pages()

    print(f"\n[2/3] Saving entity records...")
    df = save_raw(entity_records, "wikipedia_raw.csv")
    print(f"  Pages with alt names found: {len(synonym_groups)}")

    print(f"\n[3/3] Building positive pairs from alt names...")
    pairs = build_wikipedia_pairs(synonym_groups)
    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame()
    if not pairs_df.empty:
        out_path = PAIRS_DIR / "wikipedia_pairs_positive.csv"
        pairs_df.to_csv(out_path, index=False)
        print(f"  OK Saved {len(pairs_df)} positive pairs -> {out_path.name}")
    else:
        print("  Warning: No pairs found (possible API issue). Check internet connection.")

    print(f"\n  [========================================]")
    print(f"  | Pages scraped:           {len(synonym_groups):5d}          |")
    print(f"  | Entity records:          {len(df):5d}          |")
    print(f"  | Positive pairs:          {len(pairs_df):5d}          |")
    print(f"  [========================================]")
    print("\n  DONE Script 03 complete! Next: run 04_extract_kg_triples.py\n")

    return df, pairs_df


if __name__ == "__main__":
    main()