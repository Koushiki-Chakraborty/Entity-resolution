"""
utils.py — Shared helper functions for the CropDP-KGAgriNER dataset builder
AgriΛNet Entity Resolution Pipeline

WHAT IS THIS FILE?
  Instead of writing the same code 4 times (once per scraper), we put
  common functions here and import them. Think of it like a shared toolbox.

HOW TO USE:
  from utils import build_entity_record, save_raw, clean_context
"""

import re
import hashlib
import unicodedata
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Set up folder paths so every script knows where to save files
# ─────────────────────────────────────────────────────────────────────────────

# Path(__file__).parent  →  the folder THIS file is in  (entity_resolution/src/)
# .parent again          →  one level up                (entity_resolution/)
ROOT          = Path(__file__).parent.parent
RAW_DIR       = ROOT / "data" / "raw"        # scraped, uncleaned data goes here
PROCESSED_DIR = ROOT / "data" / "processed"  # cleaned merged data goes here
PAIRS_DIR     = ROOT / "data" / "pairs"      # final ML training pairs go here

# Create folders if they don't exist yet
for d in [RAW_DIR, PROCESSED_DIR, PAIRS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Entity Type Taxonomy
#
# Your KG triples file already has great type labels. We reuse them.
# For each type, we list "keywords" that help us guess the type automatically
# when scraping new data from PlantVillage or Wikipedia.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = {
    "Disease":      ["blight", "rust", "wilt", "mildew", "rot", "spot",
                     "mosaic", "smut", "scab", "canker", "necrosis",
                     "disease", "infection", "pathogen", "virus", "fungal"],
    "Crop":         ["wheat", "maize", "corn", "rice", "tomato", "potato",
                     "soybean", "cotton", "barley", "sorghum", "mango",
                     "banana", "cassava", "groundnut", "sugarcane", "crop"],
    "Organism":     ["phytophthora", "fusarium", "alternaria", "botrytis",
                     "xanthomonas", "pseudomonas", "pythium", "nematode",
                     "fungus", "bacterium", "oomycete", "cercospora"],
    "Agri_Process": ["rotation", "irrigation", "harvest", "planting",
                     "cultivation", "tillage", "composting", "grafting",
                     "supply chain", "traceability"],
    "Agri_Method":  ["ipm", "integrated pest", "hydroponics", "precision",
                     "organic farming", "biological control", "intercropping"],
    "Technology":   ["blockchain", "iot", "rfid", "gps", "remote sensing",
                     "machine learning", "ai", "sensor", "drone", "bct"],
}


def infer_entity_type(name: str, context: str = "") -> str:
    """
    WHAT THIS DOES:
      Looks at the name and description of an entity and guesses its type.
      For example: "Late Blight" → Disease   |   "Tomato" → Crop

    HOW IT WORKS:
      It converts everything to lowercase and checks if any keyword matches.
      The first matching type wins.
    """
    text = (name + " " + context).lower()
    for etype, keywords in ENTITY_TYPES.items():
        if any(kw in text for kw in keywords):
            return etype
    return "Disease"   # Default — since this is primarily a disease dataset


def normalise_name(name: str) -> str:
    """
    WHAT THIS DOES:
      Makes a "standard" version of any name so we can compare them fairly.

    EXAMPLES:
      "Late Blight"   → "late blight"
      "LATE  BLIGHT"  → "late blight"
      " late blight." → "late blight"

    WHY WE NEED THIS:
      Your lambda estimator needs to learn that "Late Blight" and "late blight"
      are the SAME entity. Normalising both to "late blight" helps detect this.
    """
    # Step A: Handle special Unicode characters (accents, special symbols)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Step B: lowercase everything
    name = name.lower().strip()

    # Step C: collapse multiple spaces into one
    name = re.sub(r"\s+", " ", name)

    # Step D: remove leading/trailing punctuation
    name = re.sub(r"^[\W_]+|[\W_]+$", "", name)

    return name


def make_entity_id(name: str, source: str) -> str:
    """
    WHAT THIS DOES:
      Creates a unique ID for each entity. Like a fingerprint.

    EXAMPLE:
      make_entity_id("Late Blight", "plantvillage") → "ENT_A3F9C1"

    WHY:
      When we merge datasets from 4 sources, we need stable IDs so the
      same entity always gets the same ID regardless of order.
    """
    raw = f"{source}::{normalise_name(name)}"
    # MD5 is a hash function — same input always gives same 6-char output
    h = hashlib.md5(raw.encode()).hexdigest()[:6].upper()
    return f"ENT_{h}"


def clean_context(text: str, max_chars: int = 300) -> str:
    """
    WHAT THIS DOES:
      Cleans up messy text from web pages and limits it to 300 characters.

    CLEANS:
      - Extra whitespace
      - Wikipedia citation numbers like [1], [23]
      - Wikipedia template tags like {{convert|...}}

    WHY 300 chars:
      Your lambda estimator will use this as a "description" feature.
      Too long = noisy. Too short = not enough signal. 300 is a sweet spot.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\d+\]", "", text)        # remove [1], [23] citations
    text = re.sub(r"{{.*?}}", "", text)         # remove {{wiki templates}}
    text = re.sub(r"\s+", " ", text).strip()   # clean again after removals

    if len(text) <= max_chars:
        return text
    # Cut at a word boundary so we don't cut mid-word
    return text[:max_chars].rsplit(" ", 1)[0]


def build_entity_record(name: str, context: str, source: str,
                         source_url: str, entity_type: str = None,
                         canonical: str = None) -> dict:
    """
    WHAT THIS DOES:
      Takes raw scraped data and packages it into a standardised dictionary
      (one row in our final CSV).

    PARAMETERS:
      name        : The surface form as found in the source ("Late Blight")
      context     : Short description
      source      : Where it came from ("plantvillage", "agrovoc", etc.)
      source_url  : The actual URL (for reproducibility / paper citation)
      entity_type : Optionally override auto-detection
      canonical   : Optionally override normalisation (rarely needed)

    RETURNS:
      A dict with all 7 columns of our entity schema.
    """
    if canonical is None:
        canonical = normalise_name(name)
    if entity_type is None:
        entity_type = infer_entity_type(name, context)

    return {
        "entity_id":   make_entity_id(name, source),
        "name":        name.strip(),
        "canonical":   canonical,
        "entity_type": entity_type,
        "context":     clean_context(context),
        "source":      source,
        "source_url":  source_url,
    }


def save_raw(records: list, filename: str) -> pd.DataFrame:
    """Save a list of dicts to data/raw/ as a CSV. Returns the DataFrame."""
    df = pd.DataFrame(records)
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {len(df):4d} records → {path.name}")
    return df


def load_all_raw() -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Reads all CSV files from data/raw/ and stacks them into one big table.
      This is how we merge PlantVillage + AGROVOC + Wikipedia + KG triples.
    """
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  Loaded {len(df):4d} rows from {f.name}")
    if not frames:
        raise FileNotFoundError("No CSV files found in data/raw/. Run scrapers first!")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  {'─'*35}")
    print(f"  TOTAL : {len(combined)} rows from {len(frames)} sources")
    return combined


def deduplicate_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      If the same entity appears twice (e.g., same canonical name, same type),
      keep the one with the LONGEST context (most information).

    WHY:
      PlantVillage and Wikipedia might both have "Late Blight".
      We want ONE canonical record for it, with the best description.
    """
    df = df.copy()
    df["_ctx_len"] = df["context"].fillna("").str.len()
    df = df.sort_values("_ctx_len", ascending=False)
    df = df.drop_duplicates(subset=["canonical", "entity_type"], keep="first")
    df = df.drop(columns=["_ctx_len"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER TO YOUR QUESTION: "What if context is missing or wrong?"
#
# This function lets you manually add or fix context for any entity.
# After running all scrapers, open all_entities.csv, find missing rows,
# and call this to patch them in.
# ─────────────────────────────────────────────────────────────────────────────

def patch_missing_context(df: pd.DataFrame, patches: dict) -> pd.DataFrame:
    """
    WHAT THIS DOES:
      Lets you manually fix missing or wrong context/fields in the dataset.

    HOW TO USE:
      patches = {
          "late blight": {
              "context": "A destructive disease of potato and tomato caused by Phytophthora infestans.",
              "entity_type": "Disease"
          },
          "alternaria solani": {
              "context": "Fungal pathogen causing early blight in tomato and potato."
          }
      }
      df = patch_missing_context(df, patches)

    The key is the CANONICAL form (lowercase normalised name).
    You can patch any column: context, entity_type, source_url, etc.
    """
    df = df.copy()
    patched_count = 0
    for canonical_key, field_updates in patches.items():
        mask = df["canonical"] == canonical_key
        if mask.sum() == 0:
            print(f"  ⚠ WARNING: '{canonical_key}' not found in dataset. Check spelling.")
            continue
        for field, value in field_updates.items():
            df.loc[mask, field] = value
        patched_count += 1

    print(f"  ✓ Patched {patched_count} entities")
    return df


def generate_name_variants(canonical: str) -> list:
    """
    WHAT THIS DOES:
      Given a canonical disease name, generates realistic surface-form
      variants. These become POSITIVE PAIRS for training.

    EXAMPLE:
      "late blight" →
        ["Late Blight", "LATE BLIGHT", "late-blight", "L. blight",
         "late blight disease", "LB"]

    WHY THIS MATTERS:
      Your lambda estimator must learn that all these mean the same thing.
      More variants = more training signal = better model.
    """
    variants = []
    words = canonical.split()

    # Variant 1: Title Case  → "Late Blight"
    variants.append(canonical.title())

    # Variant 2: ALL CAPS  → "LATE BLIGHT"
    variants.append(canonical.upper())

    # Variant 3: Hyphenated  → "late-blight"
    if len(words) > 1:
        variants.append("-".join(words))

    # Variant 4: With "disease" appended  → "late blight disease"
    if "disease" not in canonical:
        variants.append(canonical + " disease")

    # Variant 5: Abbreviated  → "LB" (first letters of each word)
    if len(words) > 1:
        abbrev = "".join(w[0].upper() for w in words)
        variants.append(abbrev)

    # Variant 6: First word only (if multi-word)  → "late"
    if len(words) > 1:
        variants.append(words[0])

    # Variant 7: "of <crop>" appended if it contains a crop name
    for crop in ["tomato", "potato", "wheat", "maize", "rice", "corn"]:
        if crop in canonical:
            variants.append(canonical.replace(crop, "").strip() + f" of {crop}")
            break

    # Remove duplicates and the original itself
    seen = {canonical}
    clean = []
    for v in variants:
        v_norm = normalise_name(v)
        if v_norm not in seen and v_norm != "":
            seen.add(v_norm)
            clean.append(v)

    return clean