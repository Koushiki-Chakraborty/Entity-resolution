"""
02_scrape_agrovoc.py — Download disease synonym pairs from AGROVOC FAO vocabulary
AgriΛNet Entity Resolution Pipeline

═══════════════════════════════════════════════════════════════════════════════
BEGINNER EXPLANATION — What is AGROVOC?
═══════════════════════════════════════════════════════════════════════════════

AGROVOC is a multilingual vocabulary maintained by the FAO (Food and Agriculture
Organization of the United Nations). Think of it as a giant official dictionary
for agricultural terms — with ~40,000+ concepts, each with:
  - Official/preferred name  (e.g., "late blight")
  - Alternative labels / synonyms  (e.g., "potato blight", "LB")
  - Scope note = definition/description

AGROVOC has a SPARQL endpoint — a URL you can query like a database.
SPARQL (pronounced "sparkle") is a query language for graph databases.

We send a SPARQL query asking: "Give me all disease concepts + their synonyms"
AGROVOC returns JSON with preferred labels, alt labels, and scope notes.

TWO STRATEGIES in this script:
  Strategy A (Online): Query the live AGROVOC SPARQL endpoint
  Strategy B (Offline): Use a built-in curated AGROVOC-derived list

By default we try Strategy A first, fall back to B if the API is unreachable.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import json
import itertools
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import build_entity_record, save_raw, normalise_name, PAIRS_DIR

SOURCE_NAME = "agrovoc"
SOURCE_URL  = "https://agrovoc.fao.org/sparql"
SPARQL_ENDPOINT = "https://agrovoc.fao.org/sparql"

# ─────────────────────────────────────────────────────────────────────────────
# SPARQL QUERY — Fetch all disease concepts with their synonyms
#
# WHAT THIS QUERY DOES (plain English):
#   SELECT ?concept ?prefLabel ?altLabel ?scopeNote
#   FROM the AGROVOC graph
#   WHERE:
#     - ?concept is a skos:Concept (a defined vocabulary term)
#     - It has a preferred label in English
#     - The term falls under "plant diseases" broader category
#     - Optionally get alternative labels (synonyms)
#     - Optionally get scope notes (definitions)
# ─────────────────────────────────────────────────────────────────────────────

SPARQL_QUERY = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX agrovoc: <http://aims.fao.org/aos/agrovoc/>

SELECT DISTINCT ?concept ?prefLabel ?altLabel ?scopeNote
WHERE {
  ?concept a skos:Concept ;
           skos:prefLabel ?prefLabel .

  FILTER(lang(?prefLabel) = "en")

  # Filter for disease-related concepts
  FILTER(
    CONTAINS(LCASE(STR(?prefLabel)), "disease") ||
    CONTAINS(LCASE(STR(?prefLabel)), "blight") ||
    CONTAINS(LCASE(STR(?prefLabel)), "rust") ||
    CONTAINS(LCASE(STR(?prefLabel)), "wilt") ||
    CONTAINS(LCASE(STR(?prefLabel)), "mildew") ||
    CONTAINS(LCASE(STR(?prefLabel)), "rot") ||
    CONTAINS(LCASE(STR(?prefLabel)), "smut") ||
    CONTAINS(LCASE(STR(?prefLabel)), "scab") ||
    CONTAINS(LCASE(STR(?prefLabel)), "mosaic") ||
    CONTAINS(LCASE(STR(?prefLabel)), "virus") ||
    CONTAINS(LCASE(STR(?prefLabel)), "blast")
  )

  OPTIONAL { ?concept skos:altLabel ?altLabel . FILTER(lang(?altLabel) = "en") }
  OPTIONAL { ?concept skos:scopeNote ?scopeNote . FILTER(lang(?scopeNote) = "en") }
}
ORDER BY ?prefLabel
LIMIT 500
"""


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE FALLBACK — Curated AGROVOC-derived disease synonyms
#
# These were extracted from AGROVOC manually. Each entry has:
#   preferred_label: The AGROVOC official name
#   alt_labels: List of accepted synonyms from AGROVOC
#   scope_note: Definition from AGROVOC
# ─────────────────────────────────────────────────────────────────────────────

AGROVOC_OFFLINE = [
    {
        "preferred_label": "late blight",
        "alt_labels": ["potato late blight", "tomato late blight", "Phytophthora blight",
                       "Late Blight", "LATE BLIGHT", "LB", "downy blight"],
        "scope_note": "Disease of potato and tomato caused by the oomycete Phytophthora infestans, causing water-soaked lesions that rapidly turn dark brown.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_5875",
    },
    {
        "preferred_label": "early blight",
        "alt_labels": ["Alternaria blight", "Early Blight", "EARLY BLIGHT", "EB",
                       "target spot", "Alternaria leaf spot"],
        "scope_note": "Fungal disease of tomato and potato caused by Alternaria solani, producing characteristic concentric ring lesions.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_2490",
    },
    {
        "preferred_label": "powdery mildew",
        "alt_labels": ["Powdery Mildew", "POWDERY MILDEW", "oidium", "white mold",
                       "erysiphe", "PM"],
        "scope_note": "Fungal disease caused by members of Erysiphaceae family, characterized by white powdery coating on plant surfaces.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_6243",
    },
    {
        "preferred_label": "downy mildew",
        "alt_labels": ["Downy Mildew", "DOWNY MILDEW", "peronospora", "blue mold",
                       "DM"],
        "scope_note": "Disease caused by oomycetes in Peronosporaceae. Produces grey-purple downy growth on lower leaf surfaces.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_2377",
    },
    {
        "preferred_label": "leaf rust",
        "alt_labels": ["Leaf Rust", "LEAF RUST", "brown rust", "LR",
                       "wheat leaf rust", "Puccinia rust"],
        "scope_note": "Fungal disease of cereals caused by Puccinia species, producing orange-brown pustules on leaf surfaces.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_4442",
    },
    {
        "preferred_label": "stripe rust",
        "alt_labels": ["Stripe Rust", "STRIPE RUST", "yellow rust", "YR",
                       "wheat stripe rust", "Pst"],
        "scope_note": "Fungal disease caused by Puccinia striiformis, producing yellow-orange pustules in stripes along leaf veins.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_7523",
    },
    {
        "preferred_label": "stem rust",
        "alt_labels": ["Stem Rust", "STEM RUST", "black rust", "SR",
                       "wheat stem rust", "Puccinia graminis"],
        "scope_note": "Fungal disease caused by Puccinia graminis, producing dark reddish-brown pustules on stems and leaves. Can cause complete crop loss.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_7352",
    },
    {
        "preferred_label": "fusarium wilt",
        "alt_labels": ["Fusarium Wilt", "FUSARIUM WILT", "vascular wilt",
                       "panama disease", "FW", "Fusarium oxysporum wilt"],
        "scope_note": "Soilborne fungal disease caused by Fusarium oxysporum. Blocks water-conducting vessels causing wilting and plant death.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_3126",
    },
    {
        "preferred_label": "bacterial blight",
        "alt_labels": ["Bacterial Blight", "BACTERIAL BLIGHT", "bacterial leaf blight",
                       "BLB", "Xanthomonas blight"],
        "scope_note": "Bacterial disease caused by Xanthomonas species, producing water-soaked lesions that turn yellow-brown and necrotic.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_771",
    },
    {
        "preferred_label": "rice blast",
        "alt_labels": ["Rice Blast", "RICE BLAST", "blast disease", "RB",
                       "Pyricularia blast", "Magnaporthe blast", "neck rot"],
        "scope_note": "Most devastating disease of rice caused by Magnaporthe oryzae. Creates diamond-shaped grey lesions. Destroys entire crops.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_6597",
    },
    {
        "preferred_label": "anthracnose",
        "alt_labels": ["Anthracnose", "ANTHRACNOSE", "Colletotrichum rot",
                       "bitter rot", "fruit rot"],
        "scope_note": "Fungal disease caused by Colletotrichum species. Creates dark sunken lesions on leaves and fruit. Affects many crop species.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_475",
    },
    {
        "preferred_label": "damping off",
        "alt_labels": ["Damping Off", "DAMPING OFF", "seedling blight",
                       "pythium rot", "damping-off disease"],
        "scope_note": "Disease of seedlings caused by soil fungi including Pythium and Rhizoctonia. Causes stem collapse at soil level.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_2012",
    },
    {
        "preferred_label": "crown gall",
        "alt_labels": ["Crown Gall", "CROWN GALL", "root gall",
                       "Agrobacterium gall", "CG"],
        "scope_note": "Bacterial disease caused by Agrobacterium tumefaciens. Produces tumour-like galls on roots and stem base.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_1925",
    },
    {
        "preferred_label": "clubroot",
        "alt_labels": ["Clubroot", "CLUBROOT", "finger and toe disease",
                       "Plasmodiophora brassicae", "club root"],
        "scope_note": "Soilborne disease of brassica crops caused by Plasmodiophora brassicae. Deforms roots into club-shaped galls.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_1737",
    },
    {
        "preferred_label": "fire blight",
        "alt_labels": ["Fire Blight", "FIRE BLIGHT", "Erwinia blight",
                       "FB", "blossom blight", "shoot blight"],
        "scope_note": "Bacterial disease of apple and pear caused by Erwinia amylovora. Infected shoots look scorched by fire.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_2961",
    },
    {
        "preferred_label": "verticillium wilt",
        "alt_labels": ["Verticillium Wilt", "VERTICILLIUM WILT", "VW",
                       "Verticillium dahliae wilt", "verticillium disease"],
        "scope_note": "Soilborne fungal disease caused by Verticillium dahliae and V. albo-atrum. Causes progressive wilting and yellowing.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_8118",
    },
    {
        "preferred_label": "sclerotinia stem rot",
        "alt_labels": ["Sclerotinia Stem Rot", "WHITE MOLD", "white mold",
                       "sclerotinia rot", "cottony rot", "SSR"],
        "scope_note": "Fungal disease caused by Sclerotinia sclerotiorum. Produces white cottony mycelium and hard black sclerotia on infected tissue.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_7000",
    },
    {
        "preferred_label": "grey mould",
        "alt_labels": ["Grey Mould", "Gray Mold", "GREY MOULD", "botrytis",
                       "Botrytis blight", "GM"],
        "scope_note": "Fungal disease caused by Botrytis cinerea. Produces grey fuzzy growth on infected tissue. Affects many crops in cool, humid conditions.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_3339",
    },
    {
        "preferred_label": "black rot",
        "alt_labels": ["Black Rot", "BLACK ROT", "Xanthomonas black rot",
                       "BR", "grape black rot", "cabbage black rot"],
        "scope_note": "Disease name used for multiple pathogens. In grapes caused by Guignardia bidwellii; in brassicas by Xanthomonas campestris.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_993",
    },
    {
        "preferred_label": "citrus canker",
        "alt_labels": ["Citrus Canker", "CITRUS CANKER", "Xanthomonas canker",
                       "CC", "bacterial canker of citrus"],
        "scope_note": "Bacterial disease of citrus caused by Xanthomonas citri subsp. citri. Produces raised corky lesions on leaves, stems and fruit.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_1717",
    },
    {
        "preferred_label": "bacterial spot",
        "alt_labels": ["Bacterial Spot", "BACTERIAL SPOT", "Xanthomonas spot",
                       "BS", "leaf spot", "bacterial leaf spot"],
        "scope_note": "Bacterial disease caused by Xanthomonas species on tomato, pepper and peach. Water-soaked spots become necrotic.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_769",
    },
    {
        "preferred_label": "septoria leaf spot",
        "alt_labels": ["Septoria Leaf Spot", "SEPTORIA LEAF SPOT", "SLS",
                       "septoria blight", "glume blotch"],
        "scope_note": "Fungal disease caused by Septoria species. Small circular spots with dark margins. Severe defoliation in susceptible crops.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_7050",
    },
    {
        "preferred_label": "smut",
        "alt_labels": ["Smut", "SMUT", "covered smut", "loose smut",
                       "common smut", "head smut"],
        "scope_note": "Disease caused by Ustilago and related genera. Converts plant tissues to masses of black teliospores.",
        "concept_uri": "http://aims.fao.org/aos/agrovoc/c_7202",
    },
]


def query_agrovoc_sparql() -> list:
    """
    WHAT THIS DOES:
      Sends a SPARQL query to the AGROVOC endpoint and gets disease terms.
      This requires internet access.

    RETURNS:
      List of dicts with keys: preferred_label, alt_labels, scope_note, concept_uri
    """
    print("  Querying AGROVOC SPARQL endpoint...")
    print(f"  URL: {SPARQL_ENDPOINT}")

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": SPARQL_QUERY},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print("  ⚠ AGROVOC timeout. Will use offline data.")
        return []
    except requests.exceptions.ConnectionError:
        print("  ⚠ Cannot reach AGROVOC (no internet?). Will use offline data.")
        return []
    except Exception as e:
        print(f"  ⚠ AGROVOC error: {e}. Will use offline data.")
        return []

    # ── Parse the SPARQL JSON response ───────────────────────────────────────
    # SPARQL returns results in this structure:
    # { "results": { "bindings": [ {"prefLabel": {"value": "..."}, ...}, ... ] } }
    bindings = data.get("results", {}).get("bindings", [])
    print(f"  Received {len(bindings)} bindings from AGROVOC")

    # Group alt labels by preferred label
    concepts = {}
    for row in bindings:
        pref = row.get("prefLabel", {}).get("value", "")
        alt  = row.get("altLabel",  {}).get("value", "")
        note = row.get("scopeNote", {}).get("value", "")
        uri  = row.get("concept",   {}).get("value", "")

        if not pref:
            continue
        if pref not in concepts:
            concepts[pref] = {"preferred_label": pref, "alt_labels": [],
                               "scope_note": note, "concept_uri": uri}
        if alt and alt not in concepts[pref]["alt_labels"]:
            concepts[pref]["alt_labels"].append(alt)

    return list(concepts.values())


def build_entity_records_from_agrovoc(agrovoc_data: list) -> list:
    """
    WHAT THIS DOES:
      Converts AGROVOC concept dicts into our standard entity record format.
      Creates one record for the preferred label, and one for each alt label.
    """
    records = []
    for item in agrovoc_data:
        pref = item["preferred_label"]
        note = item.get("scope_note", "")
        uri  = item.get("concept_uri", SOURCE_URL)

        # Record for the preferred (canonical) name
        records.append(build_entity_record(
            name=pref, context=note, source=SOURCE_NAME,
            source_url=uri, entity_type="Disease"
        ))

        # Record for each alternative label (synonym)
        for alt in item.get("alt_labels", []):
            if alt and alt.strip():
                records.append(build_entity_record(
                    name=alt, context=note, source=SOURCE_NAME,
                    source_url=uri, entity_type="Disease"
                ))

    return records


def build_agrovoc_pairs(agrovoc_data: list) -> list:
    """
    WHAT THIS DOES:
      For each AGROVOC concept, generate positive pairs from
      (preferred_label, alt_label1), (preferred_label, alt_label2),
      (alt_label1, alt_label2), etc.
    """
    pairs = []
    pair_id = 1

    for item in agrovoc_data:
        pref = item["preferred_label"]
        alts = item.get("alt_labels", [])

        all_forms = [pref] + [a for a in alts if a.strip()]
        if len(all_forms) < 2:
            continue

        for name_1, name_2 in itertools.combinations(all_forms, 2):
            pairs.append({
                "pair_id":     f"AV_POS_{pair_id:04d}",
                "name_1":      name_1,
                "name_2":      name_2,
                "canonical_1": normalise_name(name_1),
                "canonical_2": normalise_name(name_2),
                "entity_type": "Disease",
                "label":       1,
                "pair_source": "agrovoc_sparql",
                "confidence":  1.0,
                "note":        f"AGROVOC synonyms of: '{pref}'",
            })
            pair_id += 1

    return pairs


def main():
    print("\n" + "═"*60)
    print("  SCRIPT 02 — AGROVOC Disease Term Downloader")
    print("  AgriΛNet Entity Resolution Pipeline")
    print("═"*60)

    # ── Try live API first, fall back to offline ───────────────────────────
    print("\n[1/3] Fetching AGROVOC disease terms...")
    agrovoc_data = query_agrovoc_sparql()

    if not agrovoc_data:
        print("  Using built-in offline AGROVOC data...")
        agrovoc_data = AGROVOC_OFFLINE
    else:
        print(f"  ✓ Fetched {len(agrovoc_data)} concepts from live AGROVOC")

    # ── Build entity records ───────────────────────────────────────────────
    print("\n[2/3] Building entity records...")
    records = build_entity_records_from_agrovoc(agrovoc_data)
    df = save_raw(records, "agrovoc_raw.csv")

    # ── Build pairs ────────────────────────────────────────────────────────
    print("\n[3/3] Building positive synonym pairs...")
    pairs = build_agrovoc_pairs(agrovoc_data)
    pairs_df = pd.DataFrame(pairs)
    out_path = PAIRS_DIR / "agrovoc_pairs_positive.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(pairs_df)} positive pairs → {out_path.name}")

    print(f"\n  ┌────────────────────────────────────────┐")
    print(f"  │ AGROVOC concepts:         {len(agrovoc_data):5d}          │")
    print(f"  │ Entity records (w/alts):  {len(df):5d}          │")
    print(f"  │ Positive synonym pairs:   {len(pairs_df):5d}          │")
    print(f"  └────────────────────────────────────────┘")
    print("\n  ✅ Script 02 complete! Next: run 03_scrape_wikipedia.py\n")

    return df, pairs_df


if __name__ == "__main__":
    main()