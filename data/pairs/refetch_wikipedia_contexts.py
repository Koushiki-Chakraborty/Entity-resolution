"""
refetch_wikipedia_contexts.py
===============================
Fetches complete, sentence-boundary-accurate Wikipedia contexts
for all unique entities in training_ready_final.csv.

Strategy:
  1. Try exact Wikipedia page title match (REST API summary)
  2. Fall back to Wikipedia search for the best matching page
  3. Extract 2-3 complete sentences, targeting 200-400 chars
  4. Save a cache (wiki_cache.json) so runs can be resumed
  5. Produce training_ready_richcontext.csv with updated contexts

Rate limit: 1 request/sec (polite Wikipedia usage)
"""

import pandas as pd
import requests
import json
import re
import time
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV   = "training_ready_final.csv"
OUTPUT_CSV  = "training_ready_richcontext.csv"
CACHE_FILE  = "wiki_cache.json"

MIN_CHARS   = 200   # minimum acceptable context length
MAX_CHARS   = 450   # soft cap (won't cut a sentence to fit)
SLEEP_SEC   = 0.8   # seconds between Wikipedia API calls

HEADERS = {"User-Agent": "AgriLambdaNet-research/1.0 (academic; contact: research@example.com)"}

# Skip entities that are clearly not disease names (abbreviations, organisms, etc.)
# The scraper will still try them but we track results
ABBREV_SKIP = {"eb", "blb", "lb", "rb", "yr", "asr", "sbr", "gbr", "blsd",
               "nclb", "nlb", "pst", "pbs", "tbs", "sls", "gls", "gpm",
               "fhb", "foc", "clr", "ccr", "abr", "car", "glb", "plb",
               "cpm", "wpm", "spm", "sds", "tssm", "rbs", "tomv", "tmv",
               "tylcv", "ylcv", "tr4", "peb", "sbs", "hlb"}

# ── WIKIPEDIA API HELPERS ──────────────────────────────────────────────────────

def clean_extract(text: str) -> str:
    """Remove wiki markup leftovers and clean whitespace."""
    text = re.sub(r'\([^)]*\)', '', text)       # remove parenthetical refs
    text = re.sub(r'\[[^\]]*\]', '', text)       # remove [note] brackets
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_sentences(text: str, target_min=MIN_CHARS, target_max=MAX_CHARS) -> str:
    """
    Pull 2-3 complete sentences from text, targeting MIN-MAX chars.
    Always ends at a sentence boundary.
    """
    # Split on sentence-ending punctuation followed by space/end
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return ""

    result = ""
    for sent in sentences:
        candidate = (result + " " + sent).strip() if result else sent
        if len(candidate) >= target_min:
            # We have enough — stop here even if one sentence
            return candidate if len(candidate) <= target_max + 100 else result or candidate
        result = candidate

    return result   # return everything we got if still under min


def fetch_summary_api(title: str) -> str | None:
    """
    Wikipedia REST summary API — fastest, returns clean extract of first paragraph.
    Returns the extract string or None on failure.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("type") == "disambiguation":
                return None
            extract = data.get("extract", "")
            return extract if len(extract) > 30 else None
    except Exception:
        pass
    return None


def search_wikipedia(query: str) -> str | None:
    """
    Wikipedia search API — finds best matching article for a query.
    Returns the extract of the top result or None.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query + " disease plant",
        "srlimit": 3,
        "format": "json",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return None
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return None
        top_title = results[0]["title"]
        # Now fetch the actual summary for this title
        return fetch_summary_api(top_title)
    except Exception:
        pass
    return None


def get_wikipedia_context(name: str) -> tuple[str, str]:
    """
    Main entry point: given an entity name, return (context, status).
    status: 'exact' | 'search' | 'not_found'
    Tries:
      1. Exact title
      2. Title-cased exact
      3. Wikipedia search
    """
    candidates = [
        name,
        name.title(),
        name.capitalize(),
    ]

    for candidate in candidates:
        raw = fetch_summary_api(candidate)
        time.sleep(SLEEP_SEC)
        if raw and len(raw) > 80:
            cleaned = clean_extract(raw)
            ctx = extract_sentences(cleaned)
            if len(ctx) >= 80:
                return ctx, "exact"

    # Fall back to search
    raw = search_wikipedia(name)
    time.sleep(SLEEP_SEC)
    if raw and len(raw) > 80:
        cleaned = clean_extract(raw)
        ctx = extract_sentences(cleaned)
        if len(ctx) >= 80:
            return ctx, "search"

    return "", "not_found"


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# Collect all unique entity names with their existing context
entity_map: dict[str, str] = {}  # name.lower() -> best existing context
for _, row in df.iterrows():
    for col_n, col_c in [("name_a", "context_a"), ("name_b", "context_b")]:
        key = str(row[col_n]).strip().lower()
        ctx = str(row[col_c]).strip()
        # Keep longest context seen so far
        if key not in entity_map or len(ctx) > len(entity_map[key]):
            entity_map[key] = ctx

# Decide which need re-fetching
def needs_refetch(ctx: str) -> bool:
    ctx = ctx.strip()
    if len(ctx) < MIN_CHARS:
        return True
    if ctx and ctx[-1] not in ".!?)\"'":
        return True
    return False

to_fetch = {name: ctx for name, ctx in entity_map.items() if needs_refetch(ctx)}
already_good = {name: ctx for name, ctx in entity_map.items() if not needs_refetch(ctx)}

print(f"\nEntities with rich complete context already : {len(already_good)}")
print(f"Entities needing Wikipedia re-fetch         : {len(to_fetch)}")

# ── LOAD CACHE ────────────────────────────────────────────────────────────────
cache: dict[str, dict] = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache loaded: {len(cache)} entities already fetched")

# ── FETCH LOOP ────────────────────────────────────────────────────────────────
remaining = {n: c for n, c in to_fetch.items() if n not in cache}
print(f"Still need to fetch                         : {len(remaining)}\n")

stats = {"exact": 0, "search": 0, "not_found": 0, "skipped_abbrev": 0}
total = len(remaining)

for i, (name, old_ctx) in enumerate(sorted(remaining.items()), 1):
    prefix = f"[{i:3d}/{total}] {name:<45}"

    if name in ABBREV_SKIP:
        print(f"{prefix} SKIPPED (abbreviation)")
        cache[name] = {"context": old_ctx, "status": "skipped_abbrev"}
        stats["skipped_abbrev"] += 1
        continue

    ctx, status = get_wikipedia_context(name)

    if ctx and len(ctx) >= MIN_CHARS:
        cache[name] = {"context": ctx, "status": status}
        stats[status] += 1
        improvement = len(ctx) - len(old_ctx)
        print(f"{prefix} OK [{status}] {len(ctx)} chars (+{improvement})")
    else:
        # Keep old context if Wikipedia found nothing better
        cache[name] = {"context": old_ctx, "status": "not_found"}
        stats["not_found"] += 1
        print(f"{prefix} NOT FOUND (kept old: {len(old_ctx)} chars)")

    # Save cache every 20 items so progress is never lost
    if i % 20 == 0:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"  -- Cache saved at {i} items --")

# Final cache save
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

# ── BUILD UPDATED DATAFRAME ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BUILDING UPDATED DATASET")
print(f"{'='*60}")

# Merge: cache + already_good entities -> final context lookup
final_ctx: dict[str, str] = {}
for name, ctx in already_good.items():
    final_ctx[name] = ctx
for name, data in cache.items():
    ctx = data["context"]
    if len(ctx) >= 60:  # only overwrite if we got something meaningful
        final_ctx[name] = ctx

# Apply to dataframe
df_out = df.copy()
replaced_a = replaced_b = 0

for idx, row in df_out.iterrows():
    key_a = str(row["name_a"]).strip().lower()
    key_b = str(row["name_b"]).strip().lower()

    if key_a in final_ctx:
        new_ctx = final_ctx[key_a]
        if new_ctx != str(row["context_a"]).strip():
            df_out.at[idx, "context_a"] = new_ctx
            replaced_a += 1

    if key_b in final_ctx:
        new_ctx = final_ctx[key_b]
        if new_ctx != str(row["context_b"]).strip():
            df_out.at[idx, "context_b"] = new_ctx
            replaced_b += 1

# ── FINAL STATS ───────────────────────────────────────────────────────────────
print(f"\nFetch results:")
print(f"  Exact title match   : {stats['exact']}")
print(f"  Found via search    : {stats['search']}")
print(f"  Not found           : {stats['not_found']}")
print(f"  Skipped (abbrevs)   : {stats['skipped_abbrev']}")
print(f"\nRows updated:")
print(f"  context_a updated   : {replaced_a}")
print(f"  context_b updated   : {replaced_b}")

# Context length improvement
for col in ["context_a", "context_b"]:
    old_med = df[col].str.len().median()
    new_med = df_out[col].str.len().median()
    complete_old = df[col].str.strip().apply(lambda t: len(t) > 0 and t[-1] in ".!?)").sum()
    complete_new = df_out[col].str.strip().apply(lambda t: len(t) > 0 and t[-1] in ".!?)").sum()
    print(f"\n  [{col}]")
    print(f"    Median length  : {int(old_med)} -> {int(new_med)} chars")
    print(f"    Complete sent  : {complete_old} -> {complete_new} rows ({complete_new/len(df_out)*100:.1f}%)")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV} ({len(df_out)} rows)")
print("Next step: re-run prepare_dataset.py on this file.")
