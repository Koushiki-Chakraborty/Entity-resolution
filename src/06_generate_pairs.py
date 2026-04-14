"""
06_generate_pairs.py  —  Step 2: LLM Knowledge Distillation
=============================================================================

PURPOSE
-------
Uses OpenAI as a one-time "teacher" LLM to generate dual supervision labels
for every disease name pair:

  llm_match  (bool)  — do the two entries refer to the SAME crop disease?
  llm_lambda (float) — how much did the NAME drive the decision vs CONTEXT?
                       0.0 = context only  |  0.5 = balanced  |  1.0 = name only

COST MINIMIZATION STRATEGY
---------------------------
Three layers of cost control are used:

  1. OpenAI Batch API  — 50% cheaper than real-time, processes overnight
  2. Minimal prompt    — ~200 tokens/call vs ~500 in naive approaches
  3. Short context     — truncated to MAX_CONTEXT_LENGTH chars (default 120)

At these settings, 1,881 pairs cost approx. $0.04–0.06 total.

TWO MODES
---------
  --mode batch   (default, RECOMMENDED)
      Submits all pairs to OpenAI Batch API in one shot. Costs 50% less.
      You run the script once to SUBMIT, then run again to RETRIEVE when done.
      Batches complete within 24h (usually <1h for small jobs).

  --mode live
      Calls the API one-by-one in real time. Costs 2x more but instant.
      Use if you need results immediately. Has rate-limit protection.

INPUT
-----
  data/processed/crop_diseases_clean.csv
    Columns: name, canonical_id, context, source_url

OUTPUT
------
  data/pairs/llm_labeled_pairs.csv
    Columns: name_a, context_a, name_b, context_b,
             canonical_id_a, canonical_id_b, source_url_a, source_url_b,
             true_label, llm_match, llm_lambda

USAGE
-----
  # Step A — submit batch job (recommended)
  python 06_generate_pairs.py --mode batch

  # Step B — retrieve results (run after batch completes, ~1h)
  python 06_generate_pairs.py --mode retrieve --batch-id batch_abc123

  # Alternative — real-time, all pairs
  python 06_generate_pairs.py --mode live

  # Test with just 10 pairs (free sanity check)
  python 06_generate_pairs.py --mode live --dry-run 10
"""

import os
import sys
import json
import random
import time
import itertools
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "processed" / "crop_diseases_clean.csv"
PAIRS_DIR  = ROOT_DIR / "data" / "pairs"
BATCH_DIR  = ROOT_DIR / "data" / "pairs" / "batch_files"
PAIRS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

API_KEY          = os.getenv("OPENAI_API_KEY", "")
MODEL            = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OUTPUT_FILE      = PAIRS_DIR / os.getenv("OUTPUT_FILE", "llm_labeled_pairs.csv")
RANDOM_SEED      = int(os.getenv("RANDOM_SEED", 42))
NEGATIVE_RATIO   = int(os.getenv("NEGATIVE_RATIO", 2))
SLEEP_TIME       = float(os.getenv("SLEEP_BETWEEN_CALLS", 1.0))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", 50))
MAX_CTX_LEN      = int(os.getenv("MAX_CONTEXT_LENGTH", 120))

if not API_KEY or "your_" in API_KEY.lower():
    print("[ERROR] OPENAI_API_KEY is not set in .env")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)


# ==============================================================================
# PROMPT — kept as short as possible to minimize tokens
# ==============================================================================
SYSTEM_PROMPT = (
    "You are a plant pathology expert. "
    "Respond ONLY with a JSON object, no explanation."
)

def build_user_prompt(name_a: str, ctx_a: str, name_b: str, ctx_b: str) -> str:
    """
    Minimal prompt — ~130 tokens input per call.
    Two tasks in one call saves 50% vs calling separately.
    """
    a_ctx = ctx_a[:MAX_CTX_LEN].strip()
    b_ctx = ctx_b[:MAX_CTX_LEN].strip()
    return (
        f'A: "{name_a}" | {a_ctx}\n'
        f'B: "{name_b}" | {b_ctx}\n\n'
        "1. match: true if A and B are the same crop disease (aliases/synonyms count).\n"
        "2. lambda: 0.0=context drove decision, 1.0=name alone was enough.\n"
        'JSON only: {"match": true, "lambda": 0.7}'
    )


def parse_llm_response(text: str) -> dict | None:
    """Extract and validate JSON from LLM response."""
    text = text.strip()
    # Strip markdown fences
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
            break
    # Find JSON object
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        result = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    if "match" not in result or "lambda" not in result:
        return None

    # Normalize types
    if not isinstance(result["match"], bool):
        result["match"] = str(result["match"]).lower() in ("true", "1", "yes")
    result["lambda"] = round(max(0.0, min(1.0, float(result["lambda"]))), 3)
    return result


# ==============================================================================
# PAIR GENERATION
# ==============================================================================
def load_and_generate_pairs(df: pd.DataFrame) -> list[dict]:
    """Generate all positive and negative pairs from the dataset."""
    groups    = df.groupby("canonical_id")
    seen: set[tuple] = set()

    # ── Positive pairs ─────────────────────────────────────────────────────────
    positive_pairs = []
    for cid, group in groups:
        rows = group.reset_index(drop=True)
        if len(rows) < 2:
            continue
        for i, j in itertools.combinations(range(len(rows)), 2):
            key = tuple(sorted([rows.loc[i, "name"], rows.loc[j, "name"]]))
            if key in seen:
                continue
            seen.add(key)
            positive_pairs.append({
                "name_a":         rows.loc[i, "name"],
                "context_a":      rows.loc[i, "context"],
                "source_url_a":   rows.loc[i].get("source_url", ""),
                "name_b":         rows.loc[j, "name"],
                "context_b":      rows.loc[j, "context"],
                "source_url_b":   rows.loc[j].get("source_url", ""),
                "canonical_id_a": cid,
                "canonical_id_b": cid,
                "true_label":     1,
            })

    # ── Negative pairs ─────────────────────────────────────────────────────────
    target_neg = len(positive_pairs) * NEGATIVE_RATIO
    random.seed(RANDOM_SEED)
    all_rows = df.reset_index(drop=True)
    negative_pairs = []
    attempts = 0
    max_attempts = target_neg * 30

    while len(negative_pairs) < target_neg and attempts < max_attempts:
        attempts += 1
        i = random.randint(0, len(all_rows) - 1)
        j = random.randint(0, len(all_rows) - 1)
        if i == j:
            continue
        if all_rows.loc[i, "canonical_id"] == all_rows.loc[j, "canonical_id"]:
            continue
        key = tuple(sorted([all_rows.loc[i, "name"], all_rows.loc[j, "name"]]))
        if key in seen:
            continue
        seen.add(key)
        negative_pairs.append({
            "name_a":         all_rows.loc[i, "name"],
            "context_a":      all_rows.loc[i, "context"],
            "source_url_a":   all_rows.loc[i].get("source_url", ""),
            "name_b":         all_rows.loc[j, "name"],
            "context_b":      all_rows.loc[j, "context"],
            "source_url_b":   all_rows.loc[j].get("source_url", ""),
            "canonical_id_a": all_rows.loc[i, "canonical_id"],
            "canonical_id_b": all_rows.loc[j, "canonical_id"],
            "true_label":     0,
        })

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


# ==============================================================================
# MODE 1 — BATCH API (50% cheaper, recommended)
# ==============================================================================
def run_batch_submit(pairs: list[dict]) -> str:
    """
    Submits all pairs to OpenAI Batch API.
    Returns the batch_id — save this to retrieve results later.
    """
    print(f"\n[BATCH] Building JSONL file for {len(pairs)} pairs...")

    jsonl_path = BATCH_DIR / "batch_input.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, pair in enumerate(pairs):
            request = {
                "custom_id": f"pair-{idx:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(
                            pair["name_a"], pair["context_a"],
                            pair["name_b"], pair["context_b"]
                        )},
                    ],
                    "max_tokens": 30,        # JSON reply is tiny — hard cap saves output tokens
                    "temperature": 0,        # Deterministic — no creativity needed
                    "response_format": {"type": "json_object"},  # Force JSON mode
                },
            }
            f.write(json.dumps(request) + "\n")

    # Save pairs metadata so we can match results back later
    pairs_meta_path = BATCH_DIR / "pairs_metadata.json"
    with open(pairs_meta_path, "w") as f:
        json.dump(pairs, f)
    print(f"  Metadata saved: {pairs_meta_path.name}")

    # Upload the JSONL file
    print(f"[BATCH] Uploading {jsonl_path.name}...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {uploaded.id}")

    # Create the batch job
    print("[BATCH] Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    batch_id_path = BATCH_DIR / "batch_id.txt"
    batch_id_path.write_text(batch.id)

    print(f"\n{'='*62}")
    print(f"  Batch submitted successfully!")
    print(f"  Batch ID : {batch.id}")
    print(f"  Status   : {batch.status}")
    print(f"  Saved to : {batch_id_path}")
    print(f"{'='*62}")
    print(f"\n  Come back in ~1 hour and run:")
    print(f"  python 06_generate_pairs.py --mode retrieve --batch-id {batch.id}")
    print(f"\n  Or check status at: https://platform.openai.com/batches")
    return batch.id


def run_batch_retrieve(batch_id: str) -> pd.DataFrame:
    """
    Retrieves completed batch results, merges with metadata, saves CSV.
    """
    print(f"\n[RETRIEVE] Checking batch {batch_id}...")
    batch = client.batches.retrieve(batch_id)
    print(f"  Status           : {batch.status}")
    print(f"  Request counts   : {batch.request_counts}")

    if batch.status != "completed":
        remaining_map = {
            "validating": "a few minutes",
            "in_progress": "up to 1 hour",
            "finalizing": "a few minutes",
            "failed": "FAILED — check OpenAI dashboard",
            "expired": "EXPIRED — resubmit",
        }
        eta = remaining_map.get(batch.status, "unknown")
        print(f"\n  Not ready yet. Expected wait: {eta}")
        print(f"  Run again later: python 06_generate_pairs.py --mode retrieve --batch-id {batch_id}")
        return pd.DataFrame()

    # Load metadata
    pairs_meta_path = BATCH_DIR / "pairs_metadata.json"
    if not pairs_meta_path.exists():
        print("[ERROR] pairs_metadata.json not found — was batch submitted from this machine?")
        sys.exit(1)
    with open(pairs_meta_path) as f:
        pairs = json.load(f)

    # Download results
    print(f"[RETRIEVE] Downloading results (file_id={batch.output_file_id})...")
    content = client.files.content(batch.output_file_id)
    lines   = content.text.strip().split("\n")
    print(f"  Result lines: {len(lines)}")

    # Parse results into a dict keyed by custom_id
    results_map: dict[str, dict] = {}
    failed = 0
    for line in lines:
        try:
            obj = json.loads(line)
            cid = obj["custom_id"]
            body = obj["response"]["body"]
            text = body["choices"][0]["message"]["content"]
            parsed = parse_llm_response(text)
            if parsed:
                results_map[cid] = parsed
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  Parsed OK  : {len(results_map)}")
    print(f"  Failed     : {failed}")

    # Merge with pair metadata
    final_rows = []
    for idx, pair in enumerate(pairs):
        cid = f"pair-{idx:05d}"
        llm = results_map.get(cid)
        row = {
            "name_a":         pair["name_a"],
            "context_a":      pair["context_a"][:MAX_CTX_LEN],
            "name_b":         pair["name_b"],
            "context_b":      pair["context_b"][:MAX_CTX_LEN],
            "canonical_id_a": pair["canonical_id_a"],
            "canonical_id_b": pair["canonical_id_b"],
            "source_url_a":   pair["source_url_a"],
            "source_url_b":   pair["source_url_b"],
            "true_label":     pair["true_label"],
            "llm_match":      llm["match"]  if llm else None,
            "llm_lambda":     llm["lambda"] if llm else None,
        }
        final_rows.append(row)

    df_out = pd.DataFrame(final_rows)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved: {OUTPUT_FILE}  ({len(df_out)} rows)")
    return df_out


# ==============================================================================
# MODE 2 — LIVE (real-time, 2x more expensive, but instant)
# ==============================================================================
def call_openai_live(name_a: str, ctx_a: str, name_b: str, ctx_b: str) -> dict | None:
    """Single synchronous API call with retry on rate limit."""
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(name_a, ctx_a, name_b, ctx_b)},
                ],
                max_tokens=30,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parse_llm_response(response.choices[0].message.content)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = [10, 30, 60][min(attempt, 2)]
                print(f"  [429] Rate limit — waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  [ERR] {err[:80]}")
                return None
    return None


def run_live(pairs: list[dict], dry_run: int = 0) -> pd.DataFrame:
    """Real-time labeling with checkpoint saves and resume support."""

    if dry_run > 0:
        pairs = pairs[:dry_run]
        print(f"[DRY RUN] Processing only {dry_run} pairs for testing.")

    # Resume support
    completed = []
    start_idx = 0
    if OUTPUT_FILE.exists() and not dry_run:
        existing = pd.read_csv(OUTPUT_FILE)
        completed = existing.to_dict("records")
        start_idx = len(completed)
        print(f"  Resuming from pair {start_idx + 1} ({start_idx} already done)")

    pairs_todo = pairs[start_idx:]
    failed = 0

    print(f"  Pairs to label : {len(pairs_todo)}")
    print(f"  ETA            : ~{len(pairs_todo) * SLEEP_TIME / 60:.0f} min")
    print(f"  Est. cost      : ${len(pairs_todo) * 150 * 0.15 / 1_000_000:.4f}")
    print()

    try:
        for idx, pair in enumerate(pairs_todo):
            global_n = start_idx + idx + 1

            if idx % 10 == 0:
                pct = global_n / len(pairs) * 100
                eta = (len(pairs_todo) - idx) * SLEEP_TIME / 60
                print(f"  [{global_n:4d}/{len(pairs)}] {pct:4.0f}%  "
                      f"done:{len(completed)}  fail:{failed}  ETA:{eta:.0f}min")

            llm = call_openai_live(
                pair["name_a"], pair["context_a"],
                pair["name_b"], pair["context_b"]
            )

            if llm:
                completed.append({
                    "name_a":         pair["name_a"],
                    "context_a":      pair["context_a"][:MAX_CTX_LEN],
                    "name_b":         pair["name_b"],
                    "context_b":      pair["context_b"][:MAX_CTX_LEN],
                    "canonical_id_a": pair["canonical_id_a"],
                    "canonical_id_b": pair["canonical_id_b"],
                    "source_url_a":   pair["source_url_a"],
                    "source_url_b":   pair["source_url_b"],
                    "true_label":     pair["true_label"],
                    "llm_match":      llm["match"],
                    "llm_lambda":     llm["lambda"],
                })
            else:
                failed += 1

            # Checkpoint
            if (idx + 1) % CHECKPOINT_EVERY == 0 and not dry_run:
                pd.DataFrame(completed).to_csv(OUTPUT_FILE, index=False)
                print(f"  [CKPT] {len(completed)} pairs saved")

            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted — saving progress...")

    df_out = pd.DataFrame(completed)
    if not dry_run:
        df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    return df_out


# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Total labeled          : {len(df)}")

    pos_df = df[df["true_label"] == 1]
    neg_df = df[df["true_label"] == 0]
    print(f"  Positive (label=1)     : {len(pos_df)}")
    print(f"  Negative (label=0)     : {len(neg_df)}")

    if "llm_match" in df.columns and df["llm_match"].notna().any():
        llm_pos = df["llm_match"] == True
        agree = (
            ((df["true_label"] == 1) & llm_pos) |
            ((df["true_label"] == 0) & ~llm_pos)
        ).sum()
        total = df["llm_match"].notna().sum()
        print(f"  LLM agreement          : {agree}/{total}  ({agree/total*100:.1f}%)")

    if "llm_lambda" in df.columns and df["llm_lambda"].notna().any():
        lam = df["llm_lambda"].dropna()
        print(f"  Lambda mean            : {lam.mean():.3f}")
        print(f"    0.0–0.3 context-driven : {(lam <= 0.3).sum()}")
        print(f"    0.3–0.7 balanced       : {((lam > 0.3) & (lam < 0.7)).sum()}")
        print(f"    0.7–1.0 name-driven    : {(lam >= 0.7).sum()}")

    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 62)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="06_generate_pairs — LLM labeling")
    parser.add_argument("--mode",     choices=["batch", "retrieve", "live"], default="batch",
                        help="batch=submit (cheap), retrieve=get results, live=real-time")
    parser.add_argument("--batch-id", type=str, default="",
                        help="Batch ID returned from --mode batch (required for retrieve)")
    parser.add_argument("--dry-run",  type=int, default=0,
                        help="Only process N pairs (for testing, no output saved)")
    args = parser.parse_args()

    print("=" * 62)
    print("  06_generate_pairs.py  —  OpenAI Edition")
    print("=" * 62)
    print(f"  Mode    : {args.mode}")
    print(f"  Model   : {MODEL}")
    print(f"  Output  : {OUTPUT_FILE.name}")
    print("=" * 62)

    # ── Load dataset ────────────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        print(f"[ERROR] Not found: {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["name", "context", "canonical_id"]).reset_index(drop=True)
    df["context"] = df["context"].astype(str).str.strip()
    df["name"]    = df["name"].astype(str).str.strip()

    groups = df.groupby("canonical_id")
    multi  = sum(1 for _, g in groups if len(g) >= 2)

    print(f"\n  Dataset rows           : {len(df)}")
    print(f"  Canonical groups       : {df['canonical_id'].nunique()}")
    print(f"  Multi-alias groups     : {multi}")

    # ── Generate pairs (needed for all modes except retrieve) ──────────────────
    if args.mode != "retrieve":
        print("\n  Generating pairs...")
        pairs = load_and_generate_pairs(df)
        n_pos = sum(1 for p in pairs if p["true_label"] == 1)
        n_neg = sum(1 for p in pairs if p["true_label"] == 0)
        print(f"  Positive pairs         : {n_pos}")
        print(f"  Negative pairs         : {n_neg}")
        print(f"  Total                  : {len(pairs)}")

        # Cost estimate (batch price = 50% off)
        est_input_tok = len(pairs) * 150  # ~150 tokens per call with short prompt+context
        est_out_tok   = len(pairs) * 15
        live_cost  = (est_input_tok * 0.15 + est_out_tok * 0.60) / 1_000_000
        batch_cost = live_cost * 0.5
        print(f"\n  Estimated cost (live)  : ${live_cost:.4f}")
        print(f"  Estimated cost (batch) : ${batch_cost:.4f}  ← recommended")

    # ── Dispatch ────────────────────────────────────────────────────────────────
    if args.mode == "batch":
        run_batch_submit(pairs)

    elif args.mode == "retrieve":
        bid = args.batch_id
        if not bid:
            # Try reading saved batch_id
            saved = BATCH_DIR / "batch_id.txt"
            if saved.exists():
                bid = saved.read_text().strip()
                print(f"  Using saved batch ID: {bid}")
            else:
                print("[ERROR] --batch-id is required for retrieve mode")
                sys.exit(1)
        df_out = run_batch_retrieve(bid)
        print_summary(df_out)

    elif args.mode == "live":
        df_out = run_live(pairs, dry_run=args.dry_run)
        print_summary(df_out)


if __name__ == "__main__":
    main()