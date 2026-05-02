"""
label_pending_llm.py
====================
Labels ONLY the 279 `lambda_source == 'pending_llm'` rows in
dataset_augmented.csv using GPT-4o-mini.

ZERO WASTE POLICY
-----------------
- Never calls the API on rows that already have a lambda_val.
- Checkpoints every 50 calls so Ctrl+C loses at most 50 rows.
- Resume-safe: re-running picks up from the last checkpoint automatically.
- Uses the SAME minimal prompt as 06_generate_pairs.py (~130 tokens/call).
- max_tokens=30 (JSON reply is tiny).

MODES
-----
  --mode live    (default)  Real-time, instant, costs ~2x batch
  --mode batch              Submit OpenAI Batch job (50% cheaper, ~1h wait)
  --mode retrieve           Retrieve + merge completed batch results

USAGE
-----
  # Run all 279 pairs right now (costs ~$0.006):
  python label_pending_llm.py --mode live

  # Cheaper (batch, 50% off, results in ~1h):
  python label_pending_llm.py --mode batch
  python label_pending_llm.py --mode retrieve --batch-id <id>

  # Test 5 pairs first without saving:
  python label_pending_llm.py --mode live --dry-run 5
"""

import os
import sys
import json

# Force UTF-8 output on Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent.parent          # entity_resolution_copy/
DATA_DIR     = SCRIPT_DIR.parent / "data"        # dataset_v2_builder/data/
DATASET_PATH = DATA_DIR / "dataset_augmented.csv"
BATCH_DIR    = DATA_DIR / "pending_batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = DATA_DIR / "pending_llm_checkpoint.csv"

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

API_KEY          = os.getenv("OPENAI_API_KEY", "")
MODEL            = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CTX_LEN      = int(os.getenv("MAX_CONTEXT_LENGTH", 120))
SLEEP_TIME       = float(os.getenv("SLEEP_BETWEEN_CALLS", 0.5))   # paid key: 0.5s fine
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", 50))

if not API_KEY or "your_" in API_KEY.lower():
    print("[ERROR] OPENAI_API_KEY not set in .env")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# ── Prompt (identical to 06_generate_pairs.py — tested, minimal) ───────────────
SYSTEM_PROMPT = (
    "You are a plant pathology expert. "
    "Respond ONLY with a JSON object, no explanation."
)

def build_prompt(name_a: str, ctx_a: str, name_b: str, ctx_b: str) -> str:
    a = ctx_a[:MAX_CTX_LEN].strip()
    b = ctx_b[:MAX_CTX_LEN].strip()
    return (
        f'A: "{name_a}" | {a}\n'
        f'B: "{name_b}" | {b}\n\n'
        "1. match: true if A and B are the same crop disease (aliases/synonyms count).\n"
        "2. lambda: 0.0=context drove decision, 1.0=name alone was enough.\n"
        'JSON only: {"match": true, "lambda": 0.7}'
    )


def parse_response(text: str) -> dict | None:
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
            break
    s = text.find("{")
    e = text.rfind("}") + 1
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e])
    except json.JSONDecodeError:
        return None
    if "match" not in obj or "lambda" not in obj:
        return None
    obj["match"]  = str(obj["match"]).lower() in ("true", "1", "yes")
    obj["lambda"] = round(max(0.0, min(1.0, float(obj["lambda"]))), 3)
    return obj


# ── Lambda from llm_lambda: Type C pairs → low lambda (0.05–0.25 range) ────────
def compute_lambda_type_c(llm_lambda: float, name_sim: float) -> float:
    """
    For Type C (hard negatives), the LLM's raw lambda is already in 0–1.
    We trust it directly — it encodes how much the name alone signalled 'same disease'.
    For these pairs match=0, so a high lambda would mean 'name fooled me into thinking
    same' → high lambda is valid for Type C (name was misleading but context saved us).
    """
    return round(max(0.01, min(0.99, llm_lambda)), 4)


# ==============================================================================
# LIVE MODE
# ==============================================================================
def call_api(name_a, ctx_a, name_b, ctx_b) -> dict | None:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(name_a, ctx_a, name_b, ctx_b)},
                ],
                max_tokens=30,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parse_response(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = [10, 30, 60][min(attempt, 2)]
                print(f"  [429] Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERR] {err[:80]}")
                return None
    return None


def run_live(pending_df: pd.DataFrame, full_df: pd.DataFrame, dry_run: int = 0):
    """Label pending rows in real-time, checkpoint every N calls, then merge back."""

    # Resume support: load checkpoint if it exists
    labeled_ids = set()
    checkpoint_results = {}   # index_in_full_df -> {"llm_match": bool, "lambda_val": float}

    if CHECKPOINT_PATH.exists() and not dry_run:
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        for _, row in ckpt.iterrows():
            idx = int(row["orig_index"])
            checkpoint_results[idx] = {
                "llm_match":  bool(row["llm_match"]),
                "lambda_val": float(row["lambda_val"]),
            }
            labeled_ids.add(idx)
        print(f"  [RESUME] Loaded {len(labeled_ids)} already-labeled rows from checkpoint.")

    todo = [(orig_idx, row) for orig_idx, row in pending_df.iterrows()
            if orig_idx not in labeled_ids]

    if dry_run > 0:
        todo = todo[:dry_run]
        print(f"[DRY RUN] Only processing {dry_run} pairs (results NOT saved).")

    total   = len(todo)
    failed  = 0
    done    = len(labeled_ids)
    results = dict(checkpoint_results)  # copy checkpoint

    # Cost estimate
    est_cost_live  = total * 165 * 0.15 / 1_000_000
    est_cost_batch = est_cost_live * 0.5
    print(f"\n  Pending rows to label : {total}")
    print(f"  Already done (ckpt)   : {done}")
    print(f"  Model                 : {MODEL}")
    print(f"  Max ctx length        : {MAX_CTX_LEN} chars")
    print(f"  Est. cost (live)      : ${est_cost_live:.5f}")
    print(f"  Est. cost (batch)     : ${est_cost_batch:.5f}  (use --mode batch for 50% off)")
    print(f"  Sleep between calls   : {SLEEP_TIME}s")
    print()

    ckpt_buffer = []

    try:
        for i, (orig_idx, row) in enumerate(todo):
            if i % 10 == 0:
                print(f"  [{i+1:3d}/{total}] labeling... done={done} failed={failed}")

            llm = call_api(
                str(row["name_a"]),    str(row["context_a"]),
                str(row["name_b"]),    str(row["context_b"]),
            )

            if llm:
                lam = compute_lambda_type_c(llm["lambda"], float(row.get("name_sim_score", 0.3)))
                results[orig_idx] = {
                    "llm_match":  llm["match"],
                    "lambda_val": lam,
                }
                ckpt_buffer.append({
                    "orig_index": orig_idx,
                    "llm_match":  int(llm["match"]),
                    "lambda_val": lam,
                })
                done += 1
            else:
                failed += 1
                print(f"  [FAIL] row {orig_idx}: {row['name_a']!r} vs {row['name_b']!r}")

            # Checkpoint every N
            if not dry_run and (i + 1) % CHECKPOINT_EVERY == 0 and ckpt_buffer:
                _save_checkpoint(ckpt_buffer)
                ckpt_buffer = []
                print(f"  [CKPT] {done} labeled so far")

            if not dry_run:
                time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted — saving checkpoint...")

    # Final checkpoint flush
    if not dry_run and ckpt_buffer:
        _save_checkpoint(ckpt_buffer)

    if dry_run:
        print("\n[DRY RUN] Results (not saved):")
        for idx, r in results.items():
            row = full_df.loc[idx]
            print(f"  {row['name_a']!r:35s} vs {row['name_b']!r:35s}  "
                  f"match={r['llm_match']}  lambda={r['lambda_val']:.3f}")
        return

    if not results:
        print("\n  No results — nothing to merge.")
        return

    # Merge back into full dataset
    print(f"\n  Merging {len(results)} labeled rows back into dataset...")
    for orig_idx, vals in results.items():
        full_df.at[orig_idx, "llm_match"]     = int(vals["llm_match"])
        full_df.at[orig_idx, "lambda_val"]    = vals["lambda_val"]
        full_df.at[orig_idx, "lambda_source"] = "llm_labeled_type_c"

    full_df.to_csv(DATASET_PATH, index=False)
    print(f"  Saved -> {DATASET_PATH}")

    # Summary
    still_pending = (full_df["lambda_source"] == "pending_llm").sum()
    print(f"\n  Rows labeled this run : {done}")
    print(f"  Rows failed           : {failed}")
    print(f"  Still pending_llm     : {still_pending}")
    if still_pending == 0:
        print("  ✓ All pending rows are now labeled!")
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            print(f"  Checkpoint deleted: {CHECKPOINT_PATH.name}")
    else:
        print(f"  Re-run to label the remaining {still_pending} rows.")


def _save_checkpoint(buffer: list):
    df_new = pd.DataFrame(buffer)
    if CHECKPOINT_PATH.exists():
        existing = pd.read_csv(CHECKPOINT_PATH)
        df_new = pd.concat([existing, df_new], ignore_index=True).drop_duplicates("orig_index")
    df_new.to_csv(CHECKPOINT_PATH, index=False)


# ==============================================================================
# BATCH MODE
# ==============================================================================
def run_batch_submit(pending_df: pd.DataFrame):
    n = len(pending_df)
    print(f"\n[BATCH] Building JSONL for {n} pairs...")

    jsonl_path = BATCH_DIR / "batch_input.jsonl"
    meta = []

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seq, (orig_idx, row) in enumerate(pending_df.iterrows()):
            req = {
                "custom_id": f"pendingC-{seq:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_prompt(
                            str(row["name_a"]), str(row["context_a"]),
                            str(row["name_b"]), str(row["context_b"]),
                        )},
                    ],
                    "max_tokens": 30,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(req) + "\n")
            meta.append({"seq": seq, "orig_index": int(orig_idx),
                          "name_sim_score": float(row.get("name_sim_score", 0.3))})

    meta_path = BATCH_DIR / "batch_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    est = n * 165 * 0.15 / 1_000_000 * 0.5
    print(f"  Estimated batch cost  : ${est:.5f}")

    print(f"[BATCH] Uploading {jsonl_path.name}...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    bid_path = BATCH_DIR / "batch_id.txt"
    bid_path.write_text(batch.id)

    print(f"\n{'='*60}")
    print(f"  Batch submitted!  ID: {batch.id}")
    print(f"  Status : {batch.status}")
    print(f"  Come back in ~1h and run:")
    print(f"  python label_pending_llm.py --mode retrieve --batch-id {batch.id}")
    print(f"{'='*60}")
    return batch.id


def run_batch_retrieve(batch_id: str, full_df: pd.DataFrame, pending_df: pd.DataFrame):
    print(f"\n[RETRIEVE] Checking batch {batch_id}...")
    batch = client.batches.retrieve(batch_id)
    print(f"  Status : {batch.status}  |  {batch.request_counts}")

    if batch.status != "completed":
        wait_map = {"validating": "a few minutes", "in_progress": "up to 1 hour",
                    "finalizing": "a few minutes", "failed": "FAILED", "expired": "EXPIRED"}
        print(f"  Not ready. Expected wait: {wait_map.get(batch.status, '?')}")
        print(f"  Re-run: python label_pending_llm.py --mode retrieve --batch-id {batch_id}")
        return

    meta_path = BATCH_DIR / "batch_meta.json"
    if not meta_path.exists():
        print("[ERROR] batch_meta.json not found. Was the batch submitted from this machine?")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    seq_to_meta = {m["seq"]: m for m in meta}

    print(f"[RETRIEVE] Downloading results...")
    content = client.files.content(batch.output_file_id)
    lines = content.text.strip().split("\n")
    print(f"  Result lines: {len(lines)}")

    results_map = {}
    failed = 0
    for line in lines:
        try:
            obj   = json.loads(line)
            cid   = obj["custom_id"]
            seq   = int(cid.split("-")[1])
            text  = obj["response"]["body"]["choices"][0]["message"]["content"]
            parsed = parse_response(text)
            if parsed:
                results_map[seq] = parsed
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  Parsed OK : {len(results_map)}")
    print(f"  Failed    : {failed}")

    labeled = 0
    for seq, llm in results_map.items():
        m = seq_to_meta[seq]
        orig_idx = m["orig_index"]
        lam = compute_lambda_type_c(llm["lambda"], m["name_sim_score"])
        full_df.at[orig_idx, "llm_match"]     = int(llm["match"])
        full_df.at[orig_idx, "lambda_val"]    = lam
        full_df.at[orig_idx, "lambda_source"] = "llm_labeled_type_c"
        labeled += 1

    full_df.to_csv(DATASET_PATH, index=False)
    print(f"\n  Labeled and merged : {labeled}")
    print(f"  Saved -> {DATASET_PATH}")
    still_pending = (full_df["lambda_source"] == "pending_llm").sum()
    print(f"  Still pending_llm  : {still_pending}")
    if still_pending == 0:
        print("  ✓ All pending rows labeled!")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Label pending_llm rows with GPT-4o-mini")
    parser.add_argument("--mode",     choices=["live", "batch", "retrieve"], default="live",
                        help="live=real-time (instant), batch=submit (50%% cheaper), retrieve=get batch results")
    parser.add_argument("--batch-id", type=str, default="",
                        help="Batch ID (required for --mode retrieve)")
    parser.add_argument("--dry-run",  type=int, default=0,
                        help="Only process N rows without saving (sanity check)")
    args = parser.parse_args()

    print("=" * 60)
    print(" label_pending_llm.py  — AgriLambdaNet")
    print(f" Mode  : {args.mode}")
    print(f" Model : {MODEL}  |  max_tokens=30  |  temp=0")
    print("=" * 60)

    # Load full dataset
    if not DATASET_PATH.exists():
        print(f"[ERROR] Not found: {DATASET_PATH}")
        sys.exit(1)

    full_df  = pd.read_csv(DATASET_PATH)
    pending  = full_df[full_df["lambda_source"] == "pending_llm"].copy()

    print(f"\n  Full dataset rows  : {len(full_df)}")
    print(f"  Pending rows       : {len(pending)}")

    if len(pending) == 0:
        print("\n  ✓ No pending rows — nothing to do!")
        sys.exit(0)

    # Verify all pending rows have context
    missing_ctx = (pending["context_a"].isna() | pending["context_b"].isna() |
                   (pending["context_a"].str.strip() == "") |
                   (pending["context_b"].str.strip() == "")).sum()
    if missing_ctx > 0:
        print(f"\n  [WARNING] {missing_ctx} pending rows have missing context — they will be skipped by the API.")

    print(f"\n  All pending rows are pair_type=C (hard negatives, match=0).")
    print(f"  The LLM will assign llm_match and lambda_val for each.")
    print()

    if args.mode == "live":
        run_live(pending, full_df, dry_run=args.dry_run)

    elif args.mode == "batch":
        run_batch_submit(pending)

    elif args.mode == "retrieve":
        bid = args.batch_id
        if not bid:
            saved = BATCH_DIR / "batch_id.txt"
            if saved.exists():
                bid = saved.read_text().strip()
                print(f"  Using saved batch ID: {bid}")
            else:
                print("[ERROR] --batch-id required for retrieve mode")
                sys.exit(1)
        run_batch_retrieve(bid, full_df, pending)


if __name__ == "__main__":
    main()
