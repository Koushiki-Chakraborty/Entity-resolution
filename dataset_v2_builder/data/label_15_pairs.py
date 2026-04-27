"""
Label the 15 null-lambda pairs using GPT-4o-mini
=================================================
Reads OPENAI_API_KEY from the project .env file automatically,
so you don't need to export it manually.

Run:  python label_15_pairs.py
"""

import os
import json
import time
import pathlib
import pandas as pd

# ── Load .env from the project root (two levels up from this file) ─────────────
_HERE   = pathlib.Path(__file__).resolve().parent          # …/dataset_v2_builder/data/
_ROOT   = _HERE.parent.parent                              # …/entity_resolution_copy/
_DOTENV = _ROOT / ".env"

if _DOTENV.exists():
    with open(_DOTENV, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_CSV  = str(_HERE / "dataset_v2.csv")
OUTPUT_CSV = str(_HERE / "dataset_v2_labeled.csv")
MODEL      = "gpt-4o-mini"

# ── PROMPT ────────────────────────────────────────────────────────────────────
# Identical to the prompt used for the rest of the dataset.

PROMPT_TEMPLATE = """You are an agricultural expert.

Entity A: "{name_a}" | Context: "{ctx_a}"
Entity B: "{name_b}" | Context: "{ctx_b}"

Tasks:
1. Do these refer to the SAME crop disease? (true/false)
2. When deciding, how important is the NAME vs CONTEXT?
   Return lambda: 0.0 = context only, 1.0 = name only

Return ONLY valid JSON, nothing else:
{{"match": true_or_false, "lambda": 0.0_to_1.0}}"""


# ── LABELING FUNCTION ─────────────────────────────────────────────────────────

def label_one_pair(client, name_a, ctx_a, name_b, ctx_b):
    prompt = PROMPT_TEMPLATE.format(
        name_a = name_a,
        ctx_a  = str(ctx_a)[:200],
        name_b = name_b,
        ctx_b  = str(ctx_b)[:200],
    )
    try:
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
            max_tokens  = 60,       # we only need {"match":..., "lambda":...}
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        assert isinstance(result["match"], bool), f"match not bool: {result['match']}"
        lam = float(result["lambda"])
        assert 0.0 <= lam <= 1.0, f"lambda out of range: {lam}"
        return {"match": result["match"], "lambda": lam}
    except json.JSONDecodeError as e:
        print(f"    ERROR: invalid JSON returned: {raw!r} — {e}")
    except AssertionError as e:
        print(f"    ERROR: validation failed: {e}")
    except Exception as e:
        print(f"    ERROR: API call failed: {e}")
    return None


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY not found in environment or .env file.")
        print(f"Expected .env at: {_DOTENV}")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("\nERROR: openai library not installed.  Fix: pip install openai")
        return

    client = OpenAI(api_key=api_key)

    print()
    print("=" * 55)
    print(" Labeling null-lambda pairs with GPT-4o-mini")
    print("=" * 55)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: file not found: {INPUT_CSV}")
        return

    print(f"  Total rows    : {len(df)}")

    null_mask = df["lambda_val"].isnull()
    null_rows = df[null_mask].copy()
    print(f"  Null rows     : {len(null_rows)}")

    if len(null_rows) == 0:
        print("\nNothing to do — all rows already have lambda values.")
        return

    # ── Label ─────────────────────────────────────────────────────────────────
    print(f"\nSending {len(null_rows)} pairs to {MODEL} ...\n")

    ok, fail = 0, 0

    for i, (idx, row) in enumerate(null_rows.iterrows()):
        print(f"[{i+1:02d}/{len(null_rows)}] {row['name_a']!r:38s} vs {row['name_b']!r}")

        result = label_one_pair(
            client,
            row["name_a"], row["context_a"],
            row["name_b"], row["context_b"],
        )

        if result is not None:
            df.at[idx, "llm_match"]  = result["match"]
            df.at[idx, "lambda_val"] = result["lambda"]
            print(f"         match={result['match']}  lambda={result['lambda']:.3f}")
            ok += 1
        else:
            df.at[idx, "llm_match"]  = None
            df.at[idx, "lambda_val"] = None
            print("         FAILED — kept as null")
            fail += 1

        # 0.5 s gap → safe even on free-tier rate limits
        if i < len(null_rows) - 1:
            time.sleep(0.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    remaining = df["lambda_val"].isnull().sum()

    print()
    print("=" * 55)
    print(" Summary")
    print("=" * 55)
    print(f"  Labeled OK    : {ok}")
    print(f"  Failed        : {fail}")
    print(f"  Still null    : {remaining}")
    print(f"  Output file   : {OUTPUT_CSV}")
    if remaining == 0:
        print("\n  All rows labeled. Dataset is complete.")
    else:
        print(f"\n  {remaining} rows still null. Fix manually then run step3.")
    print()


if __name__ == "__main__":
    main()
