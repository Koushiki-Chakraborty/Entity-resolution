# encoder/frozen_encoder.py
# =============================================================================
# FrozenEncoder -- wraps the fine-tuned SentenceTransformer and produces
# the 4 tensors that PairAwareAgriLambdaNet needs as input.
#
# WHY FROZEN?
#   The encoder weights were already adapted to agricultural language during
#   fine-tuning. Freezing them here means training PairAwareAgriLambdaNet
#   won't accidentally corrupt the encoder -- the two objectives don't fight.
#
# WHAT THIS PRODUCES:
#   For each of the N pairs in the dataset:
#
#   name_vecs_a  [N, 384]  <- just the name of entity A
#   ctx_vecs_a   [N, 384]  <- full tagged context of entity A
#   name_vecs_b  [N, 384]  <- just the name of entity B
#   ctx_vecs_b   [N, 384]  <- full tagged context of entity B
#
#   At runtime PairAwareAgriLambdaNet concatenates:
#   [name_a | ctx_a | name_b | ctx_b | sim_name | sim_ctx]
#   = [384 | 384 | 384 | 384 | 1 | 1] = 1538 dims
#
# CACHING:
#   Encoding 1782 pairs takes ~30 s on CPU. Vectors are saved to
#   outputs/encoded_vectors.pt so subsequent runs load instantly.
#   Delete that file to force a re-encode after changing the serializer.
# =============================================================================

import sys
import io
if sys.platform.startswith("win"):
    # Prevent UnicodeEncodeError on Windows terminals (cp1252)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from encoder.config import (
    FINETUNED_MODEL_DIR,
    BASE_MODEL,
    EMB_DIM,
    VECTORS_PATH,
    ENCODE_BATCH_SIZE,
)
from encoder.serializer import serialize_name, serialize_context


class FrozenEncoder:
    """
    Loads the fine-tuned SentenceTransformer and encodes entity pairs
    into the four vector tensors used by PairAwareAgriLambdaNet.

    All weights are frozen -- pure inference, no gradient computation.
    """

    def __init__(self, force_base: bool = False):
        """
        Load the encoder model.

        Args:
            force_base : If True, use the base model (for ablation study).
                         Default False -- always use fine-tuned in production.
        """
        if force_base or not os.path.exists(FINETUNED_MODEL_DIR):
            model_path = BASE_MODEL
            if not force_base:
                print(
                    f"\n[Encoder] WARNING: Fine-tuned model not found at"
                    f" '{FINETUNED_MODEL_DIR}'."
                    f"\n  Falling back to base model '{BASE_MODEL}'."
                    f"\n  Run finetune.py first for best results.\n"
                )
        else:
            model_path = FINETUNED_MODEL_DIR

        print(f"\n[Encoder] Loading model: {model_path}")
        self.model      = SentenceTransformer(model_path)
        self.model_path = model_path

        # Freeze ALL parameters -- no gradients, no updates ever.
        frozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = False
            frozen_count += 1

        print(f"  Embedding dimension : {EMB_DIM}")
        print(f"  Frozen parameters   : {frozen_count} parameter tensors")
        print(f"  Model path          : {model_path}")

    # ── Batch encoding ────────────────────────────────────────────────────────

    def encode_batch(
        self,
        texts: list,
        batch_size: int = ENCODE_BATCH_SIZE,
        desc: str = "Encoding",
    ) -> torch.Tensor:
        """
        Encode a list of strings into a 2D tensor of shape [N, 384].

        Processes texts in chunks to avoid OOM. Uses torch.no_grad() for
        speed and memory efficiency (pure inference, no gradient tracking).

        Args:
            texts      : List of strings to encode.
            batch_size : Chunk size (lower if OOM).
            desc       : Label for the progress bar.

        Returns:
            torch.Tensor of shape [len(texts), EMB_DIM], dtype=float32.
        """
        all_vecs = []

        for start in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}"):
            chunk = texts[start : start + batch_size]
            with torch.no_grad():
                vecs = self.model.encode(
                    chunk,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            all_vecs.append(vecs.cpu())

        # Stack all chunk tensors: e.g. 28 x [64,384] -> [1782,384]
        return torch.cat(all_vecs, dim=0).float()

    # ── Full dataset encoding ─────────────────────────────────────────────────

    def encode_dataset(
        self,
        df,
        cache_path: str = VECTORS_PATH,
        force_recompute: bool = False,
    ) -> dict:
        """
        Encode all pairs in the dataframe into four tensors.

        This is the main function called before training PairAwareAgriLambdaNet.
        Reads type_a / type_b from the dataframe -- no external detection.

        Args:
            df              : Cleaned dataframe with columns:
                              name_a, context_a, type_a,
                              name_b, context_b, type_b
            cache_path      : Where to save/load the vector cache.
            force_recompute : If True, ignore the cache and re-encode.

        Returns:
            dict with keys:
                "name_vecs_a"  -- torch.Tensor [N, 384]
                "ctx_vecs_a"   -- torch.Tensor [N, 384]
                "name_vecs_b"  -- torch.Tensor [N, 384]
                "ctx_vecs_b"   -- torch.Tensor [N, 384]
        """
        # Required columns
        required = {"name_a", "context_a", "type_a",
                    "name_b", "context_b", "type_b"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[Encoder] Dataframe is missing columns: {missing}\n"
                f"  Load dataset_clean.csv which has type_a / type_b."
            )

        # Load from cache if available
        cache_file = Path(cache_path)
        if cache_file.exists() and not force_recompute:
            print(f"\n[Encoder] Loading cached vectors from {cache_file}")
            print(f"  (Delete this file or use --force to re-encode)")
            return torch.load(cache_file, weights_only=True)

        print(f"\n[Encoder] Encoding {len(df)} entity pairs ...")

        # Show the type distribution from the CSV
        all_types = Counter(
            df["type_a"].str.lower().tolist() +
            df["type_b"].str.lower().tolist()
        )
        print("\n  Entity type distribution (from CSV type_a + type_b):")
        for t, count in sorted(all_types.items(), key=lambda x: -x[1]):
            print(f"    {t:<15} : {count}")

        # Build the four text lists
        print("\n  Serializing entity strings ...")
        names_a    = [serialize_name(n)                               for n in df["name_a"]]
        contexts_a = [serialize_context(n, c, t)
                      for n, c, t in zip(df["name_a"], df["context_a"], df["type_a"])]
        names_b    = [serialize_name(n)                               for n in df["name_b"]]
        contexts_b = [serialize_context(n, c, t)
                      for n, c, t in zip(df["name_b"], df["context_b"], df["type_b"])]

        # Sanity print -- first 3 rows
        print("\n  Sample serialized strings (rows 0-2):")
        for i in range(min(3, len(df))):
            print(f"    Row {i}:")
            print(f"      name_a : {names_a[i]}")
            print(f"      ctx_a  : {contexts_a[i][:100]}...")
            print(f"      name_b : {names_b[i]}")
            print(f"      ctx_b  : {contexts_b[i][:100]}...")
        print()

        # Encode all four lists
        vectors = {
            "name_vecs_a" : self.encode_batch(names_a,    desc="Names A   "),
            "ctx_vecs_a"  : self.encode_batch(contexts_a, desc="Contexts A"),
            "name_vecs_b" : self.encode_batch(names_b,    desc="Names B   "),
            "ctx_vecs_b"  : self.encode_batch(contexts_b, desc="Contexts B"),
        }

        # Confirm shapes
        print("\n  Tensor shapes:")
        for key, tensor in vectors.items():
            print(f"    {key:<14} : {tuple(tensor.shape)}")
        n_input = 4 * EMB_DIM + 2   # 1538
        print(f"\n  PairAwareAgriLambdaNet input size: {n_input} dims")
        print(f"    = 4 x {EMB_DIM} (vecs) + 2 (sim_name + sim_ctx)")

        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, cache_file)
        print(f"\n  Vectors cached -> {cache_file}")

        return vectors

    # ── Similarity computation ────────────────────────────────────────────────

    def compute_similarities(self, vectors: dict) -> dict:
        """
        Pre-compute the two cosine similarity scores for every pair.

        These become the last 2 dims of the net's input vector:
            sim_name = cosine(name_a, name_b)  -- how similar are the labels?
            sim_ctx  = cosine(ctx_a,  ctx_b)   -- how similar are the meanings?

        The conflict detector uses:
            conflict     = |sim_name - sim_ctx|
            eff_lambda   = raw_lambda * (1 - conflict)

        Args:
            vectors : dict returned by encode_dataset()

        Returns:
            dict with keys "sim_name" and "sim_ctx", each shape [N] float.
        """
        sim_name = F.cosine_similarity(
            vectors["name_vecs_a"], vectors["name_vecs_b"], dim=-1
        )
        sim_ctx = F.cosine_similarity(
            vectors["ctx_vecs_a"], vectors["ctx_vecs_b"], dim=-1
        )
        return {"sim_name": sim_name, "sim_ctx": sim_ctx}

    # ── Verification report ───────────────────────────────────────────────────

    def verification_report(self, vectors: dict, df) -> None:
        """
        Print a sanity-check report after encoding.

        Verifies that encoded vectors separate match=1 from match=0 pairs.
        Positive separation means the encoder works correctly.
        If separation is near 0 or negative, check the model or data.

        Args:
            vectors : dict from encode_dataset()
            df      : Original dataframe (needs 'match' column)
        """
        print("\n-- Encoder Verification Report --")

        sims   = self.compute_similarities(vectors)
        labels = torch.tensor(df["match"].values)

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        for signal, sim_tensor in [("name_sim", sims["sim_name"]),
                                   ("ctx_sim",  sims["sim_ctx"])]:
            pos_mean = sim_tensor[pos_mask].mean().item()
            neg_mean = sim_tensor[neg_mask].mean().item()
            sep      = pos_mean - neg_mean
            verdict  = "GOOD" if sep > 0.1 else "LOW -- check encoder"

            print(f"\n  {signal}:")
            print(f"    match=1  mean : {pos_mean:.4f}  (n={pos_mask.sum().item()})")
            print(f"    match=0  mean : {neg_mean:.4f}  (n={neg_mask.sum().item()})")
            print(f"    separation    : {sep:+.4f}  [{verdict}]")

        # Per pair_type breakdown
        if "pair_type" in df.columns:
            print("\n  ctx_sim by pair_type:")
            ctx_sim = sims["sim_ctx"].numpy()
            match   = df["match"].values
            ptypes  = df["pair_type"].values
            for pt in sorted(set(ptypes)):
                all_mask = ptypes == pt
                pos_pt   = all_mask & (match == 1)
                neg_pt   = all_mask & (match == 0)
                info = f"n={all_mask.sum():4d}"
                if pos_pt.sum() > 0:
                    info += f"  match=1 mean={ctx_sim[pos_pt].mean():.4f}"
                if neg_pt.sum() > 0:
                    info += f"  match=0 mean={ctx_sim[neg_pt].mean():.4f}"
                print(f"    pair_type {pt} : {info}")

        # Per entity type breakdown (match=1 rows only)
        if "type_a" in df.columns:
            print("\n  ctx_sim by type_a (match=1 only):")
            ctx_sim = sims["sim_ctx"].numpy()
            match   = df["match"].values
            types   = df["type_a"].str.lower().values
            for et in sorted(set(types)):
                mask = (match == 1) & (types == et)
                if mask.sum() < 3:
                    continue
                mean_sim = ctx_sim[mask].mean()
                print(f"    {et:<14} : n={mask.sum():4d}  mean ctx_sim={mean_sim:.4f}")

        print("\n-- End Verification Report --")
