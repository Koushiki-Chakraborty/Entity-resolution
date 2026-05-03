# encoder/serializer.py
# =============================================================================
# Converts a (name, context, entity_type) triple into the TWO text strings
# that get fed into the frozen encoder per entity.
#
# WHY TWO STRINGS PER ENTITY?
# ─────────────────────────────────────────────────────────────────────────────
# PairAwareAgriLambdaNet needs two independent signals per entity:
#
#   1. name_vec  — encodes just the short label: "banana fusarium wilt"
#                  Answers: how similar are the NAMES of the two entities?
#                  High for synonyms, low for completely different labels.
#
#   2. ctx_vec   — encodes the full tagged string with name + context
#                  Answers: how similar is the MEANING of the two entities?
#                  High when they refer to the same biological concept,
#                  even when names are totally different.
#
# The lambda network learns: when to trust name, when to trust context.
# The conflict detector checks whether name and context AGREE for a pair.
# If they strongly disagree (polysemy like "rust" = 12 different diseases),
# lambda collapses toward zero -> trust context, ignore name.
#
# The four tensors produced by FrozenEncoder are:
#   name_vecs_a [N, 384]  -+
#   ctx_vecs_a  [N, 384]  -+- entity A
#   name_vecs_b [N, 384]  -+- entity B
#   ctx_vecs_b  [N, 384]  -+
#
# These 4 x 384 = 1536 dims, plus 2 pre-computed similarities = 1538 dims
# total input to PairAwareAgriLambdaNet.
# =============================================================================

from encoder.config import UNKNOWN_TYPE_VALUES, FALLBACK_TAG
from encoder.type_detector import detect_entity_type


def _resolve_tag(name: str, context: str, entity_type) -> str:
    """
    Internal helper — converts the raw type_a / type_b column value into a
    clean UPPERCASE tag string ready for use in the serialized text.

    Rules:
      - null / "unknown" / empty -> call detect_entity_type() as fallback.
      - Otherwise: uppercase + strip the existing value.
      - If detector still can't resolve -> FALLBACK_TAG ("ENTITY").

    Args:
        name        : Entity name (used by detector if type is unknown)
        context     : Entity context (used by detector if type is unknown)
        entity_type : Value from type_a or type_b column (may be NaN/unknown)

    Returns:
        One clean uppercase tag string, e.g. "FUNGUS", "PEST", "ENTITY"
    """
    raw = str(entity_type).lower().strip()

    if raw in UNKNOWN_TYPE_VALUES:
        detected = detect_entity_type(name, context)
        return detected

    # Map common aliases that can appear in the CSV.
    _ALIAS = {
        "oomycete": "FUNGUS",
        "protozoa": "PEST",
        "protozoan": "PEST",
        "mold": "FUNGUS",
        "mould": "FUNGUS",
    }
    if raw in _ALIAS:
        return _ALIAS[raw]

    return raw.upper()


def serialize_name(name: str) -> str:
    """
    Return just the raw entity name, stripped of whitespace.
    No tags. No context.

    Used to produce name_vec -- the vector capturing how similar two entity
    LABELS are, independent of their biological descriptions.

    Args:
        name : Entity name from name_a or name_b column.

    Returns:
        Stripped name string.

    Example:
        serialize_name("  banana fusarium wilt  ")
        -> "banana fusarium wilt"
    """
    return str(name).strip()


def serialize_context(name: str, context: str, entity_type) -> str:
    """
    Return a Ditto-style tagged string combining entity name, type tag,
    and context description.

    Used to produce ctx_vec -- the vector capturing the actual MEANING of
    the entity, so two entities referring to the same disease get similar
    vectors even when their names are completely different.

    Format:
        [TAG] name [/TAG] [CONTEXT] context text [/CONTEXT]

    The TAG is resolved from type_a / type_b; "unknown" / null values are
    handled automatically by the type_detector.

    Args:
        name        : Entity name string.
        context     : Entity context / description string.
        entity_type : Value from type_a or type_b (may be "unknown" or NaN).

    Returns:
        Full tagged string ready for the SentenceTransformer encoder.

    Examples:
        serialize_context("banana fusarium wilt",
                          "cubense is a fungal pathogen...", "fungus")
        -> "[FUNGUS] banana fusarium wilt [/FUNGUS] [CONTEXT] cubense is... [/CONTEXT]"

        serialize_context("blb",
                          "BLB may refer to Bad Berleburg Germany...",
                          "unknown")
        -> "[ENTITY] blb [/ENTITY] [CONTEXT] BLB may refer to... [/CONTEXT]"
    """
    tag     = _resolve_tag(name, context, entity_type)
    name    = str(name).strip()
    context = str(context).strip()
    return f"[{tag}] {name} [/{tag}] [CONTEXT] {context} [/CONTEXT]"


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Serializer Self-Test")
    print("-" * 72)

    cases = [
        # (name, context, type, expected_opening_tag)
        ("banana fusarium wilt",
         "cubense is a fungal plant pathogen that causes Panama disease",
         "fungus", "[FUNGUS]"),

        ("kashmir bee virus",
         "Iflaviridae is a family of positive sense RNA viruses",
         "virus", "[VIRUS]"),

        ("blb",
         "BLB may refer to Bad Berleburg Germany vehicle registration",
         "unknown", "[ENTITY]"),         # detector can't classify -> ENTITY

        ("pleurotus eryngii",
         "Fungiculture is the cultivation of fungi such as mushrooms",
         "unknown", "[FUNGUS]"),         # detector finds "fungi" -> FUNGUS

        ("sudden death syndrome",
         "Sudden death syndrome SDS a disease in soybean plants",
         "disease", "[DISEASE]"),

        ("xanthomonas oryzae",
         "Gram-negative bacterium causing bacterial leaf blight of rice",
         "bacteria", "[BACTERIA]"),
    ]

    all_pass = True
    for name, ctx, typ, expected_tag in cases:
        name_out = serialize_name(name)
        ctx_out  = serialize_context(name, ctx, typ)

        tag_ok   = ctx_out.startswith(expected_tag)
        status   = "[OK]  " if tag_ok else "[FAIL]"
        if not tag_ok:
            all_pass = False

        print(f"  {status}  name_out : '{name_out}'")
        print(f"          ctx_out  : {ctx_out[:80]}...")
        print()

    print("All tests passed!" if all_pass else "Some tests FAILED.")
