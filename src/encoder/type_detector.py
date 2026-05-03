# encoder/type_detector.py
# =============================================================================
# Detects the biological category of an agricultural entity from its name
# and context text using ordered keyword rules.
#
# This module is the FALLBACK — called only when type_a / type_b is null,
# "unknown", or empty. If the column already has a real value, use that
# directly; never call this function for known types.
#
# Why keyword rules, not a classifier?
#   Your dataset is 1,782 pairs — too small for a reliable type classifier.
#   Keyword rules are transparent, auditable, and trivially fixable.
# =============================================================================

# Each entry: (TAG_STRING, [lowercase keywords that signal this type])
# ORDER MATTERS — the first matching rule wins.
# Narrower/more-specific categories come FIRST.
_TYPE_RULES = [

    # ── Viruses ───────────────────────────────────────────────────────────────
    # First because many virus names contain "disease" too.
    ("VIRUS", [
        "virus", "viral", "viridae", "viricidae",
        "tobamovirus", "potyvirus", "begomovirus", "furovirus",
        "luteovirus", "geminivirus", "caulimovirus", "closterovirus",
        "iflavirus", "iflaviridae", "alphavirus", "bromovirus",
        "rna virus", "dna virus", "plant virus", "plant pathogenic virus",
        "positive sense rna", "negative sense rna",
    ]),

    # ── Bacteria ──────────────────────────────────────────────────────────────
    ("BACTERIA", [
        "bacteria", "bacterium", "bacterial",
        "xanthomonas", "pseudomonas", "erwinia", "pectobacterium",
        "streptomyces", "agrobacterium", "ralstonia", "burkholderia",
        "phytoplasma", "spiroplasma", "candidatus liberibacter",
        "gram-negative", "gram-positive", "actinobacterium",
    ]),

    # ── Fungi / Oomycetes ─────────────────────────────────────────────────────
    ("FUNGUS", [
        "fungal", "fungus", "fungi", "fungiculture",
        "ascomycete", "basidiomycete", "oomycete",
        "mold", "mould",
        "powdery mildew", "downy mildew",
        "fusarium", "alternaria", "phytophthora", "puccinia",
        "botrytis", "sclerotinia", "colletotrichum", "magnaporthe",
        "septoria", "cercospora", "pythium", "rhizoctonia",
        "aspergillus", "penicillium", "trichoderma", "plasmodiophora",
        "plasmodiophoraceae", "guignardia", "gymnosporangium",
        "mushroom", "mycelium", "spore", "hyphae", "conidium",
    ]),

    # ── Pests / Insects / Mites / Nematodes / Protozoa ───────────────────────
    ("PEST", [
        "insect", "beetle", "pest",
        "mite", "aphid", "larva", "larvae",
        "moth", "caterpillar", "thrips",
        "whitefly", "weevil", "termite", "locust", "leafhopper",
        "spider mite", "chrysomelidae", "lepidoptera", "coleoptera",
        "diptera", "hemiptera", "parasitic mite",
        "macrotermes", "diabrotica", "nematode",
        "protozoa", "protozoan",      # single-celled eukaryotic parasites
        "helminths", "roundworm",
    ]),

    # ── Weeds / Invasive Plants ───────────────────────────────────────────────
    ("WEED", [
        "weed", "invasive grass", "invasive plant", "itchgrass",
        "grass species",
        "rottboellia", "amaranthus", "cyperus", "echinochloa",
    ]),

    # ── Plants / Crops ────────────────────────────────────────────────────────
    # Comes AFTER virus/bacteria/fungus/pest to avoid false positives.
    ("PLANT", [
        "plant", "crop", "cultivar", "botanical",
        "prunus", "solanum", "zea mays", "oryza", "triticum",
        "glycine", "brassica", "lycopersicon", "mangifera",
        "cultivation of", "crop plant", "host plant",
        "tree", "shrub", "grass", "legume", "cereal",
    ]),

    # ── Generic Plant Diseases ────────────────────────────────────────────────
    # Broadest category — comes near the end to catch what slipped through.
    ("DISEASE", [
        "disease", "blight", "scorch", "leaf spot",
        "wilt", "canker", "mosaic", "yellowing", "necrosis",
        "damping off", "crown rot", "root rot", "stem rot", "rot ",
        "sudden death", "late blight", "early blight",
        "black rot", "brown spot", "leaf curl", "leaf scorch",
        "dieback", "chlorosis", "rust", "smut", "scab",
    ]),
]

# Fallback: matches something pathogen-like but no specific rule matched.
_FALLBACK_PATHOGEN_KEYWORDS = [
    "pathogen", "infects", "infection",
    "plant disease", "crop disease",
    "agricultural", "causes disease", "plant pathogen",
]


def detect_entity_type(name: str, context: str) -> str:
    """
    Return the best semantic tag for an entity from its name + context.

    Called ONLY when type_a / type_b is null or "unknown".
    If a real type is already in the CSV, use that — don't call this.

    Args:
        name    : Short label of the entity, e.g. "banana fusarium wilt"
        context : Description text

    Returns:
        One of: "VIRUS", "BACTERIA", "FUNGUS", "PEST", "WEED",
                "PLANT", "DISEASE", "PATHOGEN", "ENTITY"

    Examples:
        detect_entity_type("diabrotica", "beetle in family Chrysomelidae")
        -> "PEST"
        detect_entity_type("blb", "BLB may refer to Bad Berleburg Germany")
        -> "ENTITY"
    """
    # Combine name + context into one lowercase string for keyword scanning.
    text = (str(name) + " " + str(context)).lower()

    for tag, keywords in _TYPE_RULES:
        if any(kw in text for kw in keywords):
            return tag

    # Something pathogen-like but didn't match a specific type.
    if any(kw in text for kw in _FALLBACK_PATHOGEN_KEYWORDS):
        return "PATHOGEN"

    # Fully generic — used for noise/non-agricultural entities.
    return "ENTITY"


# ── Self-test (run directly: python -m src.encoder.type_detector) ─────────────
if __name__ == "__main__":
    import sys
    import io
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    test_cases = [
        ("banana fusarium wilt",
         "cubense is a fungal plant pathogen that causes Panama disease",
         "FUNGUS"),
        ("kashmir bee virus",
         "Iflaviridae is a family of positive sense RNA viruses",
         "VIRUS"),
        ("diabrotica",
         "Diabrotica is a large widespread genus of beetles Chrysomelidae",
         "PEST"),
        ("blb",
         "BLB may refer to Bad Berleburg Germany vehicle registration",
         "ENTITY"),
        ("pleurotus eryngii",
         "Fungiculture is the cultivation of fungi such as mushrooms",
         "FUNGUS"),
        ("protozoa",
         "Protozoa are a polyphyletic group of single-celled eukaryotes",
         "PEST"),
        ("sudden death syndrome",
         "Sudden death syndrome SDS a disease in soybean plants",
         "DISEASE"),
        ("wheat crop",
         "A cereal plant in the genus Triticum. Major food crop worldwide.",
         "PLANT"),
        ("xanthomonas oryzae",
         "Gram-negative bacterium causing bacterial leaf blight of rice.",
         "BACTERIA"),
    ]

    print("Type Detector Self-Test")
    print("-" * 68)
    all_pass = True
    for name, ctx, expected in test_cases:
        result = detect_entity_type(name, ctx)
        ok     = result == expected
        status = "[OK]   " if ok else "[FAIL] "
        if not ok:
            all_pass = False
        print(f"  {status}  {name[:38]:<38}  got={result:<12}  want={expected}")

    print()
    print("All tests passed!" if all_pass else "Some tests FAILED -- check rules above.")
