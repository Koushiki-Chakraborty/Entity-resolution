"""
01_scrape_plantvillage.py — Extract disease entities from PlantVillage (Kaggle)
AgriΛNet Entity Resolution Pipeline

═══════════════════════════════════════════════════════════════════════════════
BEGINNER EXPLANATION — What is this script doing?
═══════════════════════════════════════════════════════════════════════════════

PlantVillage is a dataset on Kaggle:
  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

It contains ~54,000 images of diseased and healthy plant leaves.
The images are organised into FOLDERS. Each folder name tells you:
    CropName___DiseaseName
    e.g. "Tomato___Late_blight"
         "Apple___Apple_scab"
         "Corn_(maize)___Common_rust_"

We DON'T need the images. We only need the FOLDER NAMES.
The folder names give us:
  1. The crop (e.g. Tomato)
  2. The disease name (e.g. Late blight)
  3. Implicit pairs: "Tomato___Late_blight" and "Late_blight" are related

Then we look up disease descriptions from a built-in reference dictionary
(so we don't need internet access for this part).

WHAT YOU NEED BEFORE RUNNING:
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  2. Unzip it — you'll get a folder like "plantvillage dataset/color/"
  3. Set PLANTVILLAGE_PATH below to where your folder is

ALTERNATIVELY: If you just want to run without downloading,
  set USE_BUILTIN_ONLY = True — the script will use our built-in disease list.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import re
import itertools
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_entity_record, save_raw, normalise_name,
    generate_name_variants, PAIRS_DIR
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Change PLANTVILLAGE_PATH to where you downloaded the Kaggle dataset
# ─────────────────────────────────────────────────────────────────────────────

PLANTVILLAGE_PATH = Path("data/raw/plantvillage_dataset/color")  # update this!
USE_BUILTIN_ONLY  = True   # Set to False if you have the Kaggle dataset

SOURCE_NAME = "plantvillage"
SOURCE_URL_TEMPLATE = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN DISEASE REFERENCE
#
# This is a carefully researched dictionary of:
#   disease canonical name → { description, pathogen, affected crops }
#
# Source: PlantVillage paper (Mohanty et al. 2016) + plant pathology literature
# This covers ALL 38 disease categories in the Kaggle PlantVillage dataset.
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_REFERENCE = {
    # ── Apple diseases ───────────────────────────────────────────────────────
    "apple scab": {
        "context": "Fungal disease of apple caused by Venturia inaequalis. Produces olive-green to black lesions on leaves and fruit surface. Major cause of apple crop losses worldwide.",
        "pathogen": "Venturia inaequalis",
        "crops": ["apple"],
        "variants": ["Apple Scab", "APPLE SCAB", "apple-scab", "scab of apple",
                     "Venturia scab", "apple scab disease"],
    },
    "apple black rot": {
        "context": "Fungal disease caused by Botryosphaeria obtusa. Creates circular brown lesions on fruit and frog-eye leaf spot. Common in humid climates.",
        "pathogen": "Botryosphaeria obtusa",
        "crops": ["apple"],
        "variants": ["Apple Black Rot", "APPLE BLACK ROT", "apple black-rot",
                     "Black Rot of apple", "ABR", "apple rot"],
    },
    "cedar apple rust": {
        "context": "Fungal disease caused by Gymnosporangium juniperi-virginianae requiring two hosts: cedar/juniper and apple. Causes yellow-orange spots on apple leaves.",
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "crops": ["apple", "cedar"],
        "variants": ["Cedar Apple Rust", "CEDAR APPLE RUST", "cedar-apple rust",
                     "CAR", "apple rust", "Gymnosporangium rust"],
    },

    # ── Corn/Maize diseases ──────────────────────────────────────────────────
    "corn common rust": {
        "context": "Fungal disease of maize caused by Puccinia sorghi. Produces small, circular to oval pustules on both leaf surfaces. Widespread in temperate regions.",
        "pathogen": "Puccinia sorghi",
        "crops": ["corn", "maize"],
        "variants": ["Corn Common Rust", "Common Rust of Maize", "COMMON RUST",
                     "maize rust", "common corn rust", "Puccinia rust", "CCR"],
    },
    "northern leaf blight": {
        "context": "Fungal disease of corn caused by Exserohilum turcicum. Creates long cigar-shaped grey-green lesions on leaves. Major yield-reducing disease.",
        "pathogen": "Exserohilum turcicum",
        "crops": ["corn", "maize"],
        "variants": ["Northern Leaf Blight", "NORTHERN LEAF BLIGHT", "NLB",
                     "northern corn leaf blight", "NCLB", "Turcicum blight",
                     "northern blight of maize"],
    },
    "gray leaf spot": {
        "context": "Fungal disease of maize caused by Cercospora zeae-maydis. Produces rectangular, tan to grey lesions with distinct margins, running parallel to leaf veins.",
        "pathogen": "Cercospora zeae-maydis",
        "crops": ["corn", "maize"],
        "variants": ["Gray Leaf Spot", "Grey Leaf Spot", "GRAY LEAF SPOT",
                     "GLS", "Cercospora leaf spot of maize", "gray spot"],
    },

    # ── Tomato diseases ──────────────────────────────────────────────────────
    "tomato late blight": {
        "context": "Devastating oomycete disease of tomato and potato caused by Phytophthora infestans. Causes rapid water-soaked lesions that turn brown-black. Responsible for the Irish Potato Famine.",
        "pathogen": "Phytophthora infestans",
        "crops": ["tomato", "potato"],
        "variants": ["Tomato Late Blight", "Late Blight of Tomato", "LATE BLIGHT",
                     "late blight", "LB", "Phytophthora blight", "tomato blight",
                     "late blight disease"],
    },
    "tomato early blight": {
        "context": "Fungal disease of tomato caused by Alternaria solani. Produces dark concentric ring lesions giving a target-board appearance. Affects leaves, stems and fruit.",
        "pathogen": "Alternaria solani",
        "crops": ["tomato", "potato"],
        "variants": ["Tomato Early Blight", "Early Blight of Tomato", "EARLY BLIGHT",
                     "early blight", "EB", "Alternaria blight", "early blight disease"],
    },
    "tomato bacterial spot": {
        "context": "Bacterial disease of tomato caused by Xanthomonas species. Produces small, water-soaked spots on leaves and fruit. Favoured by warm, wet conditions.",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "crops": ["tomato", "pepper"],
        "variants": ["Tomato Bacterial Spot", "BACTERIAL SPOT", "bacterial spot",
                     "bacterial spot of tomato", "TBS", "Xanthomonas spot"],
    },
    "tomato leaf mold": {
        "context": "Fungal disease caused by Passalora fulva (formerly Fulvia fulva). Creates pale green to yellow spots on upper leaf surface and olive-green mold on lower surface.",
        "pathogen": "Passalora fulva",
        "crops": ["tomato"],
        "variants": ["Tomato Leaf Mold", "TOMATO LEAF MOLD", "leaf mold of tomato",
                     "leaf mould", "tomato mold", "Cladosporium leaf mold"],
    },
    "tomato septoria leaf spot": {
        "context": "Fungal disease caused by Septoria lycopersici. Produces numerous small circular spots with dark borders and light grey centres. Rapidly defoliates plants.",
        "pathogen": "Septoria lycopersici",
        "crops": ["tomato"],
        "variants": ["Tomato Septoria Leaf Spot", "SEPTORIA LEAF SPOT",
                     "Septoria spot", "septoria blight", "SLS", "leaf spot of tomato"],
    },
    "tomato spider mites": {
        "context": "Two-spotted spider mite (Tetranychus urticae) infestation of tomato. Causes stippling, yellowing and bronzing of leaves. Thrives in hot dry conditions.",
        "pathogen": "Tetranychus urticae",
        "crops": ["tomato"],
        "variants": ["Tomato Spider Mites", "Spider Mite of Tomato", "SPIDER MITES",
                     "two spotted spider mite", "TSSM", "Tetranychus infestation"],
    },
    "tomato target spot": {
        "context": "Fungal disease caused by Corynespora cassiicola. Produces circular lesions with concentric rings giving a target appearance. Affects leaves, stems and fruit.",
        "pathogen": "Corynespora cassiicola",
        "crops": ["tomato"],
        "variants": ["Tomato Target Spot", "TARGET SPOT", "target spot of tomato",
                     "Corynespora spot", "target leaf spot"],
    },
    "tomato yellow leaf curl virus": {
        "context": "Viral disease caused by Tomato yellow leaf curl virus (TYLCV), transmitted by whitefly Bemisia tabaci. Causes severe yellowing, curling and stunting.",
        "pathogen": "Tomato yellow leaf curl virus (TYLCV)",
        "crops": ["tomato"],
        "variants": ["Tomato Yellow Leaf Curl Virus", "TYLCV", "yellow leaf curl",
                     "tomato leaf curl", "YLCV", "yellow curl virus"],
    },
    "tomato mosaic virus": {
        "context": "Viral disease caused by Tomato mosaic virus (ToMV). Produces mottled light and dark green mosaic pattern on leaves. Highly persistent in plant debris.",
        "pathogen": "Tomato mosaic virus (ToMV)",
        "crops": ["tomato"],
        "variants": ["Tomato Mosaic Virus", "ToMV", "tomato mosaic", "TMV",
                     "mosaic virus of tomato", "tobacco mosaic virus"],
    },

    # ── Potato diseases ──────────────────────────────────────────────────────
    "potato late blight": {
        "context": "Oomycete disease of potato caused by Phytophthora infestans. Causes dark water-soaked lesions on leaves and tuber rot. The same pathogen that caused the 1840s Irish Famine.",
        "pathogen": "Phytophthora infestans",
        "crops": ["potato"],
        "variants": ["Potato Late Blight", "Late Blight of Potato", "POTATO LATE BLIGHT",
                     "PLB", "Phytophthora blight of potato", "potato blight"],
    },
    "potato early blight": {
        "context": "Fungal disease of potato caused by Alternaria solani. Produces dark brown target-board lesions on lower leaves first, progressing upward.",
        "pathogen": "Alternaria solani",
        "crops": ["potato"],
        "variants": ["Potato Early Blight", "POTATO EARLY BLIGHT", "PEB",
                     "Alternaria blight of potato", "early blight of potato"],
    },

    # ── Grape diseases ───────────────────────────────────────────────────────
    "grape black rot": {
        "context": "Fungal disease of grape caused by Guignardia bidwellii. Causes small yellow-green spots on leaves and shrivels fruit into hard black mummies.",
        "pathogen": "Guignardia bidwellii",
        "crops": ["grape"],
        "variants": ["Grape Black Rot", "GRAPE BLACK ROT", "Black Rot of Grape",
                     "GBR", "grape rot", "Guignardia rot"],
    },
    "grape esca (black measles)": {
        "context": "Complex fungal disease of grapevine caused by multiple fungi including Phaeomoniella chlamydospora. Causes tiger-stripe leaf pattern and internal wood discolouration.",
        "pathogen": "Phaeomoniella chlamydospora / Phaeoacremonium species",
        "crops": ["grape"],
        "variants": ["Grape Esca", "Black Measles", "ESCA", "grape black measles",
                     "esca disease", "grapevine esca"],
    },
    "grape leaf blight": {
        "context": "Fungal disease of grape caused by Pseudocercospora vitis. Creates irregular brown lesions on leaves. Also called Isariopsis leaf spot.",
        "pathogen": "Pseudocercospora vitis",
        "crops": ["grape"],
        "variants": ["Grape Leaf Blight", "GRAPE LEAF BLIGHT", "Isariopsis leaf spot",
                     "grape leaf spot", "GLB"],
    },
    "grape powdery mildew": {
        "context": "Fungal disease of grape caused by Uncinula necator (Erysiphe necator). Produces white powdery growth on young leaves, shoots and berries.",
        "pathogen": "Erysiphe necator",
        "crops": ["grape"],
        "variants": ["Grape Powdery Mildew", "POWDERY MILDEW", "powdery mildew of grape",
                     "oidium", "grape oidium", "GPM"],
    },

    # ── Citrus diseases ──────────────────────────────────────────────────────
    "citrus greening": {
        "context": "Bacterial disease of citrus caused by Candidatus Liberibacter asiaticus, transmitted by Asian citrus psyllid. Causes yellowing, misshapen bitter fruit. No cure exists.",
        "pathogen": "Candidatus Liberibacter asiaticus",
        "crops": ["orange", "citrus"],
        "variants": ["Citrus Greening", "CITRUS GREENING", "Huanglongbing",
                     "HLB", "yellow dragon disease", "citrus HLB"],
    },
    "citrus haunglongbing": {
        "context": "Same as Citrus Greening — Huanglongbing (HLB) is the Chinese name for the bacterial citrus greening disease. Blotchy mottling on leaves; asymmetric, bitter fruits.",
        "pathogen": "Candidatus Liberibacter asiaticus",
        "crops": ["orange", "citrus"],
        "variants": ["Huanglongbing", "HLB", "haunglongbing", "HUANGLONGBING",
                     "yellow shoot disease"],
    },

    # ── Strawberry diseases ──────────────────────────────────────────────────
    "strawberry leaf scorch": {
        "context": "Fungal disease of strawberry caused by Diplocarpon earlianum. Creates irregular purple to brown spots on upper leaf surfaces, giving a 'scorched' appearance.",
        "pathogen": "Diplocarpon earlianum",
        "crops": ["strawberry"],
        "variants": ["Strawberry Leaf Scorch", "STRAWBERRY LEAF SCORCH",
                     "leaf scorch of strawberry", "strawberry scorch", "SLS"],
    },

    # ── Peach diseases ───────────────────────────────────────────────────────
    "peach bacterial spot": {
        "context": "Bacterial disease of peach caused by Xanthomonas arboricola pv. pruni. Causes water-soaked spots on leaves, fruit, and twigs. Causes defoliation in severe cases.",
        "pathogen": "Xanthomonas arboricola pv. pruni",
        "crops": ["peach"],
        "variants": ["Peach Bacterial Spot", "PEACH BACTERIAL SPOT",
                     "bacterial spot of peach", "PBS", "Xanthomonas peach spot"],
    },

    # ── Pepper diseases ──────────────────────────────────────────────────────
    "pepper bacterial spot": {
        "context": "Bacterial disease of pepper caused by Xanthomonas euvesicatoria. Produces water-soaked, irregular spots on leaves and fruit. Favoured by warm rainy weather.",
        "pathogen": "Xanthomonas euvesicatoria",
        "crops": ["pepper", "bell pepper"],
        "variants": ["Pepper Bacterial Spot", "PEPPER BACTERIAL SPOT",
                     "bacterial spot of pepper", "PBS pepper"],
    },

    # ── Cherry diseases ──────────────────────────────────────────────────────
    "cherry powdery mildew": {
        "context": "Fungal disease of cherry caused by Podosphaera clandestina. Forms white powdery patches on young leaves and shoots. Distorts leaf shape.",
        "pathogen": "Podosphaera clandestina",
        "crops": ["cherry"],
        "variants": ["Cherry Powdery Mildew", "CHERRY POWDERY MILDEW",
                     "powdery mildew of cherry", "CPM"],
    },

    # ── Squash diseases ──────────────────────────────────────────────────────
    "squash powdery mildew": {
        "context": "Fungal disease of squash and cucurbits caused by Podosphaera xanthii. White powdery growth on leaves reduces photosynthesis and fruit quality.",
        "pathogen": "Podosphaera xanthii",
        "crops": ["squash", "cucumber", "pumpkin"],
        "variants": ["Squash Powdery Mildew", "SQUASH POWDERY MILDEW",
                     "cucurbit powdery mildew", "SPM", "powdery mildew of squash"],
    },

    # ── Rice diseases ────────────────────────────────────────────────────────
    "rice blast": {
        "context": "Fungal disease of rice caused by Magnaporthe oryzae. Most devastating rice disease worldwide. Creates spindle-shaped grey lesions with brown borders. Can destroy entire crops.",
        "pathogen": "Magnaporthe oryzae",
        "crops": ["rice"],
        "variants": ["Rice Blast", "RICE BLAST", "blast disease of rice",
                     "RB", "rice blast disease", "Pyricularia blast", "neck rot"],
    },
    "rice brown spot": {
        "context": "Fungal disease of rice caused by Cochliobolus miyabeanus. Creates brown oval lesions on leaves. Associated with poor soil nutrition.",
        "pathogen": "Cochliobolus miyabeanus",
        "crops": ["rice"],
        "variants": ["Rice Brown Spot", "RICE BROWN SPOT", "brown spot of rice",
                     "Helminthosporium leaf spot", "RBS"],
    },
    "rice bacterial blight": {
        "context": "Bacterial disease of rice caused by Xanthomonas oryzae pv. oryzae. One of the most serious rice diseases. Causes wilting and yellowing of leaves, starting from leaf tips.",
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "crops": ["rice"],
        "variants": ["Rice Bacterial Blight", "RICE BACTERIAL BLIGHT",
                     "bacterial leaf blight of rice", "BLB", "rice blight", "Xoo blight"],
    },

    # ── Wheat diseases ───────────────────────────────────────────────────────
    "wheat stripe rust": {
        "context": "Fungal disease of wheat caused by Puccinia striiformis f. sp. tritici. Produces yellow-orange stripe-like pustules along leaf veins. A major global wheat threat.",
        "pathogen": "Puccinia striiformis f. sp. tritici",
        "crops": ["wheat"],
        "variants": ["Wheat Stripe Rust", "STRIPE RUST", "yellow rust of wheat",
                     "YR", "stripe rust", "Pst", "wheat yellow rust"],
    },
    "wheat leaf rust": {
        "context": "Fungal disease of wheat caused by Puccinia triticina. Creates orange-brown circular pustules scattered on leaf surface. The most common wheat rust worldwide.",
        "pathogen": "Puccinia triticina",
        "crops": ["wheat"],
        "variants": ["Wheat Leaf Rust", "LEAF RUST", "brown rust of wheat",
                     "LR", "leaf rust", "Pt", "wheat brown rust"],
    },
    "wheat powdery mildew": {
        "context": "Fungal disease of wheat caused by Blumeria graminis f. sp. tritici. White powdery colonies on leaves reduce photosynthesis. Favoured by cool, humid weather.",
        "pathogen": "Blumeria graminis f. sp. tritici",
        "crops": ["wheat"],
        "variants": ["Wheat Powdery Mildew", "WHEAT POWDERY MILDEW",
                     "powdery mildew of wheat", "WPM", "cereal powdery mildew"],
    },
    "wheat fusarium head blight": {
        "context": "Fungal disease of wheat caused by Fusarium graminearum. Causes premature bleaching of wheat heads and contaminates grain with mycotoxins (deoxynivalenol).",
        "pathogen": "Fusarium graminearum",
        "crops": ["wheat"],
        "variants": ["Wheat Fusarium Head Blight", "FHB", "head scab",
                     "Fusarium scab", "scab of wheat", "head blight"],
    },

    # ── Soybean diseases ─────────────────────────────────────────────────────
    "soybean rust": {
        "context": "Fungal disease of soybean caused by Phakopsora pachyrhizi. Spreads rapidly in warm humid conditions causing defoliation and yield loss up to 80%.",
        "pathogen": "Phakopsora pachyrhizi",
        "crops": ["soybean"],
        "variants": ["Soybean Rust", "SOYBEAN RUST", "Asian soybean rust",
                     "ASR", "soya rust", "SBR"],
    },
    "soybean sudden death syndrome": {
        "context": "Fungal disease caused by Fusarium virguliforme. Infects roots early in season but foliar symptoms appear at reproductive stage: interveinal chlorosis and necrosis.",
        "pathogen": "Fusarium virguliforme",
        "crops": ["soybean"],
        "variants": ["Sudden Death Syndrome", "SDS", "soybean SDS",
                     "Fusarium sudden death", "soybean sudden death"],
    },

    # ── Banana diseases ──────────────────────────────────────────────────────
    "banana fusarium wilt": {
        "context": "Fungal disease of banana caused by Fusarium oxysporum f. sp. cubense. Also called Panama disease. Blocks water transport causing yellowing and wilting. Threatens global banana supply.",
        "pathogen": "Fusarium oxysporum f. sp. cubense",
        "crops": ["banana"],
        "variants": ["Banana Fusarium Wilt", "Panama disease", "BANANA WILT",
                     "Fusarium wilt of banana", "Foc", "TR4", "banana panama disease"],
    },
    "banana black sigatoka": {
        "context": "Fungal disease of banana caused by Mycosphaerella fijiensis. Creates dark streaks on leaves progressing to large black necrotic lesions. Reduces yield by up to 50%.",
        "pathogen": "Mycosphaerella fijiensis",
        "crops": ["banana"],
        "variants": ["Black Sigatoka", "BLACK SIGATOKA", "black leaf streak disease",
                     "BLSD", "Sigatoka disease", "Mycosphaerella leaf spot"],
    },

    # ── Coffee diseases ──────────────────────────────────────────────────────
    "coffee leaf rust": {
        "context": "Fungal disease of coffee caused by Hemileia vastatrix. Orange-yellow powdery pustules on lower leaf surface. Historically devastating — destroyed Sri Lanka's coffee industry in the 1870s.",
        "pathogen": "Hemileia vastatrix",
        "crops": ["coffee"],
        "variants": ["Coffee Leaf Rust", "COFFEE RUST", "coffee rust",
                     "CLR", "Hemileia rust", "roya del cafeto"],
    },
}


def parse_plantvillage_folder_name(folder_name: str):
    """
    WHAT THIS DOES:
      PlantVillage Kaggle dataset has folders like:
        "Tomato___Late_blight"
        "Apple___Apple_scab"
        "Corn_(maize)___Common_rust_"
        "Tomato___healthy"

      We parse each folder to extract (crop_name, disease_name).
      We skip "healthy" folders — no disease to extract.

    RETURNS:
      (crop_name, disease_name) or None if it's a healthy/unusable folder.
    """
    # Replace underscores with spaces, split on triple underscore
    parts = folder_name.split("___")
    if len(parts) != 2:
        return None

    crop_raw, disease_raw = parts
    # Clean up crop name: "Corn_(maize)" → "Corn (maize)"
    crop = re.sub(r"[_]+", " ", crop_raw).strip()
    disease = re.sub(r"[_]+", " ", disease_raw).strip().strip("_").strip()

    # Skip healthy plants
    if "healthy" in disease.lower():
        return None

    return crop, disease


def get_disease_context(disease_name: str, crop: str) -> tuple:
    """
    WHAT THIS DOES:
      Looks up a disease in DISEASE_REFERENCE to get its description.
      Tries different search strategies:
        1. Exact canonical match
        2. Crop + disease combination match
        3. Disease name keyword match
        4. Falls back to a generated description

    RETURNS:
      (context_string, pathogen_string) — both might be empty string
    """
    disease_lower = disease_name.lower()
    crop_lower = crop.lower()

    # Strategy 1: Try "crop + disease" canonical
    combo = f"{crop_lower} {disease_lower}"
    if combo in DISEASE_REFERENCE:
        d = DISEASE_REFERENCE[combo]
        return d["context"], d.get("pathogen", "")

    # Strategy 2: Try disease name alone
    if disease_lower in DISEASE_REFERENCE:
        d = DISEASE_REFERENCE[disease_lower]
        return d["context"], d.get("pathogen", "")

    # Strategy 3: Check if disease name is mentioned in any reference key
    for key, d in DISEASE_REFERENCE.items():
        if disease_lower in key or key in disease_lower:
            return d["context"], d.get("pathogen", "")
        if any(disease_lower in v.lower() for v in d.get("variants", [])):
            return d["context"], d.get("pathogen", "")

    # Strategy 4: Generate a minimal context
    fallback = f"{disease_name} is a disease affecting {crop} plants."
    return fallback, ""


def scrape_from_kaggle_folders() -> list:
    """
    WHAT THIS DOES:
      Reads PlantVillage folder names from the Kaggle dataset directory.
      Returns entity records — one per unique (crop, disease) combination.
    """
    records = []

    if not PLANTVILLAGE_PATH.exists():
        print(f"  ⚠ PlantVillage folder not found at: {PLANTVILLAGE_PATH}")
        print(f"  → Falling back to built-in reference only")
        return []

    folder_names = [f.name for f in PLANTVILLAGE_PATH.iterdir() if f.is_dir()]
    print(f"  Found {len(folder_names)} folders in PlantVillage dataset")

    seen = set()
    for folder in sorted(folder_names):
        parsed = parse_plantvillage_folder_name(folder)
        if parsed is None:
            continue

        crop, disease = parsed
        key = (crop.lower(), disease.lower())
        if key in seen:
            continue
        seen.add(key)

        context, pathogen = get_disease_context(disease, crop)
        full_name = f"{disease}"  # Primary surface form

        record = {
            "entity_id":   f"PV_{len(records)+1:04d}",
            "name":        full_name,
            "canonical":   full_name.lower().strip(),
            "entity_type": "Disease",
            "context":     context[:300],
            "source":      SOURCE_NAME,
            "source_url":  SOURCE_URL_TEMPLATE,
            "crop":        crop,
            "pathogen":    pathogen,
            "folder_name": folder,
        }
        records.append(record)

    return records


def build_from_builtin_reference() -> list:
    """
    WHAT THIS DOES:
      Uses the DISEASE_REFERENCE dictionary to generate entity records.
      This works without downloading Kaggle data.
    """
    records = []
    for idx, (canonical, info) in enumerate(DISEASE_REFERENCE.items()):
        record = {
            "entity_id":   f"PV_{idx+1:04d}",
            "name":        canonical.title(),   # "late blight" → "Late Blight"
            "canonical":   canonical,
            "entity_type": "Disease",
            "context":     info["context"][:300],
            "source":      SOURCE_NAME,
            "source_url":  SOURCE_URL_TEMPLATE,
            "crop":        ", ".join(info.get("crops", [])),
            "pathogen":    info.get("pathogen", ""),
            "folder_name": "",
        }
        records.append(record)

        # Also add each variant as a separate record
        for variant in info.get("variants", []):
            variant_record = record.copy()
            variant_record["entity_id"]  = f"PV_{idx+1:04d}_V{len(records):03d}"
            variant_record["name"]       = variant
            variant_record["canonical"]  = variant.lower().strip()
            records.append(variant_record)

    return records


def build_plantvillage_pairs(records: list) -> list:
    """
    WHAT THIS DOES:
      From the entity records, creates positive training pairs.
      Two entities are a positive pair if they have the same canonical disease.

    HOW:
      We group all records by their base canonical name.
      All names within the same group become pairs with label=1.
    """
    # Group by canonical base disease (from DISEASE_REFERENCE keys)
    groups = {}  # canonical → list of surface forms

    for canonical, info in DISEASE_REFERENCE.items():
        all_names = [canonical.title()] + info.get("variants", [])
        groups[canonical] = all_names

    pairs = []
    pair_id = 1
    for canonical, names in groups.items():
        for name_1, name_2 in itertools.combinations(names, 2):
            pairs.append({
                "pair_id":     f"PV_POS_{pair_id:04d}",
                "name_1":      name_1,
                "name_2":      name_2,
                "canonical_1": name_1.lower().strip(),
                "canonical_2": name_2.lower().strip(),
                "entity_type": "Disease",
                "label":       1,
                "pair_source": "plantvillage_reference",
                "confidence":  1.0,
                "note":        f"Both refer to: '{canonical}'",
            })
            pair_id += 1

    return pairs


def main():
    print("\n" + "═"*60)
    print("  SCRIPT 01 — PlantVillage Disease Extractor")
    print("  AgriΛNet Entity Resolution Pipeline")
    print("═"*60)

    # ── Extract entities ───────────────────────────────────────────────────
    print("\n[1/3] Extracting disease entities...")
    if USE_BUILTIN_ONLY:
        print("  Mode: Built-in reference (Kaggle dataset not required)")
        records = build_from_builtin_reference()
    else:
        print("  Mode: Kaggle dataset folders")
        records = scrape_from_kaggle_folders()
        if not records:
            print("  Falling back to built-in reference...")
            records = build_from_builtin_reference()

    df = save_raw(records, "plantvillage_raw.csv")

    print(f"\n  Diseases covered: {df['canonical'].nunique()}")
    print(f"  Total records (incl. variants): {len(df)}")

    # ── Build pairs ────────────────────────────────────────────────────────
    print("\n[2/3] Building positive training pairs...")
    pairs = build_plantvillage_pairs(records)
    pairs_df = pd.DataFrame(pairs)
    out_path = PAIRS_DIR / "plantvillage_pairs_positive.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(pairs_df)} positive pairs → {out_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[3/3] Summary")
    print(f"  ┌────────────────────────────────────────┐")
    print(f"  │ Disease entries (with variants): {len(df):5d} │")
    print(f"  │ Positive pairs generated:        {len(pairs_df):5d} │")
    print(f"  └────────────────────────────────────────┘")
    print("\n  ✅ Script 01 complete! Next: run 02_scrape_agrovoc.py\n")

    return df, pairs_df


if __name__ == "__main__":
    main()