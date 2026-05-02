"""
augment_wave2_synthetic.py
==========================
Generates Wave 2 synthetic Type C (hard-negative / polysemy) pairs.

KEY DESIGN DECISION:
    lambda_val  = NaN  (LLM will assign after data extraction)
    llm_match   = NaN  (LLM will assign after data extraction)
    lambda_source = "pending_llm"

These pairs are hand-crafted cross-crop polysemy cases:
  - Same symptom word (rust, blight, wilt...) but DIFFERENT crops/pathogens
  - Rich, specific contexts so the LLM can make an informed lambda assignment

Input : dataset_augmented.csv
Output: dataset_augmented.csv  (overwritten with wave2 rows appended)
"""

import pandas as pd
import numpy as np
import hashlib
import sys

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INPUT_CSV  = "../data/dataset_augmented.csv"
OUTPUT_CSV = "../data/dataset_augmented.csv"

# ---------------------------------------------------------------------------
# SYNTHETIC PAIRS  (match=0, pair_type=C, lambda_val=NaN, llm_match=NaN)
# Each entry: (name_a, context_a, canonical_id_a, name_b, context_b, canonical_id_b)
# ---------------------------------------------------------------------------
SYNTHETIC_PAIRS = [
    # ── RUST cross-crop ──────────────────────────────────────────────────────
    ("coffee leaf rust",
     "Coffee leaf rust (Hemileia vastatrix) is a fungal disease of coffee plants producing orange-yellow powdery pustules on the lower leaf surface. It historically devastated Sri Lanka's coffee industry in the 1870s and remains the most economically damaging coffee disease globally.",
     "coffee_leaf_rust",
     "wheat leaf rust",
     "Wheat leaf rust (Puccinia triticina) is the most widespread rust disease of wheat worldwide, causing orange-brown circular pustules scattered on the leaf surface. It can reduce grain yield by 10-20% in susceptible varieties.",
     "wheat_leaf_rust"),

    ("soybean rust",
     "Asian soybean rust (Phakopsora pachyrhizi) is a fungal disease that spreads rapidly under warm, humid conditions, producing tan to brown lesions with spore pustules on leaf undersides. It can cause up to 80% yield loss in susceptible soybean crops.",
     "asian_soybean_rust",
     "coffee leaf rust",
     "Coffee leaf rust (Hemileia vastatrix) is a fungal disease of coffee plants producing orange-yellow powdery pustules on the lower leaf surface. It historically devastated Sri Lanka's coffee industry in the 1870s and remains the most economically damaging coffee disease globally.",
     "coffee_leaf_rust"),

    ("corn common rust",
     "Common rust of maize (Puccinia sorghi) produces small, circular to oval, brick-red pustules on both leaf surfaces. It is widespread in temperate maize-growing regions but rarely causes severe losses in most commercial hybrids.",
     "common_corn_rust",
     "soybean rust",
     "Asian soybean rust (Phakopsora pachyrhizi) is a fungal disease that spreads rapidly under warm, humid conditions, producing tan to brown lesions on leaf undersides. It can cause up to 80% yield loss.",
     "asian_soybean_rust"),

    ("wheat stem rust",
     "Stem rust of wheat (Puccinia graminis f. sp. tritici) produces reddish-brown, brick-red pustules mainly on stems and leaf sheaths of wheat. Race Ug99 is a highly virulent strain that threatens global wheat production.",
     "wheat_stem_rust",
     "corn common rust",
     "Common rust of maize (Puccinia sorghi) produces small, circular to oval, brick-red pustules on both leaf surfaces of maize. It is caused by a different Puccinia species than wheat rust and does not infect wheat.",
     "common_corn_rust"),

    ("cedar apple rust",
     "Cedar-apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease requiring two host plants: eastern red cedar/juniper and apple. It causes bright yellow-orange spots on apple leaves and fruit in spring.",
     "cedar_apple_rust",
     "wheat leaf rust",
     "Wheat leaf rust (Puccinia triticina) produces orange-brown pustules on wheat leaves. It is caused by a biotrophic fungus that only infects wheat and related grasses, not apple or cedar trees.",
     "wheat_leaf_rust"),

    ("stripe rust of wheat",
     "Stripe rust (Puccinia striiformis f. sp. tritici) produces yellow-orange pustules arranged in stripes along wheat leaf veins. It thrives in cool, humid conditions and is a major constraint to wheat production in highland areas.",
     "wheat_stripe_rust",
     "corn common rust",
     "Common rust of maize (Puccinia sorghi) produces scattered brick-red pustules on both surfaces of maize leaves. Unlike stripe rust, pustules are not arranged in stripes and the pathogen only infects maize.",
     "common_corn_rust"),

    # ── BLIGHT cross-crop ────────────────────────────────────────────────────
    ("tomato late blight",
     "Late blight of tomato (Phytophthora infestans) causes rapidly expanding, dark, water-soaked lesions on leaves and fruit. The same oomycete pathogen caused the Irish Potato Famine. It spreads explosively in cool, wet weather.",
     "late_blight",
     "northern corn leaf blight",
     "Northern corn leaf blight (Exserohilum turcicum) causes long, cigar-shaped, greyish-green lesions on maize leaves. It is caused by a fungus, not an oomycete, and only infects maize and related grasses, not tomato or potato.",
     "northern_corn_leaf_blight"),

    ("potato late blight",
     "Late blight of potato (Phytophthora infestans) causes dark, water-soaked lesions on leaves and tuber rot. It is the same oomycete pathogen responsible for the 1840s Irish Famine. Chemical control is difficult due to fungicide resistance.",
     "late_blight",
     "early blight of tomato",
     "Early blight of tomato (Alternaria solani) produces dark brown lesions with concentric rings giving a target-board appearance, starting on lower leaves. It is caused by a fungus (Alternaria) rather than an oomycete, and has a slower progression than late blight.",
     "early_blight"),

    ("rice blast",
     "Rice blast (Magnaporthe oryzae) is the most devastating rice disease worldwide, producing spindle-shaped, grey lesions with brown borders on leaves, nodes and panicles. It can destroy entire rice crops under warm, humid conditions.",
     "rice_blast",
     "northern corn leaf blight",
     "Northern corn leaf blight (Exserohilum turcicum) causes long cigar-shaped grey-green lesions on maize leaves. It only infects maize and related grasses, not rice, and is caused by a different fungal genus than rice blast.",
     "northern_corn_leaf_blight"),

    ("wheat fusarium head blight",
     "Fusarium head blight (Fusarium graminearum) causes premature bleaching of wheat heads and contaminates grain with mycotoxins including deoxynivalenol (DON). It is favoured by warm, humid weather during anthesis.",
     "wheat_head_blight",
     "potato late blight",
     "Late blight of potato (Phytophthora infestans) causes dark water-soaked lesions on potato leaves and tuber rot. Unlike Fusarium head blight, it is caused by an oomycete and does not produce mycotoxins.",
     "late_blight"),

    ("citrus canker",
     "Citrus canker (Xanthomonas citri subsp. citri) is a bacterial disease causing raised, corky lesions on leaves, stems and fruit of citrus trees. It reduces fruit quality and can lead to defoliation. There is no cure once a tree is infected.",
     "citrus_canker",
     "bacterial blight of rice",
     "Bacterial leaf blight of rice (Xanthomonas oryzae pv. oryzae) causes wilting and yellowing of rice leaves starting from the tips. Despite both being Xanthomonas diseases, citrus canker and rice bacterial blight affect entirely different host plants.",
     "rice_bacterial_blight"),

    # ── WILT cross-crop ──────────────────────────────────────────────────────
    ("banana fusarium wilt",
     "Panama disease (Fusarium oxysporum f. sp. cubense) is a vascular wilt disease of banana that blocks water transport, causing yellowing and wilting. Race TR4 threatens the global Cavendish banana supply and has no effective chemical cure.",
     "banana_fusarium_wilt",
     "verticillium wilt",
     "Verticillium wilt (Verticillium dahliae and V. albo-atrum) affects over 350 plant species including tomato, cotton and strawberry. It causes progressive yellowing and wilting from the bottom of the plant, leaving characteristic dark streaks in the vascular tissue.",
     "verticillium_wilt"),

    ("fusarium wilt of tomato",
     "Fusarium wilt of tomato (Fusarium oxysporum f. sp. lycopersici) is a soil-borne vascular disease causing yellowing of lower leaves, brown discoloration of the stem interior, and eventual plant death. It is host-specific and does not infect banana.",
     "fusarium_wilt_tomato",
     "banana fusarium wilt",
     "Panama disease (Fusarium oxysporum f. sp. cubense) causes vascular wilt exclusively in banana plants. The pathogen is a different forma specialis from the one that infects tomato, and the two diseases affect entirely different crop hosts.",
     "banana_fusarium_wilt"),

    # ── SPOT / LEAF SPOT cross-crop ──────────────────────────────────────────
    ("tomato septoria leaf spot",
     "Septoria leaf spot (Septoria lycopersici) produces numerous small circular spots with dark borders and light grey centres on tomato leaves. It spreads rapidly by water splash and can cause severe defoliation, reducing yield and fruit quality.",
     "tomato_septoria_leaf_spot",
     "bacterial spot of pepper",
     "Bacterial spot of pepper (Xanthomonas euvesicatoria) produces water-soaked, irregular spots on pepper leaves and fruit. Unlike Septoria leaf spot, it is caused by a bacterium rather than a fungus and can affect fruit as well as foliage.",
     "pepper_bacterial_spot"),

    ("grey leaf spot of maize",
     "Grey leaf spot of maize (Cercospora zeae-maydis) produces rectangular, tan to grey lesions with distinct parallel margins running along leaf veins. It is favoured by warm, humid conditions with heavy dew and is a major yield-reducing disease of maize.",
     "grey_leaf_spot_maize",
     "tomato septoria leaf spot",
     "Septoria leaf spot (Septoria lycopersici) causes small circular spots with grey centres on tomato leaves. Unlike grey leaf spot of maize, the lesions are circular rather than rectangular and the pathogen only infects tomato and related Solanaceae.",
     "tomato_septoria_leaf_spot"),

    ("rice brown spot",
     "Rice brown spot (Cochliobolus miyabeanus) causes oval to circular brown lesions with grey or whitish centres on rice leaves. It is associated with nutritional stress, particularly silicon and potassium deficiency, and can reduce grain quality.",
     "rice_brown_spot",
     "early blight of tomato",
     "Early blight of tomato (Alternaria solani) produces dark brown concentric ring lesions with a target-board appearance on tomato leaves. It is caused by a different fungal genus than rice brown spot and only infects Solanaceous plants.",
     "early_blight"),

    # ── MILDEW cross-crop ────────────────────────────────────────────────────
    ("grape powdery mildew",
     "Grape powdery mildew (Erysiphe necator, formerly Uncinula necator) produces white powdery colonies on young leaves, shoots and berries of grapevine. It is an obligate fungal parasite favoured by warm, dry weather with high humidity at night.",
     "grape_powdery_mildew",
     "wheat powdery mildew",
     "Wheat powdery mildew (Blumeria graminis f. sp. tritici) produces white powdery colonies on wheat leaves and stems, reducing photosynthesis. It is caused by a completely different fungal species than grape powdery mildew and cannot cross-infect between crops.",
     "wheat_powdery_mildew"),

    ("squash powdery mildew",
     "Squash powdery mildew (Podosphaera xanthii) causes white powdery growth on cucurbit leaves, reducing photosynthesis and fruit quality. It is an obligate fungal parasite specific to cucurbits such as squash, cucumber and melon.",
     "squash_powdery_mildew",
     "grape powdery mildew",
     "Grape powdery mildew (Erysiphe necator) infects only grapevine tissues. It is caused by a different Erysiphales species than squash powdery mildew and does not infect cucurbits.",
     "grape_powdery_mildew"),

    ("cherry powdery mildew",
     "Cherry powdery mildew (Podosphaera clandestina) produces white powdery patches on young cherry leaves and shoots, distorting leaf shape. It is host-specific to Prunus species and does not infect grapevine or cucurbits.",
     "cherry_powdery_mildew",
     "squash powdery mildew",
     "Squash powdery mildew (Podosphaera xanthii) infects cucurbit crops including squash, cucumber and melon. It is caused by a different fungal species from cherry powdery mildew and cannot infect stone fruit trees.",
     "squash_powdery_mildew"),

    # ── MOSAIC VIRUS cross-crop ──────────────────────────────────────────────
    ("soybean mosaic virus",
     "Soybean mosaic virus (SMV) is a potyvirus that infects soybean plants, causing mosaic, chlorosis and leaf distortion. It is aphid-transmitted and can severely reduce seed quality and yield in susceptible soybean cultivars.",
     "soybean_mosaic_virus",
     "tomato mosaic virus",
     "Tomato mosaic virus (ToMV) is a tobamovirus that infects tomato and other Solanaceous plants, causing a mottled light and dark green mosaic pattern. Unlike SMV, ToMV is mechanically transmitted and extremely persistent in plant debris.",
     "tomato_mosaic_virus"),

    ("wheat streak mosaic virus",
     "Wheat streak mosaic virus (WSMV) is a tritimovirus vectored by the wheat curl mite (Aceria tosichella). It causes yellow streaking and mosaic on wheat leaves and can cause severe yield loss in the Great Plains region.",
     "wheat_streak_mosaic_virus",
     "maize streak virus",
     "Maize streak virus (MSV) is a mastrevirus in the family Geminiviridae, transmitted by leafhoppers. It causes yellow streaking on maize leaves and is endemic to sub-Saharan Africa. MSV and WSMV are in different virus families and infect different host crops.",
     "maize_streak_virus"),

    # ── SCAB cross-crop ──────────────────────────────────────────────────────
    ("apple scab",
     "Apple scab (Venturia inaequalis) causes olive-green to black velvety lesions on apple leaves and fruit surface. It is the most economically important disease of apple worldwide, requiring numerous fungicide applications each season.",
     "apple_scab",
     "wheat fusarium head blight",
     "Fusarium head blight (Fusarium graminearum) causes bleached, sterile wheat spikelets and mycotoxin contamination of grain. Despite the common use of the word 'scab' as an alternative name for FHB, it affects wheat heads and is caused by a completely different pathogen than apple scab.",
     "wheat_head_blight"),

    ("common scab of potato",
     "Common scab of potato (Streptomyces scabiei) causes corky, brown, raised or pitted lesions on potato tuber skin. It is caused by a soil-borne actinobacterium, not a fungus, and affects the tuber surface without causing internal rot.",
     "common_scab_potato",
     "apple scab",
     "Apple scab (Venturia inaequalis) is a fungal disease causing velvety lesions on apple leaves and fruit. It is caused by an ascomycete fungus, not a bacterium, and infects apple trees rather than root crops.",
     "apple_scab"),

    # ── ROT cross-crop ───────────────────────────────────────────────────────
    ("grape black rot",
     "Grape black rot (Guignardia bidwellii) causes small yellow-green spots on leaves that turn brown, and shrivels fruit into hard black mummies. It is a major disease in humid, warm grape-growing regions of eastern North America.",
     "grape_black_rot",
     "apple black rot",
     "Apple black rot (Botryosphaeria obtusa) causes circular brown lesions on apple fruit and frog-eye leaf spots. It is caused by a different fungal genus than grape black rot and infects apple trees rather than grapevine.",
     "apple_black_rot"),

    ("rice neck rot",
     "Neck rot of rice is caused by Magnaporthe oryzae infecting the neck node of the panicle, cutting off nutrient flow and causing the entire panicle to turn white and unfilled. It is part of the rice blast disease complex.",
     "rice_blast",
     "grape black rot",
     "Grape black rot (Guignardia bidwellii) shrivels grape berries into hard black mummies and causes leaf spots on grapevine. Despite both being called rot diseases, neck rot of rice and grape black rot are caused by entirely different pathogens and affect different host plants.",
     "grape_black_rot"),

    # ── SMUT ─────────────────────────────────────────────────────────────────
    ("corn smut",
     "Corn smut (Mycosarcoma maydis) is a fungal disease forming large, grey-white galls on all above-ground parts of maize. The infected galls (huitlacoche) are edible and considered a delicacy in Mexico.",
     "corn_smut",
     "wheat loose smut",
     "Wheat loose smut (Ustilago tritici) destroys the entire wheat head, replacing the grain with masses of dark olive-brown spores that are dispersed by wind. Unlike corn smut, the galls are not edible and the disease is seed-borne.",
     "wheat_loose_smut"),

    # ── CANKER / BACTERIAL ───────────────────────────────────────────────────
    ("fire blight",
     "Fire blight (Erwinia amylovora) is a bacterial disease of apple, pear and other Rosaceae that causes shoots to turn brown and wilt in a characteristic shepherd's crook pattern. Under optimal conditions it can destroy an entire orchard in a single growing season.",
     "fire_blight",
     "citrus canker",
     "Citrus canker (Xanthomonas citri subsp. citri) causes raised corky lesions on citrus leaves, stems and fruit. Unlike fire blight, it does not cause wilting but rather reduces fruit quality and marketability. The two diseases affect completely different plant families.",
     "citrus_canker"),

    ("bacterial spot of tomato",
     "Bacterial spot of tomato (Xanthomonas species) produces small, water-soaked spots on tomato leaves and fruit that turn dark and raised. It is favoured by warm, wet conditions and can cause significant defoliation and fruit blemishing.",
     "tomato_bacterial_spot",
     "fire blight",
     "Fire blight (Erwinia amylovora) infects apple, pear and other Rosaceae with a rapidly spreading bacterial infection causing shoot die-back and the characteristic shepherd's crook wilting. It does not infect tomato or other Solanaceous plants.",
     "fire_blight"),

    # ── VIRUS CROSS ──────────────────────────────────────────────────────────
    ("tomato yellow leaf curl virus",
     "Tomato yellow leaf curl virus (TYLCV) is a begomovirus transmitted by the whitefly Bemisia tabaci. It causes severe yellowing, upward curling of leaves, and stunting of tomato plants, leading to near-total yield loss in heavily infected plants.",
     "tomato_yellow_leaf_curl",
     "banana bunchy top virus",
     "Banana bunchy top virus (BBTV) is transmitted by the banana aphid Pentalonia nigronervosa. It causes extreme stunting with stiff upright leaves forming a bunchy top appearance. Despite both being transmitted by insects and causing leaf curl/bunching symptoms, they are in different virus families and infect different crops.",
     "banana_bunchy_top"),

    ("potato virus Y",
     "Potato virus Y (PVY) is the most economically important potyvirus of potato, causing mosaic, necrosis, and significant tuber yield reduction. Strains vary in virulence from mild mosaic to severe necrotic ringspot on tubers.",
     "potato_virus_y",
     "tomato yellow leaf curl virus",
     "Tomato yellow leaf curl virus (TYLCV) is a begomovirus causing yellowing and curling of tomato leaves transmitted by whitefly. Despite both being important potato/tomato viruses, PVY is a potyvirus transmitted by aphids while TYLCV is a begomovirus transmitted by whitefly.",
     "tomato_yellow_leaf_curl"),

    # ── DOWNY MILDEW ─────────────────────────────────────────────────────────
    ("grape downy mildew",
     "Grape downy mildew (Plasmopara viticola) causes yellow oily spots on upper leaf surfaces and white downy sporulation on lower surfaces. It is an oomycete disease that can devastate vineyards under cool, wet conditions in spring.",
     "grape_downy_mildew",
     "squash powdery mildew",
     "Squash powdery mildew (Podosphaera xanthii) causes white powdery growth on cucurbit leaf surfaces and is favoured by warm, dry weather. Unlike downy mildew, it is a true fungus and produces superficial, not intercellular, mycelium.",
     "squash_powdery_mildew"),

    ("downy mildew of cucurbits",
     "Cucurbit downy mildew (Pseudoperonospora cubensis) causes angular yellow lesions on cucumber and melon leaves with purplish sporulation on the lower surface. It spreads rapidly under moist conditions and can cause complete defoliation.",
     "cucurbit_downy_mildew",
     "grape downy mildew",
     "Grape downy mildew (Plasmopara viticola) infects grapevine tissues including leaves, shoots and berries. It is caused by a different oomycete species from cucurbit downy mildew and cannot infect cucurbit plants.",
     "grape_downy_mildew"),

    # ── CLUBROOT / DAMPING OFF ───────────────────────────────────────────────
    ("clubroot",
     "Clubroot (Plasmodiophora brassicae) causes swollen, distorted club-shaped roots in brassica crops including cabbage, canola and broccoli. It is a soil-borne protist that can persist in soil for over 20 years.",
     "clubroot",
     "damping off",
     "Damping-off is caused by multiple soil-borne pathogens including Pythium, Rhizoctonia and Fusarium species, killing seedlings before or shortly after emergence. Unlike clubroot, it affects seedlings of many plant families and is not caused by a single specific pathogen.",
     "damping_off"),
]


def pair_id(a, b):
    key = "_".join(sorted([str(a).lower().strip(), str(b).lower().strip()]))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def main():
    print("=" * 60)
    print(" Wave 2 Synthetic Type C Augmentation")
    print(" lambda_val = NaN  (LLM will assign later)")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} existing rows")

    # Build existing pair key index
    existing_keys = set()
    for _, row in df.iterrows():
        existing_keys.add(pair_id(str(row["name_a"]), str(row["name_b"])))

    cols = list(df.columns)
    new_rows = []
    skipped = 0

    for entry in SYNTHETIC_PAIRS:
        na, ctx_a, can_a, nb, ctx_b, can_b = entry
        key = pair_id(na, nb)
        if key in existing_keys:
            print(f"  SKIP (exists): {na!r} vs {nb!r}")
            skipped += 1
            continue

        row = {col: None for col in cols}
        row["name_a"]              = na
        row["context_a"]           = ctx_a
        row["canonical_id_a"]      = can_a
        row["name_b"]              = nb
        row["context_b"]           = ctx_b
        row["canonical_id_b"]      = can_b
        row["match"]               = 0
        row["llm_match"]           = np.nan      # LLM fills later
        row["lambda_val"]          = np.nan      # LLM fills later
        row["lambda_source"]       = "pending_llm"
        row["pair_type"]           = "C"
        row["exclude_from_lambda"] = False

        # Compute name_sim (Jaccard)
        import re
        STOP = {"of","the","a","an","and","or","in","on","by","from","with",
                "to","for","as","at","is","are","was","be","caused","disease",
                "infection","plant","crop"}
        def tok(s):
            ts = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
            return {t for t in ts if len(t) > 2 and t not in STOP}
        sa, sb = tok(na), tok(nb)
        if sa or sb:
            sim = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        else:
            sim = 0.0
        row["name_sim_score"] = round(sim, 4)

        if "context_quality_a" in cols:
            row["context_quality_a"] = "good"
        if "context_quality_b" in cols:
            row["context_quality_b"] = "good"
        if "pair_type_reason" in cols:
            row["pair_type_reason"] = "no-match + similar names (polysemy) — synthetic wave2"

        new_rows.append(row)
        existing_keys.add(key)

    print(f"\nNew pairs added : {len(new_rows)}")
    print(f"Skipped (exist) : {skipped}")

    if new_rows:
        new_df    = pd.DataFrame(new_rows, columns=cols)
        augmented = pd.concat([df, new_df], ignore_index=True)
        augmented.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved -> {OUTPUT_CSV}  ({len(augmented)} total rows)")

        # Summary
        pt = augmented["pair_type"].value_counts()
        pending = (augmented["lambda_source"] == "pending_llm").sum()
        print("\n-- Pair type counts ---")
        for t in ["A","B","C","D"]:
            print(f"  Type {t}: {pt.get(t, 0)}")
        print(f"\n  Rows pending LLM lambda: {pending}")
        print(f"  (run your LLM labeler on rows where lambda_source == 'pending_llm')")
    else:
        print("No new rows to add.")

    print("\nDone.")


if __name__ == "__main__":
    main()
