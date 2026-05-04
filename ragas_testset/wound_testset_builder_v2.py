"""
wound_testset_builder_v2.py
===========================
Generates ragas_testset/wound_testset_v2.json

KEY FIX vs v2
─────────────
v2 had hand-written CTX strings that did NOT match actual ai_summary text in
ChromaDB chunks — silently corrupting all four RAGAS metrics.

v3 loads CTX strings DIRECTLY from the chunk JSON files at build time, using
the exact chunk_id for each source. This ensures reference_contexts == the
real page_content strings stored in db_wound_care_v3, so:
  - LLMContextRecall   measures whether the answer is entailed by REAL chunks
  - LLMContextPrecision measures whether RETRIEVED chunks match REAL references
  - Faithfulness / AnswerRelevancy score against a fair, grounded baseline

CHUNK ID MAP (verified against actual JSON files)
─────────────────────────────────────────────────
GP source  (GP_wound_dressings_kept.json):
  GP_algo      → bd2bb8e1321e  (Decision Algorithm)
  GP_type1     → 52ef696853c7  (Wound Type 1)
  GP_type2     → 4643f10b8894  (Wound Type 2)
  GP_type3     → c0a350e36ecf  (Wound Type 3)
  GP_type4     → d622ee9f4c9c  (Wound Type 4)
  GP_type5     → aad7a40107b0  (Wound Type 5)
  GP_type6     → b4ba04cb08d4  (Wound Type 6)
  GP_type7     → c4177e98524e  (Wound Type 7)
  GP_type8     → e75347f9bdb3  (Wound Type 8)
  GP_referral  → ca7a1e934891  (Hospital Referral)

WCM source (WCM_wound_care_manual_kept.json):
  WCM_film        → 2de03f803f2f  (Ch14 Film)
  WCM_hydrogel    → d81176511903  (Ch14 Hydrogel)
  WCM_hydrocolloid→ f8cb463d04cf  (Ch14 Hydrocolloid)
  WCM_alginate    → c540b3e5c067  (Ch14 Calcium Alginate)
  WCM_foam        → 77e6e32d188a  (Ch14 Foams)
  WCM_hydrofibre  → e63bd0378895  (Ch14 Hydrofibre)
  WCM_charcoal    → 861a57a2172c  (Ch14 Charcoal)
  WCM_silver      → e8c86c4e1aa6  (Ch14 Silver)
  WCM_polymeric   → 6fd9e2433cc9  (Ch14 Polymeric Membrane)
  WCM_npwt        → 05cc6ca1ddfc  (Ch16c NPWT)
  WCM_honey       → b480aa73a9c2  (Ch16a Honey)
  WCM_debride     → b5b5a6c9dcf2  (Ch15 Debridement)

AJGP source (AJGP_wound_dressings_kept.json):
  AJGP_skintear   → 55569f16010f  (Table 1 Skin tears)
  AJGP_postop     → 6555b7728ccc  (Table 1 Postoperative)
  AJGP_burns      → 5116098922da  (Table 1 Burns)
  AJGP_diabfoot   → 0d0a9fc09c73  (Table 1 Diabetic foot)
  AJGP_principles → 2dc1f26b6233  (General Principles)

SFP source (SFP_wound_dressings_kept.json):
  SFP_iodine      → 3082e1a296e7  (Antimicrobial – iodine/silver/honey table)
                    NOTE: this is the chunk with the thyroid contraindication.
                    f05456b64b8d (Cadexomer Iodine table) is also included
                    for cases referencing cadexomer specifically.
  SFP_foam        → 254dd74d7f00  (Table 2 Foams)
  SFP_alginate    → 3b666ccfba99  (Table 2 Alginates)
  SFP_hydrofiber  → 4b75fefc0517  (Table 2 Hydrofiber)
  SFP_silver      → 765445bd6358  (Table 2 Silver Barrier)
  SFP_hydrogel    → ad036dc35955  (Table 2 Hydrogels)
  SFP_film        → 9e661711e520  (Table 2 Films/Membranes)
  SFP_hydrocolloid→ b4c13d77818b  (Table 2 Hydrocolloids)

CATEGORY BREAKDOWN (28 cases)
──────────────────────────────
  Cat A — 8 core wound-type cases  (Types 1–8)
  Cat B — 10 contraindication / safety / special-population cases
  Cat C —  6 dressing selection & reasoning cases
  Cat D —  4 clinical notes override cases
"""

import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load all chunk ai_summary strings from the real JSON files
# ─────────────────────────────────────────────────────────────────────────────
# Adjust CHUNK_DIR to wherever your ingestion_output_ai/ folder lives.
# The builder will error clearly if a file is missing.

CHUNK_DIR = os.environ.get(
    "WOUND_CHUNK_DIR",
    "ingestion_output_ai"        # ← change this if your path differs
)

_CHUNK_FILES = {
    "GP":   "GP_wound_dressings_kept.json",
    "WCM":  "WCM_wound_care_manual_kept.json",
    "AJGP": "AJGP_wound_dressings_kept.json",
    "SFP":  "SFP_wound_dressings_kept.json",
}

def _load_all_chunks(chunk_dir: str) -> dict:
    """Return {chunk_id: ai_summary} for every chunk across all four sources."""
    all_chunks: dict[str, str] = {}
    for src, fname in _CHUNK_FILES.items():
        fpath = os.path.join(chunk_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Chunk file not found: {fpath}\n"
                f"Set WOUND_CHUNK_DIR env var to the folder containing the four kept JSON files."
            )
        raw = json.load(open(fpath, encoding="utf-8"))
        chunks = raw if isinstance(raw, list) else raw.get("kept_chunks", [])
        for c in chunks:
            all_chunks[c["chunk_id"]] = c["ai_summary"]
    print(f"✅ Loaded {len(all_chunks)} chunks from {chunk_dir}")
    return all_chunks


_CHUNKS = _load_all_chunks(CHUNK_DIR)


def ctx(chunk_id: str) -> str:
    """
    Retrieve the exact ai_summary for a chunk_id.
    Raises KeyError with a clear message if the ID is wrong.
    """
    if chunk_id not in _CHUNKS:
        raise KeyError(
            f"chunk_id '{chunk_id}' not found in loaded chunks. "
            f"Check the CHUNK ID MAP at the top of this file."
        )
    return _CHUNKS[chunk_id]


# ─────────────────────────────────────────────────────────────────────────────
# Chunk ID constants  (copy-paste safe — verified against actual JSON files)
# ─────────────────────────────────────────────────────────────────────────────
GP_ALGO       = "bd2bb8e1321e"
GP_TYPE1      = "52ef696853c7"
GP_TYPE2      = "4643f10b8894"
GP_TYPE3      = "c0a350e36ecf"
GP_TYPE4      = "d622ee9f4c9c"
GP_TYPE5      = "aad7a40107b0"
GP_TYPE6      = "b4ba04cb08d4"
GP_TYPE7      = "c4177e98524e"
GP_TYPE8      = "e75347f9bdb3"
GP_REFERRAL   = "ca7a1e934891"

WCM_FILM        = "2de03f803f2f"
WCM_HYDROGEL    = "d81176511903"
WCM_HYDROCOLLOID= "f8cb463d04cf"
WCM_ALGINATE    = "c540b3e5c067"
WCM_FOAM        = "77e6e32d188a"
WCM_HYDROFIBRE  = "e63bd0378895"
WCM_CHARCOAL    = "861a57a2172c"
WCM_SILVER      = "e8c86c4e1aa6"
WCM_POLYMERIC   = "6fd9e2433cc9"
WCM_NPWT        = "05cc6ca1ddfc"
WCM_HONEY       = "b480aa73a9c2"
WCM_DEBRIDE     = "b5b5a6c9dcf2"

AJGP_SKINTEAR   = "55569f16010f"
AJGP_POSTOP     = "6555b7728ccc"
AJGP_BURNS      = "5116098922da"
AJGP_DIABFOOT   = "0d0a9fc09c73"
AJGP_PRINCIPLES = "2dc1f26b6233"

# SFP_iodine: use 3082e1a296e7 — the Antimicrobial Dressings chunk that contains
# the thyroid contraindication for iodine. f05456b64b8d (Cadexomer table) does NOT
# contain the thyroid warning — do not use it for iodine-contraindication cases.
SFP_IODINE      = "3082e1a296e7"
SFP_FOAM        = "254dd74d7f00"
SFP_ALGINATE    = "3b666ccfba99"
SFP_HYDROFIBER  = "4b75fefc0517"
SFP_SILVER      = "765445bd6358"
SFP_HYDROGEL    = "ad036dc35955"
SFP_FILM        = "9e661711e520"
SFP_HYDROCOLLOID= "b4c13d77818b"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def fmt_input(n, s, g, inf, moist, edge, notes=""):
    lines = [
        f"Necrotic: {n}%, Slough: {s}%, Granulation: {g}%",
        f"Infection: {inf}",
        f"Moisture: {moist}",
        f"Edge: {edge}",
    ]
    if notes:
        lines.append(f"Notes: {notes}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TESTSET
# reference_contexts uses ctx(CHUNK_ID) — exact ai_summary from ChromaDB chunks
# ─────────────────────────────────────────────────────────────────────────────
testset = [

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY A — Core wound-type cases (Types 1–8)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "case_id":   "cat_a_type1_dry",
        "category":  "A",
        "synthesizer_name": "cat_a_type1_dry",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Low", "edge": "Advancing", "notes": "",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Low", "Advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Film dressing or thin hydrocolloid over the granulating wound bed. "
            "Film is transparent (facilitates monitoring), waterproof, and changes every 2–5 days. "
            "Hydrocolloid provides a moist environment and promotes autolysis; also 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Not required for film. Foam may be added for cushioning/protection if wound is at risk "
            "of trauma, without using silver or charcoal.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 100% healthy granulation — wound bed is clean; no debridement needed.\n"
            "- I (Infection): Not infected — antimicrobial dressings (silver, charcoal) are "
            "CONTRAINDICATED on clean granulating wounds.\n"
            "- M (Moisture): Low/dry — film or hydrocolloid maintain moisture balance without excess absorption.\n"
            "- E (Edge): Advancing — wound is progressing; maintain current dressing strategy.\n\n"
            "## Contraindications\n"
            "Silver dressings and charcoal dressings must NOT be used — they are excluded from "
            "Wound Type 1 by the clinical algorithm. Special advanced dressing materials are also excluded.\n\n"
            "## Dressing Change Frequency\n"
            "Film: every 2–5 days. Hydrocolloid: every 2–5 days. "
            "Change earlier if soiled, loose, curling, or saturated.\n\n"
            "## Application Tips\n"
            "Apply film without trapping air underneath. Remove by stretching slowly from edges. "
            "For small wounds, continue dressing until healed by secondary intention or proceed "
            "to secondary closure if wound is ready.\n\n"
            "## Clinical Notes\n"
            "No antibiotic required. Referral not indicated."
        ),
        "reference_contexts": [ctx(GP_TYPE1), ctx(WCM_FILM), ctx(WCM_HYDROCOLLOID)],
        "allowed_dressings": ["foam","hydrocolloid","film","tulle","alginate","hydrofiber","polymeric_membrane","hydrogel"],
        "contraindicated_dressings": ["silver","charcoal"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type2_wet",
        "category":  "A",
        "synthesizer_name": "cat_a_type2_wet",
        "wound_type_expected": 2,
        "time_payload": {
            "necrotic_pct": 5, "slough_pct": 5, "granulation_pct": 90,
            "infection": "Not infected", "moisture": "High", "edge": "Advancing", "notes": "",
        },
        "user_input": fmt_input(5, 5, 90, "Not infected", "High", "Advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Alginate (sheet or rope form) — absorbs up to 20× its own weight; forms a gel on contact "
            "with wound fluid. Suitable for moderate to heavy exudate; frequency 2–5 days. "
            "Always requires a secondary dressing. Alternatively, hydrofibre manages heavy exuding wounds, "
            "maintains moist healing environment, reduces maceration risk; frequency 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Foam — highly absorbent, conforms to body contours, bacterial and waterproof barrier; "
            "frequency 2–3 days. Can be used alone if exudate is moderate rather than heavy.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 90% granulation, minimal non-viable tissue — wound is essentially clean.\n"
            "- I (Infection): Not infected — no antimicrobial dressing needed.\n"
            "- M (Moisture): High exudate — primary need is high-capacity absorption; "
            "alginate or hydrofibre are the correct first-line choices.\n"
            "- E (Edge): Advancing — wound is progressing normally.\n\n"
            "## Contraindications\n"
            "Silver and charcoal not indicated (clean wound). Alginate and hydrofibre are NOT "
            "suitable for dry wounds — but this wound has high exudate so they are appropriate here.\n\n"
            "## Dressing Change Frequency\n"
            "Alginate/Hydrofibre: 2–5 days. Foam alone: 2–3 days. "
            "Change earlier if dressing is soiled, loose, curling, or saturated.\n\n"
            "## Application Tips\n"
            "Alginate residue must be washed off during wound cleansing. Polymeric membrane dressing "
            "is also suitable as a single-layer option managing moisture imbalance (2–5 days).\n\n"
            "## Clinical Notes\n"
            "Find and treat the underlying cause of high exudate. Antibiotic not indicated unless "
            "underlying cause warrants it. Referral not required."
        ),
        "reference_contexts": [ctx(GP_TYPE2), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM)],
        "allowed_dressings": ["foam","alginate","hydrofiber","polymeric_membrane"],
        "contraindicated_dressings": ["silver","charcoal"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type3_dry_infected",
        "category":  "A",
        "synthesizer_name": "cat_a_type3_dry_infected",
        "wound_type_expected": 3,
        "time_payload": {
            "necrotic_pct": 10, "slough_pct": 12, "granulation_pct": 78,
            "infection": "Locally infected", "moisture": "Low", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(10, 12, 78, "Locally infected", "Low", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — reduces bacterial bioburden in infected wounds; bactericidal with no known "
            "resistance; acts locally at the wound site; frequency 2–3 days. "
            "Alternatively, hydrogel as primary to rehydrate the dry infected wound bed and promote "
            "autolytic debridement; frequency 2–3 days; requires a secondary dressing.\n\n"
            "## Secondary Dressing\n"
            "Non-adherent tulle or low-absorbent pad if using hydrogel as primary. "
            "Hydrocolloid can serve as a combined primary/secondary for low-to-moderate exudate "
            "wounds; promotes autolysis; frequency 2–5 days.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 22% non-viable (<25% threshold) — debridement may be needed but urgent "
            "surgical referral is not yet mandatory.\n"
            "- I (Infection): Infected — antimicrobial dressing is mandatory. "
            "Silver is first-line bactericidal choice.\n"
            "- M (Moisture): Dry/low — dressing should donate moisture (hydrogel) or manage the dry "
            "wound environment without excessive absorption.\n"
            "- E (Edge): Non-advancing — wound is stalled; infection is the likely driver.\n\n"
            "## Contraindications\n"
            "Iodine-based dressings should be AVOIDED in patients with thyroid disorders "
            "(iodine may be absorbed systemically). Confirm no thyroid history before prescribing.\n\n"
            "## Dressing Change Frequency\n"
            "Silver: 2–3 days. Hydrogel: 2–3 days. Hydrocolloid: 2–5 days.\n\n"
            "## Application Tips\n"
            "Apply silver dressing with silver side facing the wound bed. "
            "For hydrogel, apply directly to wound bed and cover with secondary dressing.\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — based on culture and sensitivity (C&S) report of infected tissue. "
            "Debridement may be needed. Referral not indicated for Type 3 (non-viable tissue <25%)."
        ),
        "reference_contexts": [ctx(GP_TYPE3), ctx(WCM_SILVER), ctx(WCM_HYDROGEL), ctx(SFP_IODINE)],
        "allowed_dressings": ["tulle","hydrogel","hydrocolloid","silver","iodine"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type4_wet_infected",
        "category":  "A",
        "synthesizer_name": "cat_a_type4_wet_infected",
        "wound_type_expected": 4,
        "time_payload": {
            "necrotic_pct": 12, "slough_pct": 10, "granulation_pct": 78,
            "infection": "Locally infected", "moisture": "High", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(12, 10, 78, "Locally infected", "High", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — bactericidal, no known resistance, acts locally; frequency 2–3 days. "
            "Combined with alginate to manage heavy exudate (absorbs up to 20× its weight; 2–5 days) "
            "or hydrofibre (manages heavy exuding infected wounds; 2–5 days).\n\n"
            "## Secondary Dressing\n"
            "Foam — highly absorbent, bacterial barrier; frequency 2–3 days. "
            "Can be used over alginate or hydrofibre as secondary.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 22% non-viable (<25%) — minimal debridement burden.\n"
            "- I (Infection): Infected + high exudate (purulent) — antimicrobial mandatory; "
            "silver is first-line. Iodine is an option but avoid in thyroid disorders.\n"
            "- M (Moisture): High — alginate or hydrofibre needed for exudate control; "
            "foam as secondary or single-layer absorbent.\n"
            "- E (Edge): Non-advancing — infection and exudate load are likely drivers.\n\n"
            "## Contraindications\n"
            "Iodine: avoid if patient has thyroid disorder. "
            "Occlusive foam without silver should not be used alone on infected wounds.\n\n"
            "## Dressing Change Frequency\n"
            "Silver: 2–3 days. Alginate/Hydrofibre: 2–5 days. Foam: 2–3 days.\n\n"
            "## Application Tips\n"
            "Alginate residue must be washed off during cleansing. "
            "Hydrofibre forms a gel layer that allows non-traumatic removal.\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — based on C&S report. Debridement may be needed. "
            "Referral not required for Type 4 (<25% non-viable)."
        ),
        "reference_contexts": [ctx(GP_TYPE4), ctx(WCM_SILVER), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE)],
        "allowed_dressings": ["alginate","foam","silver","hydrofiber","polymeric_membrane","iodine"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type5_dry_necrotic",
        "category":  "A",
        "synthesizer_name": "cat_a_type5_dry_necrotic",
        "wound_type_expected": 5,
        "time_payload": {
            "necrotic_pct": 45, "slough_pct": 25, "granulation_pct": 30,
            "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(45, 25, 30, "Not infected", "Low", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Hydrogel — primary choice for dry necrotic wounds; gently rehydrates dry necrotic tissue, "
            "promotes autolytic debridement, softens necrotic tissue, reduces pain; "
            "needs a secondary dressing; frequency 2–3 days.\n\n"
            "## Secondary Dressing\n"
            "Non-adherent pad or hydrocolloid. Hydrocolloid can serve as combined dressing: "
            "cleans and debrides by autolysis, promotes granulation, effective for low-to-moderate exudate; "
            "frequency 2–5 days. Polymeric membrane is also suitable (dry-to-moderate moisture management; 2–5 days).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 70% non-viable (>25% threshold) — debridement is mandatory.\n"
            "- I (Infection): Not infected — no antimicrobial dressing needed.\n"
            "- M (Moisture): Dry — hydrogel donates moisture to rehydrate necrotic eschar; "
            "alginate/hydrofibre are NOT suitable for dry wounds.\n"
            "- E (Edge): Non-advancing — high necrotic burden is likely driving stall.\n\n"
            "## Contraindications\n"
            "Alginate: NOT suitable for dry wounds. Hydrofibre: NOT appropriate for dry wounds. "
            "Silver/charcoal: not indicated (non-infected wound).\n\n"
            "## Dressing Change Frequency\n"
            "Hydrogel: 2–3 days. Hydrocolloid: 2–5 days. Polymeric membrane: 2–5 days.\n\n"
            "## Application Tips\n"
            "Apply hydrogel directly to wound bed; always cover with secondary dressing. "
            "Periwound skin may need protection from maceration.\n\n"
            "## Clinical Notes\n"
            "Debridement IS needed. No antibiotic required. Referral not indicated."
        ),
        "reference_contexts": [ctx(GP_TYPE5), ctx(WCM_HYDROGEL), ctx(WCM_DEBRIDE), ctx(WCM_ALGINATE)],
        "allowed_dressings": ["hydrogel","hydrocolloid","polymeric_membrane"],
        "contraindicated_dressings": ["alginate","silver","charcoal"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type6_wet_necrotic",
        "category":  "A",
        "synthesizer_name": "cat_a_type6_wet_necrotic",
        "wound_type_expected": 6,
        "time_payload": {
            "necrotic_pct": 30, "slough_pct": 35, "granulation_pct": 35,
            "infection": "Not infected", "moisture": "High", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(30, 35, 35, "Not infected", "High", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Alginate — highly absorbent, haemostatic, forms gel on contact with wound fluid; "
            "manages the high exudate load; frequency 2–5 days; needs secondary dressing. "
            "Hydrofibre is an alternative: manages heavy exuding wounds, reduces maceration, "
            "longer wear time; frequency 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Foam — highly absorbent, bacterial barrier; frequency 2–3 days. "
            "Polymeric membrane (antiseptic + surfactant cleansing + moisture management; 2–5 days).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 65% non-viable (>25%) — surgical/mechanical debridement is recommended.\n"
            "- I (Infection): Not infected — antimicrobial dressing not required.\n"
            "- M (Moisture): High — alginate or hydrofibre are primary exudate managers.\n"
            "- E (Edge): Non-advancing — high necrotic burden driving stall.\n\n"
            "## Contraindications\n"
            "Hydrogel and hydrocolloid are not primary choices for high exudate wounds. "
            "Polymeric membrane is also not indicated for heavily exudative wounds as sole dressing.\n\n"
            "## Dressing Change Frequency\n"
            "Alginate/Hydrofibre: 2–5 days. Foam: 2–3 days.\n\n"
            "## Application Tips\n"
            "Alginate residue must be washed off during cleansing. May need repeated debridement sessions.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 6 must be referred to hospital. "
            "Antibiotic: may or may not be needed based on underlying cause."
        ),
        "reference_contexts": [ctx(GP_TYPE6), ctx(GP_REFERRAL), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE)],
        "allowed_dressings": ["alginate","foam","polymeric_membrane","hydrofiber"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type7_dry_infected_necrotic",
        "category":  "A",
        "synthesizer_name": "cat_a_type7_dry_infected_necrotic",
        "wound_type_expected": 7,
        "time_payload": {
            "necrotic_pct": 40, "slough_pct": 30, "granulation_pct": 30,
            "infection": "Locally infected", "moisture": "Low", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(40, 30, 30, "Locally infected", "Low", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — first-line antimicrobial for infected wounds with high necrotic burden; "
            "bactericidal, no known resistance, locally acting; frequency 2–3 days. "
            "Hydrogel can be co-applied to rehydrate the dry necrotic tissue and promote autolytic "
            "debridement; frequency 2–3 days; needs secondary dressing.\n\n"
            "## Secondary Dressing\n"
            "Hydrocolloid (promotes autolysis, moist environment; 2–5 days) or non-adherent pad.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 70% non-viable (>25%) — surgical debridement strongly recommended.\n"
            "- I (Infection): Infected — silver is primary antimicrobial choice. "
            "Iodine is an option but avoid in thyroid disorders.\n"
            "- M (Moisture): Dry — hydrogel donates moisture; alginate/hydrofibre contraindicated in dry wounds.\n"
            "- E (Edge): Non-advancing — combination of infection and necrosis driving stall.\n\n"
            "## Contraindications\n"
            "Iodine: avoid if patient has thyroid disorder. "
            "Alginate and hydrofibre: NOT suitable for dry wounds.\n\n"
            "## Dressing Change Frequency\n"
            "Silver: 2–3 days. Hydrogel: 2–3 days. Hydrocolloid: 2–5 days.\n\n"
            "## Application Tips\n"
            "Apply silver with silver side facing wound bed. "
            "Polymeric membrane dressing (antiseptic properties, 2–5 days) is also listed for Type 7.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 7 must be referred to hospital. "
            "Antibiotic: YES — based on C&S report. "
            "Surgical/mechanical debridement is STRONGLY recommended."
        ),
        "reference_contexts": [ctx(GP_TYPE7), ctx(GP_REFERRAL), ctx(WCM_SILVER), ctx(WCM_HYDROGEL)],
        "allowed_dressings": ["silver","hydrogel","hydrocolloid","iodine","polymeric_membrane"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_a_type8_wet_infected_necrotic",
        "category":  "A",
        "synthesizer_name": "cat_a_type8_wet_infected_necrotic",
        "wound_type_expected": 8,
        "time_payload": {
            "necrotic_pct": 30, "slough_pct": 35, "granulation_pct": 35,
            "infection": "Locally infected", "moisture": "High", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(30, 35, 35, "Locally infected", "High", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing as antimicrobial layer (bactericidal, no known resistance; 2–3 days) "
            "combined with alginate for heavy exudate management (up to 20× absorption, 2–5 days) or "
            "hydrofibre (manages heavy exuding infected wounds; 2–5 days).\n\n"
            "## Secondary Dressing\n"
            "Foam (highly absorbent, bacterial barrier; 2–3 days). "
            "If malodour is present, add charcoal dressing (odour absorbent; needs secondary dressing; "
            "frequency 2 days).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 65% non-viable (>25%) — debridement strongly recommended.\n"
            "- I (Infection): Infected with high exudate — antimicrobial mandatory. "
            "Silver is bactericidal first-line. Iodine option but avoid in thyroid disorders.\n"
            "- M (Moisture): High — alginate or hydrofibre for primary exudate management.\n"
            "- E (Edge): Non-advancing — driven by infection and necrotic burden.\n\n"
            "## Contraindications\n"
            "Iodine: avoid if thyroid disorder. "
            "Occlusive foam without silver should not be used alone on infected wounds.\n\n"
            "## Dressing Change Frequency\n"
            "Silver: 2–3 days. Alginate/Hydrofibre: 2–5 days. Foam: 2–3 days. Charcoal: 2 days.\n\n"
            "## Application Tips\n"
            "Alginate residue must be washed off during cleansing. May need repeated debridement.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 8 must be referred to hospital. "
            "Antibiotic: YES — based on C&S report. "
            "Surgical/mechanical debridement STRONGLY recommended; may need repeated sessions."
        ),
        "reference_contexts": [ctx(GP_TYPE8), ctx(GP_REFERRAL), ctx(WCM_SILVER), ctx(WCM_ALGINATE), ctx(WCM_CHARCOAL)],
        "allowed_dressings": ["alginate","silver","hydrofiber","foam","polymeric_membrane","charcoal","iodine"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY B — Contraindication, safety, and special-population cases
    # ═══════════════════════════════════════════════════════════════════════

    {
        "case_id":   "cat_b_iodine_thyroid",
        "category":  "B",
        "synthesizer_name": "cat_b_iodine_thyroid",
        "wound_type_expected": 3,
        "time_payload": {
            "necrotic_pct": 8, "slough_pct": 15, "granulation_pct": 77,
            "infection": "Locally infected", "moisture": "Low", "edge": "Non-advancing",
            "notes": "Patient has hypothyroidism, currently on levothyroxine 100mcg OD.",
        },
        "user_input": fmt_input(8, 15, 77, "Locally infected", "Low", "Non-advancing",
                                "Patient has hypothyroidism, currently on levothyroxine 100mcg OD."),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — SAFE alternative for infection control; bactericidal, no known "
            "resistance, locally acting; frequency 2–3 days. "
            "Hydrogel can be combined to rehydrate the dry wound; frequency 2–3 days.\n\n"
            "## Secondary Dressing\n"
            "Hydrocolloid (moist environment, autolysis; 2–5 days) or non-adherent pad.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 23% non-viable (<25%) — debridement may be needed.\n"
            "- I (Infection): Infected — antimicrobial mandatory. "
            "IODINE IS CONTRAINDICATED in this patient.\n"
            "- M (Moisture): Dry — silver + hydrogel combination addresses both infection and dryness.\n"
            "- E (Edge): Non-advancing — infection is likely driver.\n\n"
            "## Contraindications\n"
            "IODINE-BASED DRESSINGS ARE CONTRAINDICATED — iodine may be absorbed systemically and "
            "must be avoided in patients with thyroid disorders (hypothyroidism, hyperthyroidism, "
            "patients on levothyroxine or antithyroid medications).\n\n"
            "## Dressing Change Frequency\n"
            "Silver: 2–3 days. Hydrogel: 2–3 days. Hydrocolloid: 2–5 days.\n\n"
            "## Application Tips\n"
            "Document thyroid status and flag the iodine contraindication clearly in the wound care plan.\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — based on C&S report. Debridement may be needed. Referral not required."
        ),
        "reference_contexts": [ctx(SFP_IODINE), ctx(GP_TYPE3), ctx(WCM_SILVER)],
        "allowed_dressings": ["tulle","hydrogel","hydrocolloid","silver"],
        "contraindicated_dressings": ["iodine"],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_silver_clean_granulating",
        "category":  "B",
        "synthesizer_name": "cat_b_silver_clean_granulating",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Low", "edge": "Advancing",
            "notes": "Previous clinician applied a silver dressing.",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Low", "Advancing",
                                "Previous clinician applied a silver dressing."),
        "reference": (
            "## Primary Dressing\n"
            "Film dressing — transparent, waterproof, bacterial barrier; frequency 2–5 days. "
            "Hydrocolloid is also appropriate — moist environment, autolysis, easy to use; 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Not required for film. Foam (without silver) may be added for cushioning.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 100% granulation — clean wound bed; no antimicrobial needed.\n"
            "- I (Infection): Not infected — silver is CONTRAINDICATED on clean granulating wounds.\n"
            "- M (Moisture): Dry — film or hydrocolloid maintain appropriate moisture balance.\n"
            "- E (Edge): Advancing — wound is healing; maintain current strategy.\n\n"
            "## Contraindications\n"
            "SILVER DRESSING IS INCORRECT — the clinical algorithm explicitly excludes silver, charcoal, "
            "and special advanced dressing materials from Wound Type 1 (clean healthy granulating wound).\n\n"
            "## Dressing Change Frequency\n"
            "Film: 2–5 days. Hydrocolloid: 2–5 days.\n\n"
            "## Clinical Notes\n"
            "No antibiotic required. Referral not indicated."
        ),
        "reference_contexts": [ctx(GP_TYPE1), ctx(WCM_FILM), ctx(WCM_HYDROCOLLOID)],
        "allowed_dressings": ["foam","hydrocolloid","film","tulle","hydrogel"],
        "contraindicated_dressings": ["silver","charcoal"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_skin_tear_fragile",
        "category":  "B",
        "synthesizer_name": "cat_b_skin_tear_fragile",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Low", "edge": "Advancing",
            "notes": "82-year-old patient. Skin tear right forearm. Fragile papery skin. Previous nurse applied adhesive bordered foam dressing.",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Low", "Advancing",
                                "82-year-old patient. Skin tear right forearm. Fragile papery skin. Previous nurse applied adhesive bordered foam dressing."),
        "reference": (
            "## Primary Dressing\n"
            "Silicone-coated foam dressing applied directly over the wound (after repositioning the skin flap). "
            "Silicone foam does not adhere to fragile skin and is atraumatic on removal.\n\n"
            "## Secondary Dressing\n"
            "Barrier wipe applied under the foam to secure application, reduce maceration, "
            "and protect periwound skin during removal. If bleeding present, apply haemostatic alginate "
            "as primary dressing under the silicone foam.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): Clean, granulating skin tear — no debridement needed.\n"
            "- I (Infection): Not infected — no antimicrobial needed.\n"
            "- M (Moisture): Low — silicone foam maintains moisture without excess absorption.\n"
            "- E (Edge): Advancing — skin flap is viable; reposition before dressing.\n\n"
            "## Contraindications\n"
            "ADHESIVE BORDERED FOAM IS INCORRECT — adhesive products must NOT be used on fragile skin, "
            "especially on forearms and hands of the elderly; they may cause further skin tears.\n\n"
            "## Application Tips\n"
            "Use remover wipes to remove dressing from fragile skin. "
            "Remove in a direction that does not disturb viable tissue edges and flaps.\n\n"
            "## Clinical Notes\n"
            "No antibiotic required. Referral not indicated."
        ),
        "reference_contexts": [ctx(AJGP_SKINTEAR), ctx(SFP_FOAM)],
        "allowed_dressings": ["silicone_foam"],
        "contraindicated_dressings": ["adhesive_bordered_foam"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_npwt_necrotic_eschar",
        "category":  "B",
        "synthesizer_name": "cat_b_npwt_necrotic_eschar",
        "wound_type_expected": 8,
        "time_payload": {
            "necrotic_pct": 50, "slough_pct": 20, "granulation_pct": 30,
            "infection": "Locally infected", "moisture": "High", "edge": "Non-advancing",
            "notes": "Surgeon is considering applying NPWT vacuum dressing as the sole treatment. 50% black necrotic eschar present.",
        },
        "user_input": fmt_input(50, 20, 30, "Locally infected", "High", "Non-advancing",
                                "Surgeon is considering applying NPWT vacuum dressing as the sole treatment. 50% black necrotic eschar present."),
        "reference": (
            "## Primary Dressing\n"
            "NPWT is NOT appropriate as sole treatment. Correct interim dressings for Type 8:\n"
            "Silver dressing (bactericidal; 2–3 days) combined with alginate (high exudate absorption; "
            "2–5 days) or hydrofibre (heavy exuding infected wounds; 2–5 days).\n\n"
            "## Secondary Dressing\n"
            "Foam (bacterial barrier, highly absorbent; 2–3 days). "
            "Charcoal if malodour present (2 days, needs secondary dressing).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 70% non-viable (>25%) with 50% necrotic eschar — "
            "surgical debridement is mandatory BEFORE NPWT can be considered.\n"
            "- I (Infection): Infected — antimicrobial mandatory; silver is first-line.\n"
            "- M (Moisture): High — alginate or hydrofibre for exudate management.\n"
            "- E (Edge): Non-advancing.\n\n"
            "## Contraindications\n"
            "NPWT IS CONTRAINDICATED in necrotic wound bed or eschar — necrotic tissue acts as a "
            "barrier to new tissue growth. NPWT is also only an ADJUNCT treatment; it does not replace "
            "surgical procedures. Additional NPWT contraindications: clotting disorders, untreated "
            "infection, neoplastic tissue in wound area.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 8. Antibiotic: YES — C&S report. "
            "Surgical/mechanical debridement STRONGLY recommended first."
        ),
        "reference_contexts": [ctx(WCM_NPWT), ctx(GP_TYPE8), ctx(GP_REFERRAL)],
        "allowed_dressings": ["alginate","silver","hydrofiber","foam","polymeric_membrane","charcoal","iodine"],
        "contraindicated_dressings": ["npwt"],
        "antibiotic_required": True,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_alginate_dry_wound",
        "category":  "B",
        "synthesizer_name": "cat_b_alginate_dry_wound",
        "wound_type_expected": 5,
        "time_payload": {
            "necrotic_pct": 40, "slough_pct": 30, "granulation_pct": 30,
            "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing",
            "notes": "Previous clinician applied alginate dressing. Wound completely dry — no exudate visible.",
        },
        "user_input": fmt_input(40, 30, 30, "Not infected", "Low", "Non-advancing",
                                "Previous clinician applied alginate dressing. Wound completely dry — no exudate visible."),
        "reference": (
            "## Primary Dressing\n"
            "Hydrogel — correct choice for this dry necrotic wound; rehydrates necrotic tissue, "
            "promotes autolytic debridement; frequency 2–3 days; requires secondary dressing.\n\n"
            "## Secondary Dressing\n"
            "Hydrocolloid (combined debridement and moisture management; 2–5 days) or non-adherent pad. "
            "Polymeric membrane is also suitable (dry-to-moderate moisture management; 2–5 days).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 70% non-viable (>25%) — debridement is needed.\n"
            "- I (Infection): Not infected — no antimicrobial needed.\n"
            "- M (Moisture): Dry — alginate is INCORRECT for dry wounds.\n"
            "- E (Edge): Non-advancing — necrotic burden is the driver.\n\n"
            "## Contraindications\n"
            "ALGINATE IS INCORRECT for dry wounds — it requires exudate to form its gel and provide "
            "function. Applied to a dry wound, it provides no benefit.\n\n"
            "## Dressing Change Frequency\n"
            "Hydrogel: 2–3 days. Hydrocolloid: 2–5 days.\n\n"
            "## Clinical Notes\n"
            "Debridement IS needed. No antibiotic required. Referral not indicated for Type 5."
        ),
        "reference_contexts": [ctx(GP_TYPE5), ctx(WCM_ALGINATE), ctx(WCM_HYDROGEL)],
        "allowed_dressings": ["hydrogel","hydrocolloid","polymeric_membrane"],
        "contraindicated_dressings": ["alginate"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_honey_dry_necrotic",
        "category":  "B",
        "synthesizer_name": "cat_b_honey_dry_necrotic",
        "wound_type_expected": 5,
        "time_payload": {
            "necrotic_pct": 50, "slough_pct": 20, "granulation_pct": 30,
            "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing",
            "notes": "Clinician considering honey dressing for debridement. Wound is dry with significant necrotic eschar.",
        },
        "user_input": fmt_input(50, 20, 30, "Not infected", "Low", "Non-advancing",
                                "Clinician considering honey dressing for debridement. Wound is dry with significant necrotic eschar."),
        "reference": (
            "## Primary Dressing\n"
            "Hydrogel — correct choice; gently rehydrates dry necrotic tissue, promotes autolytic "
            "debridement, reduces pain; frequency 2–3 days; needs secondary dressing.\n\n"
            "## Secondary Dressing\n"
            "Hydrocolloid (promotes autolysis, moist environment; 2–5 days) or non-adherent pad.\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 70% non-viable — debridement is mandatory.\n"
            "- I (Infection): Not infected — no antimicrobial needed.\n"
            "- M (Moisture): Dry — need a moisture-donating dressing, not an absorbent.\n"
            "- E (Edge): Non-advancing.\n\n"
            "## Contraindications\n"
            "HONEY IS CONTRAINDICATED for dry necrotic wounds — honey may cause further drying of "
            "the wound. Alginate is also contraindicated on dry wounds.\n\n"
            "## Clinical Notes\n"
            "Debridement IS needed. No antibiotic. No referral required for Type 5."
        ),
        "reference_contexts": [ctx(WCM_HONEY), ctx(GP_TYPE5), ctx(WCM_HYDROGEL)],
        "allowed_dressings": ["hydrogel","hydrocolloid","polymeric_membrane"],
        "contraindicated_dressings": ["honey"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_postop_clean",
        "category":  "B",
        "synthesizer_name": "cat_b_postop_clean",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Low", "edge": "Advancing",
            "notes": "Clean postoperative incision day 3, healing by primary intention. Sutured. No exudate. Patient wants to shower daily.",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Low", "Advancing",
                                "Clean postoperative incision day 3, healing by primary intention. Sutured. No exudate. Patient wants to shower daily."),
        "reference": (
            "## Primary Dressing\n"
            "Film dressing — best choice for postoperative wounds without exudate; dress over sutures. "
            "Transparent for easy monitoring, waterproof (allows daily showering), gas-permeable; "
            "wear time 2–5 days. "
            "Thin hydrocolloid is also acceptable: moist environment, autolysis, waterproof; 2–5 days.\n\n"
            "## Contraindications\n"
            "Foam and alginate are NOT indicated — no exudate present.\n\n"
            "## Dressing Change Frequency\n"
            "Film: every 2–5 days. Change if soiled, loose, or lifting at edges.\n\n"
            "## Application Tips\n"
            "Film is waterproof — showering is permitted with film in place. "
            "Apply without trapping air. Remove by stretching slowly from edges.\n\n"
            "## Clinical Notes\n"
            "If wound dehisces: organise prompt surgical review. "
            "No antibiotic required. Referral not indicated."
        ),
        "reference_contexts": [ctx(AJGP_POSTOP), ctx(WCM_FILM), ctx(SFP_FILM)],
        "allowed_dressings": ["film","hydrocolloid"],
        "contraindicated_dressings": ["foam","alginate"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_burns_hand",
        "category":  "B",
        "synthesizer_name": "cat_b_burns_hand",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing",
            "notes": "Small superficial partial-thickness burn from hot water. Located on palm of right hand. First aid (cool running water) already done. Blisters present.",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Moderate", "Advancing",
                                "Small superficial partial-thickness burn from hot water. Located on palm of right hand. First aid (cool running water) already done. Blisters present."),
        "reference": (
            "## Primary Dressing\n"
            "Hydrogel — after initial first aid; provides moist environment, reduces pain, rehydrates "
            "wound bed; frequency 2–3 days. "
            "Hydrocolloid is also suitable: maintains moist environment, promotes autolysis; 2–5 days. "
            "Film dressing as an alternative for superficial burns: transparent, waterproof; 2–5 days.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — burns located on hands (and feet, face, genitalia) must be referred "
            "to a burns specialist regardless of size or depth. "
            "No antibiotic unless signs of infection develop."
        ),
        "reference_contexts": [ctx(AJGP_BURNS), ctx(WCM_HYDROGEL), ctx(WCM_HYDROCOLLOID)],
        "allowed_dressings": ["hydrogel","hydrocolloid","film"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_referral_type6",
        "category":  "B",
        "synthesizer_name": "cat_b_referral_type6",
        "wound_type_expected": 6,
        "time_payload": {
            "necrotic_pct": 30, "slough_pct": 30, "granulation_pct": 40,
            "infection": "Not infected", "moisture": "High", "edge": "Non-advancing", "notes": "",
        },
        "user_input": fmt_input(30, 30, 40, "Not infected", "High", "Non-advancing"),
        "reference": (
            "## Primary Dressing\n"
            "Alginate — highly absorbent, forms gel on contact with wound fluid; "
            "for heavy exudate management; frequency 2–5 days; needs secondary dressing. "
            "Hydrofibre is an alternative: manages heavy exuding wounds, reduces maceration; 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Foam (highly absorbent, bacterial barrier; 2–3 days) or polymeric membrane (2–5 days).\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 6 must be referred to hospital for extensive wound care "
            "(surgical debridement, vacuum dressing, systemic complication management). "
            "Antibiotic: may or may not be needed based on underlying cause."
        ),
        "reference_contexts": [ctx(GP_TYPE6), ctx(GP_REFERRAL), ctx(WCM_ALGINATE)],
        "allowed_dressings": ["alginate","foam","polymeric_membrane","hydrofiber"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_b_diabetic_foot",
        "category":  "B",
        "synthesizer_name": "cat_b_diabetic_foot",
        "wound_type_expected": 3,
        "time_payload": {
            "necrotic_pct": 10, "slough_pct": 15, "granulation_pct": 75,
            "infection": "Locally infected", "moisture": "Moderate", "edge": "Non-advancing",
            "notes": "Patient has Type 2 diabetes. Plantar surface right foot. Pedal pulses should be checked. Moderate exudate.",
        },
        "user_input": fmt_input(10, 15, 75, "Locally infected", "Moderate", "Non-advancing",
                                "Patient has Type 2 diabetes. Plantar surface right foot. Pedal pulses should be checked. Moderate exudate."),
        "reference": (
            "## Primary Dressing\n"
            "Antimicrobial primary dressing — silver dressing (bactericidal, no known resistance; 2–3 days).\n\n"
            "## Secondary Dressing\n"
            "Silicone foam for moderate exudate — applied WITHOUT borders on the foot and anchored "
            "with tape or bandages (never use adhesive borders on diabetic foot).\n\n"
            "## Contraindications\n"
            "Do NOT use bordered adhesive foam on feet. "
            "Hydrocolloid is not recommended for diabetic foot ulcers.\n\n"
            "## Application Tips\n"
            "Assess pedal pulses and sensation — if poor perfusion detected, refer to diabetic foot "
            "clinic or vascular surgeon.\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — based on C&S report. Referral to diabetic foot clinic if poor perfusion."
        ),
        "reference_contexts": [ctx(AJGP_DIABFOOT), ctx(WCM_SILVER), ctx(SFP_HYDROCOLLOID)],
        "allowed_dressings": ["silver","iodine","silicone_foam"],
        "contraindicated_dressings": ["bordered_foam","hydrocolloid"],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY C — Dressing selection and reasoning cases
    # ═══════════════════════════════════════════════════════════════════════

    {
        "case_id":   "cat_c_dressing_saturation",
        "category":  "C",
        "synthesizer_name": "cat_c_dressing_saturation",
        "wound_type_expected": 2,
        "time_payload": {
            "necrotic_pct": 5, "slough_pct": 5, "granulation_pct": 90,
            "infection": "Not infected", "moisture": "High", "edge": "Advancing",
            "notes": "Foam dressing applied 3 days ago. On examination: dressing is soiled, fluid has struck through (strikethrough), edges are curling. Patient asking if it needs changing.",
        },
        "user_input": fmt_input(5, 5, 90, "Not infected", "High", "Advancing",
                                "Foam dressing applied 3 days ago. On examination: dressing is soiled, fluid has struck through (strikethrough), edges are curling. Patient asking if it needs changing."),
        "reference": (
            "## Primary Dressing\n"
            "Change the dressing IMMEDIATELY — all three change criteria are met (soiled, strikethrough, curling edges). "
            "Replace with alginate as primary (absorbs up to 20× weight; 2–5 days) or hydrofibre "
            "(manages heavy exuding wounds; reduces maceration; 2–5 days) under foam as secondary.\n\n"
            "## Secondary Dressing\n"
            "Foam — highly absorbent, waterproof; frequency 2–3 days.\n\n"
            "## Application Tips\n"
            "A dressing MUST be changed when: soiled, loose/slipping, edges curling, or "
            "fluid accumulated (strikethrough visible). Scheduled frequency is a guideline only — "
            "clinical judgment overrides.\n\n"
            "## Clinical Notes\n"
            "No antibiotic required. Referral not indicated for Type 2."
        ),
        "reference_contexts": [ctx(GP_TYPE2), ctx(WCM_FOAM), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE)],
        "allowed_dressings": ["alginate","foam","hydrofiber","polymeric_membrane"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_c_malodour_type8",
        "category":  "C",
        "synthesizer_name": "cat_c_malodour_type8",
        "wound_type_expected": 8,
        "time_payload": {
            "necrotic_pct": 30, "slough_pct": 35, "granulation_pct": 35,
            "infection": "Locally infected", "moisture": "High", "edge": "Non-advancing",
            "notes": "Significant malodour noted by nursing staff and patient family. Dressing changes very distressing due to smell.",
        },
        "user_input": fmt_input(30, 35, 35, "Locally infected", "High", "Non-advancing",
                                "Significant malodour noted by nursing staff and patient family. Dressing changes very distressing due to smell."),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing as antimicrobial (bactericidal, no known resistance; 2–3 days) combined "
            "with alginate or hydrofibre for heavy exudate management (2–5 days).\n\n"
            "## Secondary Dressing\n"
            "Charcoal dressing — specifically indicated for malodour management; absorbs odours; "
            "frequency 2 days; must have secondary dressing over it. "
            "Foam provides additional absorbency.\n\n"
            "## Application Tips\n"
            "Charcoal dressing must not be cut — cutting disrupts the charcoal layer and reduces "
            "odour-absorbing effectiveness.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 8. Antibiotic: YES — C&S. "
            "Surgical/mechanical debridement STRONGLY recommended."
        ),
        "reference_contexts": [ctx(GP_TYPE8), ctx(WCM_CHARCOAL), ctx(WCM_SILVER), ctx(GP_REFERRAL)],
        "allowed_dressings": ["alginate","silver","hydrofiber","foam","polymeric_membrane","charcoal","iodine"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_c_heavy_exudate_maceration",
        "category":  "C",
        "synthesizer_name": "cat_c_heavy_exudate_maceration",
        "wound_type_expected": 2,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 5, "granulation_pct": 95,
            "infection": "Not infected", "moisture": "High", "edge": "Advancing",
            "notes": "Foam dressing saturates within 24 hours. Periwound skin showing early maceration signs. Need better exudate management.",
        },
        "user_input": fmt_input(0, 5, 95, "Not infected", "High", "Advancing",
                                "Foam dressing saturates within 24 hours. Periwound skin showing early maceration signs. Need better exudate management."),
        "reference": (
            "## Primary Dressing\n"
            "Alginate — highest absorption capacity; forms a gel that absorbs and contains exudate; "
            "frequency 2–5 days; needs secondary dressing. Residue must be washed off during cleansing. "
            "Hydrofibre is an equally strong alternative: manages heavy exuding wounds, "
            "reduces maceration risk, longer wear time; frequency 2–5 days.\n\n"
            "## Secondary Dressing\n"
            "Foam — highly absorbent; frequency 2–3 days. "
            "Using alginate or hydrofibre as primary under foam significantly extends wear time.\n\n"
            "## Application Tips\n"
            "Apply a skin barrier/protectant to periwound skin before dressing to address existing "
            "maceration. Consider polymeric membrane dressing as single-layer option (2–5 days).\n\n"
            "## Clinical Notes\n"
            "No antibiotic. Referral not required. Find and treat underlying cause of high exudate."
        ),
        "reference_contexts": [ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM), ctx(GP_TYPE2)],
        "allowed_dressings": ["alginate","hydrofiber","foam","polymeric_membrane"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_c_dry_infected_combo",
        "category":  "C",
        "synthesizer_name": "cat_c_dry_infected_combo",
        "wound_type_expected": 3,
        "time_payload": {
            "necrotic_pct": 10, "slough_pct": 12, "granulation_pct": 78,
            "infection": "Locally infected", "moisture": "Low", "edge": "Non-advancing",
            "notes": "Wound needs both moisture donation and infection control simultaneously.",
        },
        "user_input": fmt_input(10, 12, 78, "Locally infected", "Low", "Non-advancing",
                                "Wound needs both moisture donation and infection control simultaneously."),
        "reference": (
            "## Primary Dressing\n"
            "Hydrogel as primary — rehydrates dry infected wound, promotes autolytic debridement; "
            "frequency 2–3 days; must have secondary dressing.\n\n"
            "## Secondary Dressing\n"
            "Silver dressing as antimicrobial secondary layer over hydrogel — "
            "bactericidal, no known resistance, locally acting; frequency 2–3 days. "
            "This combination addresses both the dry wound environment (hydrogel) and infection "
            "(silver) simultaneously.\n\n"
            "## Contraindications\n"
            "Iodine: avoid if thyroid disorder. Alginate/hydrofibre: not suitable for dry wounds.\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — C&S report. Debridement may be needed. Referral not required."
        ),
        "reference_contexts": [ctx(WCM_HYDROGEL), ctx(WCM_SILVER), ctx(GP_TYPE3)],
        "allowed_dressings": ["tulle","hydrogel","hydrocolloid","silver","iodine"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_c_time_assessment_mixed",
        "category":  "C",
        "synthesizer_name": "cat_c_time_assessment_mixed",
        "wound_type_expected": 7,
        "time_payload": {
            "necrotic_pct": 25, "slough_pct": 25, "granulation_pct": 50,
            "infection": "Locally infected", "moisture": "Moderate", "edge": "Non-advancing",
            "notes": "50% non-viable tissue: 25% black necrosis, 25% yellow slough. Infected — pus, pain, malodour. Moderate exudate.",
        },
        "user_input": fmt_input(25, 25, 50, "Locally infected", "Moderate", "Non-advancing",
                                "50% non-viable tissue: 25% black necrosis, 25% yellow slough. Infected — pus, pain, malodour. Moderate exudate."),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — first-line antimicrobial; bactericidal, no known resistance; 2–3 days. "
            "Combined with hydrogel for moisture donation to the necrotic component (2–3 days).\n\n"
            "## Secondary Dressing\n"
            "Hydrocolloid (promotes autolysis for slough removal; 2–5 days) or non-adherent pad. "
            "Charcoal may be added given malodour (2 days, needs secondary).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- T (Tissue): 50% non-viable (>25% threshold) — algorithm routes to Types 7/8. "
            "Surgical debridement strongly recommended.\n"
            "- I (Infection): Infected with pus and malodour — silver mandatory; charcoal for odour.\n"
            "- M (Moisture): Moderate — Type 7 dressing list is primary.\n"
            "- E (Edge): Non-advancing.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Types 7 and 8 require hospital referral. "
            "Antibiotic: YES — C&S. Surgical/mechanical debridement STRONGLY recommended."
        ),
        "reference_contexts": [ctx(GP_ALGO), ctx(GP_TYPE7), ctx(GP_REFERRAL), ctx(WCM_SILVER)],
        "allowed_dressings": ["silver","hydrogel","hydrocolloid","iodine","polymeric_membrane","charcoal"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_c_film_vs_hydrocolloid",
        "category":  "C",
        "synthesizer_name": "cat_c_film_vs_hydrocolloid",
        "wound_type_expected": 1,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
            "infection": "Not infected", "moisture": "Low", "edge": "Advancing",
            "notes": "Clinician asking: should I use film or hydrocolloid for this clean granulating low-exudate wound? What are the differences?",
        },
        "user_input": fmt_input(0, 0, 100, "Not infected", "Low", "Advancing",
                                "Clinician asking: should I use film or hydrocolloid for this clean granulating low-exudate wound? What are the differences?"),
        "reference": (
            "## Primary Dressing\n"
            "Both film and hydrocolloid are suitable for Wound Type 1 (clean granulating, low exudate):\n"
            "- Film: transparent (best for monitoring), waterproof (allows showering), "
            "gas-permeable but impermeable to bacteria/liquid; wear time 2–5 days.\n"
            "- Hydrocolloid: provides moist environment, promotes autolysis, bacterial barrier, "
            "waterproof; wear time 2–5 days. Forms yellow gel — normal, cleanse on change.\n\n"
            "## Contraindications\n"
            "Silver and charcoal are CONTRAINDICATED for Wound Type 1. "
            "Film: avoid in draining or infected wounds. "
            "Hydrocolloid: NOT recommended for highly exudative or infected wounds.\n\n"
            "## Clinical Notes\n"
            "No antibiotic. Referral not indicated. Consider secondary closure if wound is ready."
        ),
        "reference_contexts": [ctx(WCM_FILM), ctx(WCM_HYDROCOLLOID), ctx(GP_TYPE1), ctx(SFP_FILM)],
        "allowed_dressings": ["film","hydrocolloid","foam","tulle"],
        "contraindicated_dressings": ["silver","charcoal"],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY D — Clinical notes override cases
    # ═══════════════════════════════════════════════════════════════════════

    {
        "case_id":   "cat_d_notes_infection_override",
        "category":  "D",
        "synthesizer_name": "cat_d_notes_infection_override",
        "wound_type_expected": 4,
        "time_payload": {
            "necrotic_pct": 0, "slough_pct": 5, "granulation_pct": 95,
            "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing",
            "notes": "Wound has become more painful over 3 days. Increased warmth and redness around wound edges. Cloudy exudate noted today. Patient reports foul odour.",
        },
        "user_input": fmt_input(0, 5, 95, "Not infected", "Moderate", "Advancing",
                                "Wound has become more painful over 3 days. Increased warmth and redness around wound edges. Cloudy exudate noted today. Patient reports foul odour."),
        "reference": (
            "## Primary Dressing\n"
            "Silver dressing — despite structured label showing 'Not infected', clinical notes contain "
            "multiple infection indicators requiring override. Silver is bactericidal, no known "
            "resistance; frequency 2–3 days.\n\n"
            "## Secondary Dressing\n"
            "Foam based on moderate exudate level (2–3 days).\n\n"
            "## Rationale by T.I.M.E. Factor\n"
            "- I (Infection): OVERRIDE — notes contain: increased wound pain, periwound erythema, "
            "wound warmth, cloudy exudate, foul odour — these are infection red flags. "
            "Treat as INFECTED. Maps to Wound Type 4 (wet, infected, <25% non-viable).\n\n"
            "## Clinical Notes\n"
            "Antibiotic: YES — based on C&S report. Reassess at 48 hours. "
            "If spreading erythema, escalate care immediately."
        ),
        "reference_contexts": [ctx(GP_TYPE4), ctx(WCM_SILVER), ctx(SFP_IODINE)],
        "allowed_dressings": ["silver","iodine","alginate","foam","hydrofiber","polymeric_membrane"],
        "contraindicated_dressings": [],
        "antibiotic_required": True,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_d_notes_diabetic_nonhealing",
        "category":  "D",
        "synthesizer_name": "cat_d_notes_diabetic_nonhealing",
        "wound_type_expected": 3,
        "time_payload": {
            "necrotic_pct": 10, "slough_pct": 20, "granulation_pct": 70,
            "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing",
            "notes": "Patient is diabetic with peripheral neuropathy. Plantar foot wound. No infection currently. Dry wound. Wound has not progressed despite appropriate dressings for 6 weeks. No offloading in use.",
        },
        "user_input": fmt_input(10, 20, 70, "Not infected", "Low", "Non-advancing",
                                "Patient is diabetic with peripheral neuropathy. Plantar foot wound. No infection currently. Dry wound. Wound has not progressed despite appropriate dressings for 6 weeks. No offloading in use."),
        "reference": (
            "## Primary Dressing\n"
            "Antimicrobial primary dressing as precaution given diabetic status — silver dressing "
            "(bactericidal, no known resistance; 2–3 days).\n\n"
            "## Secondary Dressing\n"
            "Low-absorbent pad for low exudate (per AJGP diabetic foot protocol). "
            "Silicone foam if on foot — WITHOUT borders, anchored with tape or bandages.\n\n"
            "## Contraindications\n"
            "Adhesive bordered foam on feet: CONTRAINDICATED. "
            "Hydrocolloid: not recommended for diabetic foot ulcers.\n\n"
            "## Application Tips\n"
            "Check pedal pulses and sensation. Dressing alone will not heal a neuropathic plantar ulcer — "
            "OFFLOADING IS ESSENTIAL.\n\n"
            "## Clinical Notes\n"
            "Non-healing wound at 6 weeks — REFERRAL to diabetic foot clinic is required. "
            "No systemic antibiotic currently if no infection signs, but monitor closely."
        ),
        "reference_contexts": [ctx(AJGP_DIABFOOT), ctx(WCM_SILVER), ctx(SFP_HYDROCOLLOID)],
        "allowed_dressings": ["silver","silicone_foam"],
        "contraindicated_dressings": ["bordered_foam","hydrocolloid"],
        "antibiotic_required": False,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_d_notes_malodour_clean",
        "category":  "D",
        "synthesizer_name": "cat_d_notes_malodour_clean",
        "wound_type_expected": 2,
        "time_payload": {
            "necrotic_pct": 5, "slough_pct": 10, "granulation_pct": 85,
            "infection": "Not infected", "moisture": "High", "edge": "Advancing",
            "notes": "Wound looks clinically clean but significant malodour reported. No purulence. High exudate. Possible bacterial colonisation.",
        },
        "user_input": fmt_input(5, 10, 85, "Not infected", "High", "Advancing",
                                "Wound looks clinically clean but significant malodour reported. No purulence. High exudate. Possible bacterial colonisation."),
        "reference": (
            "## Primary Dressing\n"
            "Alginate (high exudate management; 2–5 days) as primary. "
            "If bacterial colonisation suspected despite no frank infection, consider silver as "
            "antimicrobial precaution (2–3 days).\n\n"
            "## Secondary Dressing\n"
            "Charcoal dressing — specifically targets malodour; absorbs odours effectively; "
            "frequency 2 days; must have secondary dressing. "
            "Foam as final outer secondary (2–3 days).\n\n"
            "## Application Tips\n"
            "Do not cut charcoal dressing — disrupts the active charcoal layer.\n\n"
            "## Clinical Notes\n"
            "Monitor for emerging infection signs. No systemic antibiotic currently. Referral not required."
        ),
        "reference_contexts": [ctx(WCM_CHARCOAL), ctx(WCM_ALGINATE), ctx(GP_TYPE2)],
        "allowed_dressings": ["alginate","charcoal","silver","foam","hydrofiber"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": False,
        "answer": "",
        "retrieved_contexts": [],
    },

    {
        "case_id":   "cat_d_notes_npwt_adjunct",
        "category":  "D",
        "synthesizer_name": "cat_d_notes_npwt_adjunct",
        "wound_type_expected": 6,
        "time_payload": {
            "necrotic_pct": 20, "slough_pct": 30, "granulation_pct": 50,
            "infection": "Not infected", "moisture": "High", "edge": "Non-advancing",
            "notes": "Surgical debridement was completed yesterday. Wound bed is now clean. Surgeon is considering NPWT. Is NPWT now appropriate post-debridement?",
        },
        "user_input": fmt_input(20, 30, 50, "Not infected", "High", "Non-advancing",
                                "Surgical debridement was completed yesterday. Wound bed is now clean. Surgeon is considering NPWT. Is NPWT now appropriate post-debridement?"),
        "reference": (
            "## Primary Dressing\n"
            "NPWT may now be considered as an ADJUNCT following adequate surgical debridement. "
            "Interim standard dressings: alginate for heavy exudate (2–5 days) or hydrofibre (2–5 days) "
            "while NPWT setup is organised.\n\n"
            "## Secondary Dressing\n"
            "Foam as secondary (2–3 days) if NPWT not immediately available.\n\n"
            "## Contraindications\n"
            "NPWT contraindications that must still be checked before proceeding: "
            "(1) Clotting disorders, (2) remaining necrotic tissue/eschar in wound bed, "
            "(3) neoplastic tissue in wound area. All must be cleared before NPWT is applied.\n\n"
            "## Clinical Notes\n"
            "REFERRAL REQUIRED — Wound Type 6 warrants hospital management. "
            "NPWT is an adjunct, not standalone treatment. "
            "No antibiotic if genuinely not infected."
        ),
        "reference_contexts": [ctx(WCM_NPWT), ctx(GP_TYPE6), ctx(GP_REFERRAL), ctx(WCM_ALGINATE)],
        "allowed_dressings": ["alginate","foam","polymeric_membrane","hydrofiber"],
        "contraindicated_dressings": [],
        "antibiotic_required": False,
        "referral_required": True,
        "answer": "",
        "retrieved_contexts": [],
    },

]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./ragas_testset/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_FIELDS = [
    "case_id", "category", "synthesizer_name", "wound_type_expected",
    "time_payload", "user_input", "reference", "reference_contexts",
    "allowed_dressings", "contraindicated_dressings",
    "antibiotic_required", "referral_required",
    "answer", "retrieved_contexts",
]

print(f"\n🔍 Validating {len(testset)} test cases...")
errors = []
for tc in testset:
    for field in REQUIRED_FIELDS:
        if field not in tc:
            errors.append(f"  ❌ [{tc.get('case_id','?')}] missing field: {field}")
    tp = tc.get("time_payload", {})
    total_pct = tp.get("necrotic_pct", 0) + tp.get("slough_pct", 0) + tp.get("granulation_pct", 0)
    if total_pct != 100:
        errors.append(f"  ⚠️  [{tc.get('case_id','?')}] tissue percentages sum to {total_pct}% (expected 100%)")
    if not tc.get("reference_contexts"):
        errors.append(f"  ⚠️  [{tc.get('case_id','?')}] empty reference_contexts")
    # Verify each context string is non-empty (proves it came from a real chunk)
    for i, rc in enumerate(tc.get("reference_contexts", [])):
        if not rc or len(rc) < 50:
            errors.append(f"  ⚠️  [{tc.get('case_id','?')}] reference_contexts[{i}] suspiciously short: {repr(rc[:60])}")

if errors:
    print("\n".join(errors))
    print(f"\n⚠️  {len(errors)} validation issue(s) found.")
else:
    print("✅ All test cases valid.")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "wound_testset_v2.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(testset, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved {len(testset)} test cases → {json_path}")

# Save CSV
import csv
csv_path = os.path.join(OUTPUT_DIR, "wound_testset_v2.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=[
        "case_id", "category", "synthesizer_name", "wound_type_expected",
        "time_payload", "user_input", "reference", "reference_contexts",
        "allowed_dressings", "contraindicated_dressings",
        "antibiotic_required", "referral_required",
        "answer", "retrieved_contexts",
    ])
    w.writeheader()
    for r in testset:
        w.writerow({
            "case_id":                   r["case_id"],
            "category":                  r["category"],
            "synthesizer_name":          r["synthesizer_name"],
            "wound_type_expected":       r["wound_type_expected"],
            "time_payload":              json.dumps(r["time_payload"]),
            "user_input":                r["user_input"],
            "reference":                 r["reference"],
            "reference_contexts":        json.dumps(r["reference_contexts"]),
            "allowed_dressings":         json.dumps(r["allowed_dressings"]),
            "contraindicated_dressings": json.dumps(r["contraindicated_dressings"]),
            "antibiotic_required":       r["antibiotic_required"],
            "referral_required":         r["referral_required"],
            "answer":                    r["answer"],
            "retrieved_contexts":        json.dumps(r["retrieved_contexts"]),
        })
print(f"✅ CSV saved → {csv_path}")

# Summary
print("\n=== TESTSET SUMMARY ===")
cats = {}
for tc in testset:
    c = tc["category"]
    cats[c] = cats.get(c, 0) + 1
for c, n in sorted(cats.items()):
    print(f"  Category {c}: {n} cases")
print(f"  TOTAL: {len(testset)} cases")

print("\n=== CASE LIST ===")
for tc in testset:
    tp   = tc["time_payload"]
    nv   = tp["necrotic_pct"] + tp["slough_pct"]
    refs = len(tc["reference_contexts"])
    print(
        f"  [{tc['category']}] {tc['case_id']:<45} "
        f"T{tc['wound_type_expected']} | "
        f"NV={nv:>3}% | "
        f"Inf={str(tp['infection'])[:3]} | "
        f"Moist={tp['moisture'][:3]} | "
        f"ctx={refs} | "
        f"antibiotic={'Y' if tc['antibiotic_required'] else 'N'} | "
        f"referral={'Y' if tc['referral_required'] else 'N'}"
    )