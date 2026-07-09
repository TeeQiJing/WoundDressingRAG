"""
wound_testset_builder_v5.py
===========================
Generates ragas_testset/wound_testset_v5.json (+ .csv)

v5 = FYP2 testset. Differences from v3:
  - KB v5 (db_wound_care_v5_bge): 8 FYP1 sources + DyaMed (9th). 160 chunks.
  - `reference` rewritten to the **patient-friendly schema** (FYP2 Master Plan Part 13),
    with internal [S#] citations and a mandatory ## Warning section.
  - `reference` names dressing TYPE first (rule-anchored) then an EXAMPLE PRODUCT
    (DyaMed, quoted) — Part 14 dressing-class bridge.
  - `reference_contexts` is RANKED + GRADED (Testset v5 Plan §4): the ordered list of
    ai_summaries (for RAGAS Context Recall/Precision + Faithfulness) PLUS
    `reference_contexts_meta` carrying {rank, chunk_id, grade, role} for MRR / NDCG.
  - antibiotic_required / referral_required follow the MOH algorithm (Master Plan Part 12).
  - Cat F (multimodal: image_ref) and Cat G (adversarial T.I.M.E.–image discrepancy) added.

SCOPE: this file ships the **curated v5 core** — 8 Cat A (WT1–8) + 7 FYP2 cases
(contraindication, escalation, cavity/depth, etiology, multimodal, adversarial).
Extend by following the same make_case(...) pattern (Testset v5 Plan §8/§9).

NO HALLUCINATION POLICY: every reference_contexts chunk_id is verified to exist in
the KB at build time; every clinical claim in `reference` is grounded in a cited chunk.
Grades: 3=binding (answer wrong without it), 2=highly relevant, 1=supporting.
"""

import json, os, csv

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load all chunk ai_summaries (9 sources incl. DyaMed)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_DIR = os.environ.get("WOUND_CHUNK_DIR", "ingestion_output_ai")
_CHUNK_FILES = {
    "GP":     "GP_wound_dressings_kept.json",
    "WCM":    "WCM_wound_care_manual_kept.json",
    "AJGP":   "AJGP_wound_dressings_kept.json",
    "SFP":    "SFP_wound_dressings_kept.json",
    "EWMA":   "EWMA_wound_bed_preparation_kept.json",
    "ISTAP":  "ISTAP_skin_tear_kept.json",
    "ANZBA":  "ANZBA_burns_kept.json",
    "RCH":    "RCH_wound_care_kept.json",
    "DYAMED": "DYAMED_clinical_protocol_kept.json",
}

_CHUNKS, _ABBR = {}, {}
for _src, _fn in _CHUNK_FILES.items():
    _raw = json.load(open(os.path.join(CHUNK_DIR, _fn), encoding="utf-8"))
    _ch  = _raw if isinstance(_raw, list) else _raw.get("kept_chunks", [])
    for _c in _ch:
        _CHUNKS[_c["chunk_id"]] = _c["ai_summary"]
        _ABBR[_c["chunk_id"]]   = _src
print(f"Loaded {len(_CHUNKS)} chunks from {CHUNK_DIR}")

def ctx(cid: str) -> str:
    if cid not in _CHUNKS:
        raise KeyError(f"chunk_id '{cid}' not in KB — fix the ID map.")
    return _CHUNKS[cid]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Chunk ID constants
# ─────────────────────────────────────────────────────────────────────────────
# GP (MOH wound algorithm — the rule anchor)
GP_ALGO="bd2bb8e1321e"; GP_T1="52ef696853c7"; GP_T2="4643f10b8894"; GP_T3="c0a350e36ecf"
GP_T4="d622ee9f4c9c"; GP_T5="aad7a40107b0"; GP_T6="b4ba04cb08d4"; GP_T7="c4177e98524e"
GP_T8="e75347f9bdb3"; GP_REFERRAL="ca7a1e934891"
# WCM (dressing properties + application + debridement)
WCM_FILM="2de03f803f2f"; WCM_HYDROGEL="d81176511903"; WCM_HYDROCOLLOID="f8cb463d04cf"
WCM_ALGINATE="c540b3e5c067"; WCM_FOAM="77e6e32d188a"; WCM_HYDROFIBRE="e63bd0378895"
WCM_CHARCOAL="861a57a2172c"; WCM_SILVER="e8c86c4e1aa6"; WCM_POLYMERIC="6fd9e2433cc9"
WCM_NPWT="05cc6ca1ddfc"; WCM_HONEY="b480aa73a9c2"; WCM_DEBRIDE="b5b5a6c9dcf2"
# SFP (dressing primer + contraindications)
SFP_IODINE="3082e1a296e7"; SFP_FOAM="254dd74d7f00"; SFP_ALGINATE="3b666ccfba99"
SFP_HYDROFIBER="4b75fefc0517"; SFP_SILVER="765445bd6358"; SFP_HYDROGEL="ad036dc35955"
SFP_FILM="9e661711e520"; SFP_HYDROCOLLOID="b4c13d77818b"
# AJGP (etiology)
AJGP_SKINTEAR="55569f16010f"; AJGP_POSTOP="6555b7728ccc"; AJGP_BURNS="5116098922da"
AJGP_DIABFOOT="0d0a9fc09c73"; AJGP_PRINCIPLES="2dc1f26b6233"
# EWMA (VLU / DFU TIME)
EWMA_VLU_TISSUE="142adfaa2033"; EWMA_VLU_INFECTION="bd87881f1796"
EWMA_VLU_MOISTURE="9d379b10e0c1"; EWMA_VLU_EDGE="a60e6a06f137"
# DyaMed — WT protocols (products + steps + change freq)
DY_WT1="bd1a0a4658be"; DY_WT2="386d711b7823"; DY_WT3="d0424bcb7e3d"; DY_WT4="94f9cdb6d2ee"
DY_WT5="9efbefb6bef1"; DY_WT6="da30a820379a"; DY_WT7="48c5c2d1d2ed"; DY_WT8="1e784a4f21c8"
# DyaMed — product monographs (brand + dressing_class + how-to + frequency)
DY_DERMACYN_SOL="d0d9383399f5"; DY_DERMACYN_GEL="40eafc42c039"
DY_FLAMINAL_HYDRO="fc481e763ab3"; DY_FLAMINAL_FORTE="a996f61ef9f7"
DY_ZORFLEX="eacdde76eaed"; DY_DRAWTEX="b036ac66ab55"
DY_RENOCARE_THIN="bd038b453dfa"; DY_RENOCARE_B="8931cff41d40"; DY_RENOFOAM="f0bf52afd21a"
# DyaMed — selection trees
DY_SEL_NONINF="0453a8df1ad3"; DY_SEL_INF="7642fa6bce77"; DY_SEL_PRINCIPLES="3f0933527aa2"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — helpers
# ─────────────────────────────────────────────────────────────────────────────
def rc(rank, cid, grade, role, why=""):
    """One ranked+graded gold context."""
    return {"rank": rank, "chunk_id": cid, "grade": grade, "role": role, "why": why}

def build_ctx(rcs):
    """Return (ordered ai_summary list for RAGAS, ranked+graded meta list)."""
    rcs = sorted(rcs, key=lambda r: r["rank"])
    texts = [ctx(r["chunk_id"]) for r in rcs]                       # for CR/CP/Faithfulness
    meta  = [{"rank": r["rank"], "chunk_id": r["chunk_id"], "grade": r["grade"],
              "abbrev": _ABBR[r["chunk_id"]], "role": r["role"], "why": r["why"]}
             for r in rcs]                                          # for MRR/NDCG
    return texts, meta

def fmt_input(n, s, g, inf, moist, edge, notes=""):
    lines = [f"Necrotic: {n}%, Slough: {s}%, Granulation: {g}%",
             f"Infection: {inf}", f"Moisture: {moist}", f"Edge: {edge}"]
    if notes: lines.append(f"Notes: {notes}")
    return "\n".join(lines)

def make_case(case_id, category, wt, time_payload, reference, rcs, *,
              allowed, example_products, contraindicated,
              antibiotic, referral, change_frequency, escalation_flags,
              conditional_contraindications=None,
              wound_depth="superficial", image_ref=None, demographics=None):
    texts, meta = build_ctx(rcs)
    return {
        "case_id": case_id, "category": category, "wound_type_expected": wt,
        "time_payload": time_payload,
        "user_input": fmt_input(time_payload["necrotic_pct"], time_payload["slough_pct"],
                                time_payload["granulation_pct"], time_payload["infection"],
                                time_payload["moisture"], time_payload["edge"],
                                time_payload.get("notes", "")),
        "demographics": demographics or {"diabetic": False, "age_group": "adult"},
        "wound_depth": wound_depth,
        "image_ref": image_ref,                       # only Cat F/G; caption generated at eval time
        "reference": reference,
        "reference_contexts": texts,                  # RAGAS (ordered)
        "reference_contexts_meta": meta,              # MRR / NDCG (ranked + graded)
        "allowed_dressings": allowed,
        "example_products": example_products,         # type -> DyaMed product (Part 14)
        "contraindicated_dressings": contraindicated,
        "conditional_contraindications": conditional_contraindications or [],
        "antibiotic_required": antibiotic,
        "referral_required": referral,
        "expected_change_frequency": change_frequency,
        "escalation_flags_expected": escalation_flags,
        "answer": "", "retrieved_contexts": [],
    }

WARN_INFECT = ("See a clinician urgently if you notice spreading redness, warmth, "
               "swelling, increasing pain, pus, fever, or a bad smell — these can be signs "
               "the infection is worsening.")
WARN_CLEAN  = ("Seek review promptly if the wound develops spreading redness, warmth, "
               "swelling, increasing pain, pus, or a bad smell — these may indicate infection.")

testset = []

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY A — Core wound types 1–8 (Part 13 patient schema)
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_a_wt1", "A", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing well — mostly healthy pink tissue (granulation), little fluid, and no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a simple protective dressing — a **film** or **thin hydrocolloid**. [S1][S3][S4]\n"
        "- **Secondary:** usually none needed — a thin hydrocolloid is applied on its own, with no secondary dressing required. [S2]\n\n"
        "## Example Products\n"
        "- Hydrocolloid: **RenoCare Thin** (a thin hydrocolloid sheet). [S5]\n"
        "(Products are examples — any film or hydrocolloid of this type is suitable.)\n\n"
        "## Dressings to Avoid\n"
        "- **Silver** and **charcoal** dressings — not needed on a clean, non-infected wound; the wound-care algorithm excludes them for this wound type. [S1]\n\n"
        "## How Often to Change\n"
        "- Film or hydrocolloid: every **2–5 days**; RenoCare Thin can stay up to 7 days. Change sooner if it leaks, lifts, or looks soiled. [S3][S4][S5]\n\n"
        "## Antibiotics?\n"
        "Not needed — your wound is clean and not infected. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "No urgent referral needed. Keep caring for it at home and monitor. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean gently with sterile water or Dermacyn antimicrobial solution, moving from the centre outward. [S2]\n"
        "2. If using a **film**: apply it smoothly with no air trapped underneath [S4]; to remove it without stripping the fragile new skin, gently stretch the film and pull slowly from the edges. [S4]\n"
        "3. If using a **hydrocolloid (e.g. RenoCare Thin)**: warm it between your palms first to help it stick, apply firmly, and you can tape the edges to stop them rolling up. [S5] A little yellow gel may form underneath — this is normal; clean it off at the next change. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 dressing list + silver/charcoal exclusion"),
         rc(2,DY_WT1,3,"primary_protocol","DyaMed WT1 protocol: products, cleanse/apply steps, change freq"),
         rc(3,WCM_HYDROCOLLOID,2,"primary_product","hydrocolloid properties + application + frequency"),
         rc(4,WCM_FILM,2,"primary_product","film properties + application"),
         rc(5,DY_RENOCARE_THIN,2,"example_product","RenoCare Thin = thin hydrocolloid (brand example)")],
    allowed=["film","hydrocolloid","foam","tulle","alginate","hydrofiber","polymeric_membrane","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin","film":"transparent film dressing"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days (RenoCare Thin up to 7 days)"},
    escalation_flags=["monitor for new infection signs"],
    image_ref="ragas_testset/wound_images/WT01_medetec_0021.png"))

testset.append(make_case(
    "cat_a_wt2", "A", 2,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"High","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing but produces a lot of fluid, with no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a high-absorbency dressing — **alginate/alginogel** or **hydrofibre**. [S1][S3][S4]\n"
        "- **Secondary:** an absorbent **foam** to hold fluid and protect the area. [S5]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Forte** (higher-absorbency alginogel for heavy exudate). [S6]\n"
        "- Foam secondary: **RenoFoam** (polyurethane foam). [S7]\n\n"
        "## Dressings to Avoid\n"
        "- None specifically contraindicated for this wound type. [S1]\n\n"
        "## How Often to Change\n"
        "- Alginogel: every other day, extending up to 4 days as fluid settles. [S6]\n"
        "- Foam: every 2–3 days (RenoFoam can stay 3–7 days depending on the amount of fluid); change sooner if it leaks. [S5][S7]\n\n"
        "## Antibiotics?\n"
        "May or may not be needed — it depends on the underlying cause. See a clinician if infection signs appear. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not urgently. If the wound is not improving, get a review to find and treat the underlying cause. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse gently with sterile water or Dermacyn antimicrobial solution, moving from the centre outward — Dermacyn is recommended at every change even without infection, to control inflammation and aid healing; rinse off any biodegradable dressing residue from the last change. [S2][S3]\n"
        "2. If using Flaminal Forte: apply a 0.5 cm layer (about half a thumb's width) evenly over the wound bed. [S6]\n"
        "3. Cover with an absorbent foam reaching beyond the wound edges. [S2][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 dressing list + 'treat underlying cause'"),
         rc(2,DY_WT2,3,"primary_protocol","DyaMed WT2 protocol: Flaminal Hydro/Drawtex/RenoFoam + steps"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate properties/application"),
         rc(4,WCM_HYDROFIBRE,2,"primary_product","hydrofibre alternative"),
         rc(5,WCM_FOAM,2,"secondary_product","foam secondary properties + frequency"),
         rc(6,DY_FLAMINAL_FORTE,2,"example_product","Flaminal Forte alginogel monograph (high exudate) + EOD-4d"),
         rc(7,DY_RENOFOAM,1,"example_product","RenoFoam foam monograph")],
    allowed=["alginate","hydrofiber","foam","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["monitor for new infection signs","review if not improving"],
    image_ref="ragas_testset/wound_images/wsnet_0494.png"))

testset.append(make_case(
    "cat_a_wt3", "A", 3,
    {"necrotic_pct":0,"slough_pct":20,"granulation_pct":80,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection, is fairly dry, and has stalled (the edge is not advancing). A little dead tissue (slough) is present. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** an **antimicrobial** dressing — **silver**; or a **hydrogel** to rehydrate and lift the slough. [S1][S3][S4]\n"
        "- **Secondary:** **tulle** or **hydrocolloid** to hold the primary dressing. [S1][S5]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel** (HOCl). [S6]\n"
        "- Activated-carbon contact layer (low-adherent for dry wounds): **Zorflex LA**. [S7]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine** dressings if you have a **thyroid disorder** — iodine can be absorbed and should be avoided. [S8]\n\n"
        "## How Often to Change\n"
        "- Silver / hydrogel: every 2–3 days; Dermacyn Hydrogel every other day. [S3][S6]\n"
        "- Zorflex can stay 3–7 days. [S7]\n\n"
        "## Antibiotics?\n"
        "Likely needed — your wound shows infection. See a clinician for a swab (culture & sensitivity); do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency referral (this wound type does not require hospital referral), but get a prompt clinical review for the infection and a wound swab. Return for review if the wound is not improving. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse gently from the centre outward using sterile water or Dermacyn solution. [S2]\n"
        "2. If using Dermacyn Hydrogel: apply a thin 3–5 mm layer to the wound bed only — keep it off the healthy surrounding skin to stop that skin softening (maceration). [S2][S4][S6]\n"
        "3. If using a silver dressing: place it silver-side down onto the wound bed. [S3]\n"
        "4. If using Zorflex LA: cut it about 1–2 cm larger than the wound so it covers ~1 cm of the surrounding skin; it can stay 3–7 days. [S2][S7]\n"
        "5. Cover with a tulle or hydrocolloid secondary. If using a hydrocolloid, it may give off a smell and form a yellow gel — this is normal autolytic cleaning, not new pus; clean it off at the next change. [S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T3,3,"algorithm_anchor","WT3 dressing list + antibiotic (C&S) + debridement"),
         rc(2,DY_WT3,3,"primary_protocol","DyaMed WT3 protocol: Dermacyn Hydrogel/Zorflex LA + steps"),
         rc(3,WCM_SILVER,2,"primary_product","silver properties/application"),
         rc(4,WCM_HYDROGEL,2,"primary_product","hydrogel rehydrate/deslough"),
         rc(5,WCM_HYDROCOLLOID,2,"secondary_product","hydrocolloid secondary application"),
         rc(6,DY_DERMACYN_GEL,2,"example_product","Dermacyn Hydrogel monograph + EOD"),
         rc(7,DY_ZORFLEX,1,"example_product","Zorflex/LA monograph (activated carbon)"),
         rc(8,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["silver","hydrogel","tulle","hydrocolloid","iodine"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel","activated_carbon":"Zorflex LA"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD / 2-3 days","Zorflex":"3-7 days"},
    escalation_flags=["wound swab (C&S)","watch for spreading infection"],
    image_ref="ragas_testset/wound_images/WT03_wsnet_0096.png"))

testset.append(make_case(
    "cat_a_wt4", "A", 4,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Locally infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection and produces a lot of fluid. The bed is granulating but the edge has stalled. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** an **antimicrobial, high-absorbency** dressing — **alginate/alginogel** (optionally **silver**) or **hydrofibre**. [S1][S3][S4]\n"
        "- **Secondary:** an absorbent **foam** or hydroconductive layer. [S5]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Forte** (higher-absorbency, antimicrobial). [S6]\n"
        "- Hydroconductive absorbent secondary: **Drawtex**. [S7]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine** dressings if you have a **thyroid disorder** — avoid; iodine can be absorbed. [S8]\n\n"
        "## How Often to Change\n"
        "- Flaminal Forte: every other day, extending up to 4 days as fluid settles. [S6]\n"
        "- Drawtex: every 3–4 days (or with the primary dressing). [S7]\n\n"
        "## Antibiotics?\n"
        "Likely needed — your wound is infected. See a clinician for a swab (culture & sensitivity); do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency referral (this wound type does not require hospital referral), but get a prompt clinical review for the infection. Return for review if the wound is not improving. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse using sterile water or Dermacyn solution, wiping from the centre outward. [S2]\n"
        "2. Apply Flaminal Forte ~0.5 cm thick (about half a thumb's width) evenly over the wound bed. [S6]\n"
        "3. Cover with an absorbent secondary (Drawtex or foam) reaching beyond the wound edges. [S2][S7]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T4,3,"algorithm_anchor","WT4 dressing list + antibiotic (C&S)"),
         rc(2,DY_WT4,3,"primary_protocol","DyaMed WT4 protocol: Flaminal Forte/Zorflex/Drawtex + steps"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate properties"),
         rc(4,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(5,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(6,DY_FLAMINAL_FORTE,2,"example_product","Flaminal Forte alginogel monograph"),
         rc(7,DY_DRAWTEX,2,"example_product","Drawtex hydroconductive monograph"),
         rc(8,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["alginate","silver","hydrofiber","foam","polymeric_membrane","iodine"],
    example_products={"alginogel":"Flaminal Forte","hydroconductive":"Drawtex"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","Drawtex":"3-4 days"},
    escalation_flags=["wound swab (C&S)","watch for spreading infection"],
    image_ref="ragas_testset/wound_images/WT04_wsnet_0466.png"))

testset.append(make_case(
    "cat_a_wt5", "A", 5,
    {"necrotic_pct":45,"slough_pct":25,"granulation_pct":30,"infection":"Not infected","moisture":"Low","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound has a large amount of dead tissue (over a quarter), is fairly dry, and has stalled — but shows no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a **hydrogel** to rehydrate and lift the dead tissue (autolytic debridement). [S1][S3]\n"
        "- **Secondary:** a **hydrocolloid** (or polymeric membrane) to keep it moist. [S1][S5]\n\n"
        "## Example Products\n"
        "- Hydrogel: **Dermacyn WoundCare Hydrogel**. [S6]\n"
        "- Hydrocolloid: **RenoCare Thin**. [S7]\n\n"
        "## Dressings to Avoid\n"
        "- **Alginate** — it needs wound fluid to work and is not suitable for a dry wound. [S4]\n\n"
        "## How Often to Change\n"
        "- Hydrogel: every other day to every 2–3 days [S3][S6]; hydrocolloid every 2–5 days (RenoCare Thin can stay up to 7 days). [S5][S7]\n\n"
        "## Antibiotics?\n"
        "Not needed — your wound is not infected. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency, but the large amount of dead tissue means you should have a clinical review to plan debridement. [S1][S4]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse from the centre outward using sterile water or Dermacyn solution; letting Dermacyn-soaked gauze sit on the wound also helps soften and lift the dead tissue. [S2]\n"
        "2. Apply Dermacyn Hydrogel 3–5 mm onto the wound bed only — keep it off the healthy surrounding skin to prevent it softening (maceration). [S2][S3][S6]\n"
        "3. Cover with a hydrocolloid: cut RenoCare Thin 2–3 cm larger than the wound, warm it between your palms first to help it stick, apply firmly, and tape the edges to stop roll-up. [S2][S5][S7]\n"
        "4. The hydrocolloid may smell and form a yellow gel — this is normal autolytic cleaning, not new pus; wash it off at the next change. [S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T5,3,"algorithm_anchor","WT5 dressing list + 'debridement is needed'"),
         rc(2,DY_WT5,3,"primary_protocol","DyaMed WT5 protocol: Dermacyn Hydrogel/RenoCare + steps"),
         rc(3,WCM_HYDROGEL,2,"primary_product","hydrogel rehydrate/deslough"),
         rc(4,WCM_ALGINATE,2,"contraindication","alginate not helpful on dry wounds"),
         rc(5,WCM_HYDROCOLLOID,2,"secondary_product","hydrocolloid secondary"),
         rc(6,DY_DERMACYN_GEL,2,"example_product","Dermacyn Hydrogel monograph"),
         rc(7,DY_RENOCARE_THIN,1,"example_product","RenoCare Thin hydrocolloid")],
    allowed=["hydrogel","hydrocolloid","polymeric_membrane"],
    example_products={"hydrogel":"Dermacyn WoundCare Hydrogel","hydrocolloid":"RenoCare Thin"},
    contraindicated=["alginate"],
    antibiotic=False, referral=False,
    change_frequency={"hydrogel":"EOD / 2-3 days","hydrocolloid":"2-5 days (RenoCare Thin up to 7 days)"},
    escalation_flags=["clinical review to plan debridement"],
    image_ref="ragas_testset/wound_images/WT05_medetec_0065.png"))

testset.append(make_case(
    "cat_a_wt6", "A", 6,
    {"necrotic_pct":0,"slough_pct":65,"granulation_pct":35,"infection":"Not infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound has a large amount of dead tissue (over a quarter) and produces a lot of fluid, with no signs of infection — but it needs specialist debridement. [S1][S3]\n\n"
        "## Dressing You Need\n"
        "- **Primary (interim):** a high-absorbency **alginate/alginogel** or **hydrofibre**. [S1][S4][S5]\n"
        "- **Secondary:** an absorbent **foam**. [S6]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Forte** (enzyme alginogel, 5.5% alginate). [S7]\n"
        "- Foam secondary: **RenoFoam** (polyurethane foam). [S8]\n\n"
        "## Dressings to Avoid\n"
        "- None specifically contraindicated; hydrogel is not listed for this wound type. [S1]\n\n"
        "## How Often to Change\n"
        "- Flaminal Forte: every other day, extending up to 4 days as fluid settles. [S7]\n"
        "- RenoFoam: every 3–7 days depending on fluid — change once you see fluid approaching the edge of the dressing. [S8]\n\n"
        "## Antibiotics?\n"
        "May or may not be needed — this will be decided at specialist review (based on the underlying cause). [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — referral is needed.** This wound needs hospital/clinic review for surgical or mechanical debridement, which may need to be repeated. The dressings above are interim while you arrange this. [S1][S3]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse from the centre outward using sterile water or Dermacyn solution; if you used an alginate, gently wash off any leftover gelled residue from the wound bed. [S2][S4]\n"
        "2. Apply Flaminal Forte ~0.5 cm thick (about half a thumb's width) over the wound bed — straight from the tube, or with a clean spatula or syringe. [S7]\n"
        "3. Cover with an absorbent foam (RenoFoam) reaching beyond the wound edges. If the wound dries out, foam can stick — moisten it and remove gently if so. [S2][S6]\n\n"
        "## ⚠️ Warning — Get Help Now\n"
        "Do not delay the referral. Seek urgent care if the wound develops spreading redness, warmth, fever, or a bad smell."
    ),
    rcs=[rc(1,GP_T6,3,"algorithm_anchor","WT6 dressing list + surgical/mechanical debridement + referral recommended"),
         rc(2,DY_WT6,3,"primary_protocol","DyaMed WT6 protocol: cleanse/apply steps + Flaminal Forte/Drawtex/RenoFoam + change freq"),
         rc(3,GP_REFERRAL,3,"referral","WT6 requires hospital referral (extensive care / surgical debridement)"),
         rc(4,WCM_ALGINATE,2,"primary_product","alginate properties + residue wash-off"),
         rc(5,WCM_HYDROFIBRE,2,"primary_product","hydrofibre alternative"),
         rc(6,WCM_FOAM,2,"secondary_product","foam secondary + adhesion-if-dry warning"),
         rc(7,DY_FLAMINAL_FORTE,2,"example_product","Flaminal Forte enzyme alginogel monograph + 0.5 cm"),
         rc(8,DY_RENOFOAM,1,"example_product","RenoFoam monograph + 3-7 days / edge cue")],
    allowed=["alginate","hydrofiber","foam","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=True,
    change_frequency={"alginogel":"EOD up to 4 days","RenoFoam":"3-7 days (change when exudate nears edge)"},
    escalation_flags=["referral for surgical/mechanical debridement"],
    image_ref="ragas_testset/wound_images/WT06_medetec_0298.png"))

testset.append(make_case(
    "cat_a_wt7", "A", 7,
    {"necrotic_pct":40,"slough_pct":30,"granulation_pct":30,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection and has a large amount of dead tissue (over a quarter), and is fairly dry. Because of this, it needs professional review and debridement. [S1][S3]\n\n"
        "## Dressing You Need\n"
        "- **Primary (interim):** an **antimicrobial** dressing — **silver** to reduce bacteria, or a **hydrogel** to rehydrate the dry dead tissue. [S1][S4][S7]\n"
        "- **Secondary:** a **hydrocolloid**. [S1][S6]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel**. [S7]\n"
        "- Activated-carbon contact layer (low-adherent, for dry wounds): **Zorflex LA**. [S2][S8]\n\n"
        "## Dressings to Avoid\n"
        "- **Alginate** — not suitable for a dry wound. [S5]\n"
        "- **Iodine** dressings if you have a **thyroid disorder** — iodine can be absorbed. [S9]\n\n"
        "## How Often to Change\n"
        "- Silver: every 2–3 days [S4]; Dermacyn Hydrogel: every other day [S7]; hydrocolloid: every 2–5 days [S6]; Zorflex: 3–7 days. [S8]\n\n"
        "## Antibiotics?\n"
        "Needed — your wound is infected. A clinician will take a wound swab (culture & sensitivity) and prescribe the right antibiotic; do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — urgent referral.** This wound (Type 7) needs hospital review for surgical or mechanical debridement, which is strongly recommended. The dressings above are interim while you arrange this. [S1][S3]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse from the centre outward using sterile water or Dermacyn solution. If a hydrocolloid was on before, it may have formed a yellow gel and a smell — this is normal; wash it off. [S2][S6]\n"
        "2. Apply ONE primary dressing:\n"
        "   - Dermacyn Hydrogel: a thin layer on the wound bed only — keep it off the surrounding skin. [S7]\n"
        "   - Silver dressing: place it silver-side down on the wound bed (it may slightly discolour the wound — this is harmless). [S4]\n"
        "   - Zorflex LA: cut it 1–2 cm larger than the wound so it covers ~1 cm of the surrounding skin. [S2][S8]\n"
        "3. Cover with a hydrocolloid secondary — apply the adhesive side without touching the wound bed, and watch the surrounding skin for softening (maceration). [S6]\n\n"
        "## ⚠️ Warning — Get Help Now\n"
        "Arrange the referral urgently. Seek same-day emergency care for signs of serious or spreading infection — fever or chills (possible sepsis), spreading redness or severe cellulitis, rapidly increasing pain, or a foul smell — or if heart or kidney problems are worsening. [S3]"
    ),
    rcs=[rc(1,GP_T7,3,"algorithm_anchor","WT7 dressing list + antibiotic + surgical debridement"),
         rc(2,DY_WT7,3,"primary_protocol","DyaMed WT7 protocol: cleanse + Zorflex LA (low-adherent) + steps + change freq"),
         rc(3,GP_REFERRAL,3,"referral","WT7 referral; sepsis/cellulitis/heart/renal triggers"),
         rc(4,WCM_SILVER,2,"primary_product","silver antimicrobial + may discolour wound"),
         rc(5,WCM_ALGINATE,2,"contraindication","alginate not helpful on dry wounds"),
         rc(6,WCM_HYDROCOLLOID,2,"secondary_product","hydrocolloid secondary + yellow gel/odour + maceration"),
         rc(7,DY_DERMACYN_GEL,2,"example_product","Dermacyn Hydrogel monograph + avoid periwound"),
         rc(8,DY_ZORFLEX,2,"example_product","Zorflex monograph: carbon, periwound 1 cm, 3-7 days"),
         rc(9,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["silver","hydrogel","hydrocolloid","iodine","polymeric_membrane"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel","activated_carbon":"Zorflex LA"},
    contraindicated=["alginate"],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=True,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD","hydrocolloid":"2-5 days","Zorflex":"3-7 days"},
    escalation_flags=["urgent referral for surgical debridement","wound swab (C&S)"],
    image_ref="ragas_testset/wound_images/WT07_wsnet_0539.png"))

testset.append(make_case(
    "cat_a_wt8", "A", 8,
    {"necrotic_pct":30,"slough_pct":35,"granulation_pct":35,"infection":"Locally infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection, has a large amount of dead tissue (over a quarter), and produces a lot of fluid. This is the most serious wound type and needs urgent professional review and debridement. [S1][S3]\n\n"
        "## Dressing You Need\n"
        "- **Primary (interim):** an **antimicrobial, high-absorbency** dressing — **alginate/alginogel**, **silver**, or **hydrofibre**. [S1][S4][S5]\n"
        "- **Secondary:** an absorbent **foam**; add a **charcoal** outer layer if there is a bad smell. [S1][S6][S7]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Forte**. [S8]\n"
        "- Activated carbon (antimicrobial + odour): **Zorflex**. [S9]\n"
        "- Absorbent foam secondary: **RenoFoam** (or Drawtex). [S2][S7]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine** dressings if you have a **thyroid disorder** — iodine can be absorbed. [S10]\n\n"
        "## How Often to Change\n"
        "- Flaminal Forte: every other day, up to 4 days as fluid settles [S8]; silver: every 2–3 days [S4]; charcoal: every 2 days [S6]; Zorflex: 3–7 days [S9]. Overall, change the DyaMed dressings every 2–4 days, or sooner if soaked. [S2]\n\n"
        "## Antibiotics?\n"
        "Needed — your wound is infected. A clinician will take a wound swab (culture & sensitivity) and prescribe the right antibiotic; do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — urgent referral.** This wound (Type 8) needs hospital review for surgical or mechanical debridement, which is strongly recommended and may need to be repeated. The dressings above are interim while you arrange this. [S1][S3]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse from the centre outward using sterile water or Dermacyn solution. If you used an alginate, gently wash off any leftover gelled residue from the wound bed. [S2][S5]\n"
        "2. Apply ONE primary dressing:\n"
        "   - Flaminal Forte: a 0.5 cm layer (about half a thumb's width) over the wound bed — from the tube, or with a clean spatula or syringe. [S8]\n"
        "   - Silver dressing: place it silver-side down on the wound bed (it may slightly discolour the wound — this is harmless). [S4]\n"
        "   - Zorflex: cut it 1–2 cm larger than the wound so it covers ~1 cm of the surrounding skin. [S2][S9]\n"
        "3. Cover with an absorbent foam reaching beyond the wound edges; if the wound dries out the foam can stick, so moisten and remove gently. Add a charcoal layer (changed every 2 days) if there is a bad smell. [S2][S6][S7]\n\n"
        "## ⚠️ Warning — Get Help Now\n"
        "Arrange the referral urgently. Seek same-day emergency care for signs of serious or spreading infection — fever or chills (possible sepsis), spreading redness or severe cellulitis, fast-worsening pain, or a foul smell — or if heart or kidney problems are worsening. [S3]"
    ),
    rcs=[rc(1,GP_T8,3,"algorithm_anchor","WT8 dressing list + antibiotic + surgical debridement"),
         rc(2,DY_WT8,3,"primary_protocol","DyaMed WT8 protocol: cleanse + Flaminal Forte/Zorflex/Drawtex/RenoFoam + foam beyond margins + change 2-4 days"),
         rc(3,GP_REFERRAL,3,"referral","WT8 referral; sepsis/cellulitis/heart/renal triggers"),
         rc(4,WCM_SILVER,2,"primary_product","silver antimicrobial + may discolour wound"),
         rc(5,WCM_ALGINATE,2,"primary_product","alginate exudate management + residue wash-off"),
         rc(6,WCM_CHARCOAL,2,"secondary_product","charcoal odour control, change every 2 days"),
         rc(7,WCM_FOAM,2,"secondary_product","foam secondary + adhesion-if-dry warning"),
         rc(8,DY_FLAMINAL_FORTE,2,"example_product","Flaminal Forte monograph + 0.5 cm"),
         rc(9,DY_ZORFLEX,2,"example_product","Zorflex monograph: carbon, periwound 1 cm, 3-7 days"),
         rc(10,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["alginate","silver","hydrofiber","foam","polymeric_membrane","charcoal","iodine"],
    example_products={"alginogel":"Flaminal Forte","activated_carbon":"Zorflex","foam":"RenoFoam"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=True,
    change_frequency={"alginogel":"EOD up to 4 days","silver":"2-3 days","charcoal":"2 days","Zorflex":"3-7 days"},
    escalation_flags=["urgent referral for surgical debridement","wound swab (C&S)"],
    image_ref="ragas_testset/wound_images/WT08_medetec_0175.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY B — contraindication / safety
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_b_iodine_thyroid", "B", 3,
    {"necrotic_pct":8,"slough_pct":15,"granulation_pct":77,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I have a thyroid condition and take levothyroxine daily."},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection and is fairly dry, with a little dead tissue. You have told us about a thyroid condition — that changes which dressing is safe for you. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** an **antimicrobial** dressing — **silver**, or an antimicrobial **hydrogel**. [S2][S3][S4]\n"
        "- **Secondary:** **hydrocolloid** or a non-adherent pad. [S5]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel**. [S6]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine dressings — avoid.** Because of your thyroid condition, iodine can be absorbed into the body and is not safe for you. Silver is the safe antimicrobial choice. [S1]\n\n"
        "## How Often to Change\n"
        "- Silver / hydrogel: every 2–3 days (Dermacyn Hydrogel EOD); hydrocolloid every 2–5 days. [S3][S6]\n\n"
        "## Antibiotics?\n"
        "Likely needed — see a clinician for a swab (culture & sensitivity). [S2]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency referral, but get a review for the infection. Review again if no improvement in 2–4 weeks. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean with saline. [S2]\n"
        "2. Apply silver (silver side down) or antimicrobial hydrogel — **check the label says no iodine**. [S1][S3]\n"
        "3. Cover with hydrocolloid/non-adherent pad. [S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,SFP_IODINE,3,"contraindication","iodine systemic absorption / thyroid — the binding safety fact"),
         rc(2,GP_T3,3,"algorithm_anchor","WT3 dressing list + antibiotic"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,WCM_HYDROGEL,2,"primary_product","hydrogel"),
         rc(5,WCM_HYDROCOLLOID,2,"secondary_product","hydrocolloid secondary"),
         rc(6,DY_DERMACYN_GEL,2,"example_product","Dermacyn Hydrogel monograph")],
    allowed=["silver","hydrogel","hydrocolloid","tulle"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["iodine"],
    antibiotic=True, referral=False,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD","hydrocolloid":"2-5 days"},
    escalation_flags=["avoid iodine (thyroid)","wound swab (C&S)"],
    image_ref="ragas_testset/wound_images/WT03_wsnet_0096.png"))

testset.append(make_case(
    "cat_b_silver_on_clean", "B", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing",
     "notes":"The nurse last time put a silver dressing on it."},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing well, with no signs of infection. A plain protective dressing is all it needs. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a **film** or **thin hydrocolloid**. [S1][S3]\n"
        "- **Secondary:** usually none. [S3]\n\n"
        "## Example Products\n"
        "- Hydrocolloid: **RenoCare Thin**. [S4]\n\n"
        "## Dressings to Avoid\n"
        "- **Silver (and charcoal) — not needed and not recommended here.** The wound-care algorithm excludes silver for a clean, non-infected wound; using it adds no benefit. [S1]\n\n"
        "## How Often to Change\n"
        "- Film/hydrocolloid: every 2–5 days (RenoCare Thin up to 7 days). [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Not needed — the wound is clean and not infected. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "No. Continue at home and monitor. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean gently with saline. [S2]\n"
        "2. Apply a film or hydrocolloid smoothly. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 excludes silver/charcoal — the binding rule"),
         rc(2,DY_WT1,3,"primary_protocol","DyaMed WT1 protocol + steps"),
         rc(3,WCM_HYDROCOLLOID,2,"primary_product","hydrocolloid/film properties"),
         rc(4,DY_RENOCARE_THIN,2,"example_product","RenoCare Thin hydrocolloid")],
    allowed=["film","hydrocolloid","foam","tulle","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days"},
    escalation_flags=["stop unnecessary silver on clean wound"],
    image_ref="ragas_testset/wound_images/WT01_medetec_0021.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY C — escalation logic (notes-driven)
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_c_diabetic_foot_escalation", "C", 7,
    {"necrotic_pct":0,"slough_pct":70,"granulation_pct":30,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I am diabetic. This is on the sole of my foot and the area around it feels numb."},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of infection with a large amount of dead tissue, on a diabetic foot with reduced feeling. Diabetic foot wounds need extra caution. [S1][S2][S4]\n\n"
        "## Dressing You Need\n"
        "- **Primary (interim):** an **antimicrobial** dressing — **silver** or antimicrobial **hydrogel**. [S1][S3]\n"
        "- **Secondary:** **hydrocolloid** or non-adherent pad. [S3]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel**. [S5]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine** if you have a thyroid disorder. Alginate/hydrofibre are not helpful on a dry wound. [S3]\n\n"
        "## How Often to Change\n"
        "- Silver/hydrogel: every 2–3 days. [S3][S5]\n\n"
        "## Antibiotics?\n"
        "Needed — your wound is infected. A clinician will prescribe after a swab. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — referral is needed.** Diabetic foot wounds with infection and dead tissue need prompt specialist review (offloading, vascular check, and debridement). Do not manage this at home alone. [S2][S4]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean with saline or Dermacyn solution. [S2]\n"
        "2. Apply antimicrobial dressing while you arrange the referral. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n"
        "Seek care promptly — diabetic foot infections can worsen fast. Go urgently if you develop fever, spreading redness, increasing pain, black tissue, or a bad smell."
    ),
    rcs=[rc(1,GP_T7,3,"algorithm_anchor","WT7 dressing + antibiotic + surgical debridement"),
         rc(2,GP_REFERRAL,3,"referral","referral for extensive care"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,AJGP_DIABFOOT,2,"etiology","diabetic foot management — offloading/vascular/referral"),
         rc(5,DY_DERMACYN_GEL,1,"example_product","Dermacyn Hydrogel monograph")],
    allowed=["silver","hydrogel","hydrocolloid","iodine"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["alginate","hydrofiber","iodine (if thyroid disorder)"],
    antibiotic=True, referral=True,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD"},
    escalation_flags=["diabetic foot — urgent referral","offloading + vascular assessment"],
    demographics={"diabetic":True,"age_group":"adult"},
    image_ref="ragas_testset/wound_images/cat_c_fusc_0902.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY D — wound depth / cavity (G4-C)
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_d_cavity_wt2", "D", 2,
    {"necrotic_pct":0,"slough_pct":15,"granulation_pct":85,"infection":"Not infected","moisture":"High","edge":"Non-advancing",
     "notes":"The wound is a hollow pocket/cavity that dips below the surface, not flat."},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing with a lot of fluid, but it is a deep cavity (a pocket) rather than a flat surface — so it needs a dressing that can fill the space. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a **cavity-filling** absorbent — **alginate/alginogel rope** or **hydrofibre ribbon** (not a flat sheet). [S1][S3][S4]\n"
        "- **Secondary:** an absorbent **foam** over the top. [S5]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Hydro** (can be applied into the cavity). [S6]\n"
        "- Foam secondary: **RenoFoam**. [S7]\n\n"
        "## Dressings to Avoid\n"
        "- Flat film/thin sheets alone — they cannot fill a cavity and may trap fluid. [S1]\n\n"
        "## How Often to Change\n"
        "- Alginogel: EOD up to 4 days; foam every 2–3 days. [S5][S6]\n\n"
        "## Antibiotics?\n"
        "May or may not be needed — depends on the cause; see a clinician if infection signs appear. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Get a review to confirm how deep the cavity is and that it is being filled correctly (cavities should be loosely filled, not packed tightly). [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean and gently irrigate the cavity with saline. [S2]\n"
        "2. Loosely fill the cavity with alginate rope / hydrofibre ribbon — do not over-pack. [S3][S6]\n"
        "3. Cover with an absorbent foam. [S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 dressing list"),
         rc(2,DY_WT2,3,"primary_protocol","DyaMed WT2 protocol + steps"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate sheet/rope (cavity) properties"),
         rc(4,WCM_HYDROFIBRE,2,"primary_product","hydrofibre ribbon"),
         rc(5,WCM_FOAM,2,"secondary_product","foam secondary / cavity filler"),
         rc(6,DY_FLAMINAL_HYDRO,2,"example_product","Flaminal Hydro monograph"),
         rc(7,DY_RENOFOAM,1,"example_product","RenoFoam monograph")],
    allowed=["alginate","hydrofiber","foam"],
    example_products={"alginogel":"Flaminal Hydro","foam":"RenoFoam"},
    contraindicated=["film (alone on cavity)"],
    antibiotic=False, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["confirm cavity depth","loose fill, do not over-pack"],
    wound_depth="cavity",
    image_ref="ragas_testset/wound_images/cat_d_medetec_0373.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY E — etiology (EWMA VLU)
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_e_vlu", "E", 4,
    {"necrotic_pct":5,"slough_pct":15,"granulation_pct":80,"infection":"Locally infected","moisture":"High","edge":"Non-advancing",
     "notes":"This is a chronic venous leg ulcer (gaiter area), present for months, with leg swelling."},
    reference=(
        "## Your Wound\n"
        "Your wound is a venous leg ulcer that shows signs of infection and heavy fluid. These ulcers are driven by the leg veins, so dressings alone are not the whole treatment. [S1][S2][S4]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** an **antimicrobial, high-absorbency** dressing — **alginate/alginogel** or **silver**. [S1][S3]\n"
        "- **Secondary:** absorbent **foam**; compression therapy is the key treatment (applied by a clinician). [S4][S5]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Forte**; absorbent secondary: **Drawtex**. [S6][S7]\n\n"
        "## Dressings to Avoid\n"
        "- **Iodine** if you have a thyroid disorder. [S3]\n\n"
        "## How Often to Change\n"
        "- Alginogel EOD up to 4 days; Drawtex 3–4 days — but follow the clinician's compression schedule. [S6][S7]\n\n"
        "## Antibiotics?\n"
        "Likely needed — your ulcer is infected; see a clinician for a swab. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — get a clinical review.** Venous leg ulcers need a vascular assessment and **compression bandaging**, which dressings cannot replace. [S2][S4]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean with saline. [S2]\n"
        "2. Apply the antimicrobial alginogel. [S6]\n"
        "3. Cover with absorbent foam; compression to be applied by your clinician. [S4][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T4,3,"algorithm_anchor","WT4 dressing + antibiotic"),
         rc(2,DY_WT4,3,"primary_protocol","DyaMed WT4 protocol + steps"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,EWMA_VLU_INFECTION,2,"etiology","VLU infection control + compression context"),
         rc(5,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(6,DY_FLAMINAL_FORTE,2,"example_product","Flaminal Forte monograph"),
         rc(7,DY_DRAWTEX,1,"example_product","Drawtex monograph")],
    allowed=["alginate","silver","hydrofiber","foam","iodine"],
    example_products={"alginogel":"Flaminal Forte","hydroconductive":"Drawtex"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=True,
    change_frequency={"alginogel":"EOD up to 4 days","Drawtex":"3-4 days"},
    escalation_flags=["vascular assessment","compression bandaging"],
    image_ref="ragas_testset/wound_images/cat_e_medetec_0142.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY F — multimodal (image + T.I.M.E.) for G4-A/B
# ═════════════════════════════════════════════════════════════════════════════

testset.append(make_case(
    "cat_f_wt2_image", "F", 2,
    {"necrotic_pct":0,"slough_pct":15,"granulation_pct":85,"infection":"Not infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing with a lot of fluid, and no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** high-absorbency **alginate/alginogel** or **hydrofibre**. [S1][S3]\n"
        "- **Secondary:** absorbent **foam**. [S5]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Hydro**; foam: **RenoFoam**. [S6][S7]\n\n"
        "## Dressings to Avoid\n"
        "- None specifically contraindicated for this wound type. [S1]\n\n"
        "## How Often to Change\n"
        "- Alginogel EOD up to 4 days; foam every 2–3 days. [S5][S6]\n\n"
        "## Antibiotics?\n"
        "May or may not be needed — depends on the cause. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not urgently; review if not improving. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean with saline. [S2]\n2. Apply alginogel. [S6]\n3. Cover with foam. [S5]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 dressing list"),
         rc(2,DY_WT2,3,"primary_protocol","DyaMed WT2 protocol"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate"),
         rc(4,WCM_HYDROFIBRE,2,"primary_product","hydrofibre"),
         rc(5,WCM_FOAM,2,"secondary_product","foam"),
         rc(6,DY_FLAMINAL_HYDRO,2,"example_product","Flaminal Hydro"),
         rc(7,DY_RENOFOAM,1,"example_product","RenoFoam")],
    allowed=["alginate","hydrofiber","foam","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Hydro","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["monitor for infection"],
    image_ref="ragas_testset/wound_images/cat_f_medetec_0283.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY G — adversarial T.I.M.E.–image discrepancy (VLM should flag)
# ═════════════════════════════════════════════════════════════════════════════

# VLM-DISC set (7 cases): each feeds a DELIBERATELY WRONG CV label; the image's true state differs.
# Design: the gold FOLLOWS the (wrong) label's recommendation (the rule layer is deterministic) but adds
# a prominent DISCREPANCY caveat that the blind VLM caption should surface. `escalation_flags_expected`
# = the flag the VLM should raise. Images are the already-validated Cat-A/special images, reused with a
# flipped label (a controlled pair). antibiotic/referral follow classify_wound() on the wrong label.
_DISC_WARN = ("Seek review promptly, and urgently if there is spreading redness, warmth, swelling, "
              "increasing pain, pus, fever, or a bad smell.")

# ── (a) MISSED INFECTION — image infected, label says 'not infected' ──────────────────────────
testset.append(make_case(
    "cat_g_miss_infection_wt1", "G", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\nThe automated reading says this wound is clean and not infected — **but the photo shows features that can indicate infection (surrounding redness, slough).** When the picture and the labels disagree, be cautious. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** a simple **film** or **thin hydrocolloid** (per the current labels). [S1][S3]\n- **Secondary:** usually none. [S3]\n\n"
        "## Example Products\n- Hydrocolloid: **RenoCare Thin**. [S4]\n\n"
        "## Dressings to Avoid\n- Silver/charcoal are not needed for a clean wound — but see the note below. [S1]\n\n"
        "## How Often to Change\n- Every 2–5 days; sooner if it leaks or the wound looks worse. [S3]\n\n"
        "## Antibiotics?\nThe labels say none, **but because the photo may show infection, ask a clinician to check (they may swab).** Do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n**Yes — get a review.** The photo and the automated labels disagree on infection; confirm before relying on 'not infected'. [S1]\n\n"
        "## Step-by-Step Care\n1. Clean gently with saline. [S2]\n2. Apply the film/hydrocolloid. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\nThe photo suggests infection may already be present. " + _DISC_WARN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 dressing list (per current labels)"),
         rc(2,DY_WT1,2,"primary_protocol","DyaMed WT1 protocol"),
         rc(3,WCM_HYDROCOLLOID,2,"primary_product","hydrocolloid/film"),
         rc(4,DY_RENOCARE_THIN,1,"example_product","RenoCare Thin"),
         rc(5,GP_T3,2,"contraindication","if infection confirmed → WT3 antimicrobial pathway")],
    allowed=["film","hydrocolloid","foam","tulle","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days"},
    escalation_flags=["VLM should flag visual infection despite 'not infected' label","possible WT1->WT3 miss (antibiotic)"],
    image_ref="ragas_testset/wound_images/WT03_wsnet_0096.png"))

testset.append(make_case(
    "cat_g_miss_infection_wt2", "G", 2,
    {"necrotic_pct":5,"slough_pct":10,"granulation_pct":85,"infection":"Not infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\nThe labels say clean and not infected with heavy fluid — **but the photo shows redness/slough that can indicate infection.** [S1]\n\n"
        "## Dressing You Need\n- **Primary:** high-absorbency **alginate/alginogel**. [S1][S3]\n- **Secondary:** absorbent **foam**. [S4]\n\n"
        "## Example Products\n- Alginogel: **Flaminal Forte**; foam: **RenoFoam**. [S5]\n\n"
        "## Dressings to Avoid\n- None specifically for the labelled type — but see the note below. [S1]\n\n"
        "## How Often to Change\n- Alginogel EOD up to 4 days; foam 2–3 days. [S5]\n\n"
        "## Antibiotics?\nThe labels say none, **but because the photo suggests infection, ask a clinician to swab and decide.** Do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\n**Yes — get a review** for the possible infection the labels missed. [S1]\n\n"
        "## Step-by-Step Care\n1. Clean with saline. [S2]\n2. Apply the alginogel; cover with foam. [S4][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\nThe photo suggests infection may be present. " + _DISC_WARN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 dressing list (per current labels)"),
         rc(2,DY_WT2,2,"primary_protocol","DyaMed WT2 protocol"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate high absorbency"),
         rc(4,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(5,DY_FLAMINAL_FORTE,1,"example_product","Flaminal Forte / RenoFoam"),
         rc(6,GP_T4,2,"contraindication","if infection confirmed → WT4 antibiotic pathway")],
    allowed=["alginate","hydrofiber","foam","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["VLM should flag visual infection despite 'not infected' label","possible WT2->WT4 miss (antibiotic)"],
    image_ref="ragas_testset/wound_images/medetec_0066.png"))

# ── (b) MISSED NECROSIS/DEAD TISSUE — image necrotic/sloughy, label says 'clean granulating' ────
testset.append(make_case(
    "cat_g_miss_necrosis_wt1", "G", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\nThe automated reading says this wound is clean and healing — **but the photo shows a large amount of dead tissue (black eschar / slough)** that the labels missed. Dead tissue changes the plan. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** a simple **film/hydrocolloid** per the current labels [S1][S3] — **but if the dead tissue in the photo is real, a moisture-donating hydrogel to lift it is needed instead.** [S5]\n- **Secondary:** none. [S3]\n\n"
        "## Example Products\n- Hydrocolloid: **RenoCare Thin**. [S4]\n\n"
        "## Dressings to Avoid\n- Nothing specific for the labelled type. [S1]\n\n"
        "## How Often to Change\n- Every 2–5 days. [S3]\n\n"
        "## Antibiotics?\nNot indicated by the labels. [S1]\n\n"
        "## Do You Need to See a Doctor?\n**Yes — get a review.** The photo shows significant dead tissue not in the labels, which needs debridement planning. [S1][S5]\n\n"
        "## Step-by-Step Care\n1. Clean gently with saline. [S2]\n2. Apply the labelled dressing, but have the dead tissue assessed. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\nHave the wound reviewed — significant dead tissue can hide deeper problems. " + _DISC_WARN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 dressing list (per current labels)"),
         rc(2,DY_WT1,2,"primary_protocol","DyaMed WT1 protocol"),
         rc(3,WCM_HYDROCOLLOID,2,"primary_product","hydrocolloid/film"),
         rc(4,DY_RENOCARE_THIN,1,"example_product","RenoCare Thin"),
         rc(5,GP_T5,2,"contraindication","if necrosis real → WT5 debridement pathway")],
    allowed=["film","hydrocolloid","foam","tulle","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days"},
    escalation_flags=["VLM should flag necrotic/dead tissue despite 'clean granulating' label","possible WT1->WT5 miss (debridement)"],
    image_ref="ragas_testset/wound_images/WT05_medetec_0065.png"))

testset.append(make_case(
    "cat_g_miss_necrosis_wt2", "G", 2,
    {"necrotic_pct":0,"slough_pct":5,"granulation_pct":95,"infection":"Not infected","moisture":"High","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\nThe labels say mostly clean and healing — **but the photo shows a lot of dead/sloughy tissue** the labels missed. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** absorbent **alginate/alginogel** per the labels. [S1][S3]\n- **Secondary:** **foam**. [S4]\n\n"
        "## Example Products\n- Alginogel: **Flaminal Forte**; foam: **RenoFoam**. [S5]\n\n"
        "## Dressings to Avoid\n- Nothing specific for the labelled type. [S1]\n\n"
        "## How Often to Change\n- Alginogel EOD up to 4 days; foam 2–3 days. [S5]\n\n"
        "## Antibiotics?\nNot indicated by the labels. [S1]\n\n"
        "## Do You Need to See a Doctor?\n**Yes — get a review.** The heavy dead tissue in the photo may need debridement and possibly referral. [S1][S6]\n\n"
        "## Step-by-Step Care\n1. Clean with saline. [S2]\n2. Apply the alginogel; cover with foam; have the dead tissue assessed. [S4][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\nHave the wound reviewed for the dead tissue the labels missed. " + _DISC_WARN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 dressing list (per current labels)"),
         rc(2,DY_WT2,2,"primary_protocol","DyaMed WT2 protocol"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate"),
         rc(4,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(5,DY_FLAMINAL_FORTE,1,"example_product","Flaminal Forte / RenoFoam"),
         rc(6,GP_T6,2,"contraindication","if >25% dead tissue → WT6 debridement + referral")],
    allowed=["alginate","hydrofiber","foam","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["VLM should flag heavy slough/dead tissue despite 'clean' label","possible WT2->WT6 miss (debridement/referral)"],
    image_ref="ragas_testset/wound_images/WT06_medetec_0298.png"))

# ── (c) CV OVER-CALL — image looks clean, label says 'infected' (false-positive resistance) ─────
testset.append(make_case(
    "cat_g_overcall_wt3", "G", 3,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Locally infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\nThe automated reading marks this as infected — **but the photo looks like a clean, healthy granulating wound with no clear signs of infection.** The infection label may be a false alarm. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** per the labels, an **antimicrobial** dressing (**silver** or antimicrobial **hydrogel**) [S1][S3] — **though a plain dressing may be enough if the wound is actually clean.** [S1]\n- **Secondary:** tulle or hydrocolloid. [S5]\n\n"
        "## Example Products\n- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel**. [S4]\n\n"
        "## Dressings to Avoid\n- Iodine if you have a thyroid disorder. [S6]\n\n"
        "## How Often to Change\n- Silver/hydrogel every 2–3 days. [S3][S4]\n\n"
        "## Antibiotics?\nThe labels say infected, **but the photo does not clearly show infection — a clinician should confirm before starting antibiotics.** Do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\nNot urgent, but **have the infection label confirmed** — the photo suggests the wound may be clean. [S1]\n\n"
        "## Step-by-Step Care\n1. Clean with saline. [S2]\n2. Apply the antimicrobial dressing while the infection is confirmed. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\nIf genuine infection signs appear (spreading redness, pus, fever), seek care. Otherwise a plain dressing may suffice once confirmed."
    ),
    rcs=[rc(1,GP_T3,3,"algorithm_anchor","WT3 dressing list (per current labels)"),
         rc(2,DY_WT3,2,"primary_protocol","DyaMed WT3 protocol"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,DY_DERMACYN_GEL,1,"example_product","Dermacyn Hydrogel"),
         rc(5,WCM_HYDROCOLLOID,2,"secondary_product","hydrocolloid secondary"),
         rc(6,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["silver","hydrogel","tulle","hydrocolloid","iodine"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD / 2-3 days"},
    escalation_flags=["VLM should flag that the wound looks clean despite 'infected' label","possible CV over-call (WT3->WT1)"],
    image_ref="ragas_testset/wound_images/WT01_medetec_0021.png"))

testset.append(make_case(
    "cat_g_overcall_wt4", "G", 4,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Locally infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\nThe reading marks this as infected with heavy fluid — **but the photo looks like a clean granulating wound without clear infection.** The infection label may be a false alarm. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** per the labels, **alginate/alginogel** (optionally **silver**). [S1][S3]\n- **Secondary:** absorbent **foam**. [S4]\n\n"
        "## Example Products\n- Alginogel: **Flaminal Forte**; foam: **RenoFoam**. [S5]\n\n"
        "## Dressings to Avoid\n- Iodine if you have a thyroid disorder. [S6]\n\n"
        "## How Often to Change\n- Alginogel EOD up to 4 days; foam 2–3 days. [S5]\n\n"
        "## Antibiotics?\nThe labels say infected, **but the photo does not clearly show infection — confirm with a clinician (swab) before antibiotics.** Do not self-medicate. [S1]\n\n"
        "## Do You Need to See a Doctor?\nNot urgent, but **confirm the infection label** — the photo suggests the wound may be clean. [S1]\n\n"
        "## Step-by-Step Care\n1. Clean with saline. [S2]\n2. Apply the alginogel; cover with foam. [S4][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\nIf real infection signs appear (spreading redness, pus, fever), seek care."
    ),
    rcs=[rc(1,GP_T4,3,"algorithm_anchor","WT4 dressing list (per current labels)"),
         rc(2,DY_WT4,2,"primary_protocol","DyaMed WT4 protocol"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate"),
         rc(4,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(5,DY_FLAMINAL_FORTE,1,"example_product","Flaminal Forte / RenoFoam"),
         rc(6,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["alginate","silver","hydrofiber","foam","polymeric_membrane","iodine"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["VLM should flag that the wound looks clean despite 'infected' label","possible CV over-call (WT4->WT2)"],
    image_ref="ragas_testset/wound_images/wsnet_0494.png"))

testset.append(make_case(
    "cat_g_overcall_clean", "G", 4,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Locally infected","moisture":"High","edge":"Non-advancing","notes":""},
    reference=(
        "## Your Wound\nThe reading marks this leg wound as infected — **but the photo shows a granulating wound without obvious infection (no pus or spreading redness).** The infection label may be a false alarm. [S1]\n\n"
        "## Dressing You Need\n- **Primary:** per the labels, **alginate/alginogel**. [S1][S3]\n- **Secondary:** absorbent **foam**. [S4]\n\n"
        "## Example Products\n- Alginogel: **Flaminal Forte**; foam: **RenoFoam**. [S5]\n\n"
        "## Dressings to Avoid\n- Iodine if you have a thyroid disorder. [S6]\n\n"
        "## How Often to Change\n- Alginogel EOD up to 4 days; foam 2–3 days. [S5]\n\n"
        "## Antibiotics?\nThe labels say infected, **but the photo does not clearly show infection — a clinician should confirm before antibiotics.** [S1]\n\n"
        "## Do You Need to See a Doctor?\nNot urgent, but **confirm the infection label** and, as a leg ulcer, get a vascular/compression assessment. [S1]\n\n"
        "## Step-by-Step Care\n1. Clean with saline. [S2]\n2. Apply the alginogel; cover with foam. [S4][S5]\n\n"
        "## ⚠️ Warning — Get Help Now\nIf real infection signs appear (spreading redness, pus, fever), seek care."
    ),
    rcs=[rc(1,GP_T4,3,"algorithm_anchor","WT4 dressing list (per current labels)"),
         rc(2,DY_WT4,2,"primary_protocol","DyaMed WT4 protocol"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate"),
         rc(4,WCM_FOAM,2,"secondary_product","foam secondary"),
         rc(5,DY_FLAMINAL_FORTE,1,"example_product","Flaminal Forte / RenoFoam"),
         rc(6,SFP_IODINE,1,"contraindication","iodine thyroid caution")],
    allowed=["alginate","silver","hydrofiber","foam","polymeric_membrane","iodine"],
    example_products={"alginogel":"Flaminal Forte","foam":"RenoFoam"},
    contraindicated=[],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["VLM should flag that the wound looks clean despite 'infected' label","possible CV over-call (WT4->WT2)"],
    image_ref="ragas_testset/wound_images/cat_g_overcall_clean.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY B (expansion) — comorbidity / contraindication (note-driven, reused images)
# ═════════════════════════════════════════════════════════════════════════════
ISTAP_PATHWAY="c3d5e1f498ba"; ISTAP_PRODUCT="3bde291790e7"

# B3 — skin tear on fragile elderly skin → atraumatic silicone foam, AVOID adhesive
testset.append(make_case(
    "cat_b_skin_tear_fragile", "B", 1,
    {"necrotic_pct":0,"slough_pct":5,"granulation_pct":95,"infection":"Not infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I have very thin, fragile, papery skin (I am elderly) and this is a skin tear with a loose flap."},
    reference=(
        "## Your Wound\n"
        "You have a **skin tear** on fragile, ageing skin, with a skin flap and no signs of infection. Fragile skin needs gentle, non-sticky care to avoid making the tear worse. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- First, gently **realign the skin flap** back over the wound if you can. [S1]\n"
        "- **Primary:** a **silicone-coated (atraumatic) foam** placed directly over the wound — it lifts off without tearing the skin. [S1][S3]\n"
        "- If the tear is **bleeding**, put a **haemostatic alginate** underneath the silicone foam. [S1][S4]\n\n"
        "## Example Products\n"
        "- Silicone foam: **RenoFoam** (use without an adhesive border on fragile skin). [S3]\n\n"
        "## Dressings to Avoid\n"
        "- **Adhesive dressings, films, and tape directly on the fragile skin** — peeling them off can cause a new skin tear. Use a barrier wipe and secure with a bandage instead. [S1][S2]\n\n"
        "## How Often to Change\n"
        "- Leave undisturbed for several days; change the foam every **2–3 days** or when strike-through appears. Mark an arrow on the dressing showing the safe peel direction (away from the flap). [S1][S3]\n\n"
        "## Antibiotics?\n"
        "Not needed — there are no signs of infection. [S2]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not urgently. Seek review if the flap turns dark/dusky, or if redness, swelling, or discharge appear. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean gently with saline; do not scrub. [S2]\n"
        "2. Ease the flap back into place with a damp gauze or gloved finger. [S1]\n"
        "3. Apply the silicone foam (borderless) and secure with a light bandage — no tape on skin. [S1][S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,AJGP_SKINTEAR,3,"algorithm_anchor","skin tear: silicone foam, haemostatic alginate if bleeding, AVOID adhesive on fragile skin"),
         rc(2,ISTAP_PATHWAY,3,"primary_protocol","ISTAP skin-tear assessment/treatment pathway (flap realignment, atraumatic)"),
         rc(3,WCM_FOAM,2,"primary_product","foam properties/application/frequency"),
         rc(4,SFP_ALGINATE,2,"primary_product","alginate (haemostatic) for bleeding"),
         rc(5,ISTAP_PRODUCT,1,"example_product","ISTAP product-selection categories for skin tears")],
    allowed=["foam","silicone_foam","alginate","tulle"],
    example_products={"silicone_foam":"RenoFoam"},
    contraindicated=["film","adhesive","tape","hydrocolloid"],
    antibiotic=False, referral=False,
    change_frequency={"silicone_foam":"2-3 days"},
    escalation_flags=["avoid adhesives on fragile skin","watch flap viability (dusky/dark)"],
    image_ref="ragas_testset/wound_images/WT01_medetec_0021.png"))

# B4 — infected wound, patient allergic to bees → AVOID honey, use alternative antimicrobial
testset.append(make_case(
    "cat_b_honey_bee_allergy", "B", 3,
    {"necrotic_pct":0,"slough_pct":15,"granulation_pct":85,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I am allergic to bees and bee stings."},
    reference=(
        "## Your Wound\n"
        "Your wound shows signs of local infection and needs an **antimicrobial** dressing. Medical **honey** is one antimicrobial option — but because you are **allergic to bees**, honey should be avoided for you, and we will use a non-honey antimicrobial instead. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a non-honey **antimicrobial** dressing — **silver**, or an **antimicrobial hydrogel** (HOCl). [S1][S3][S4]\n"
        "- **Secondary:** a simple absorbent pad or tulle. [S1]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel** (HOCl — no honey). [S4]\n\n"
        "## Dressings to Avoid\n"
        "- **Honey (medical-grade) dressings** — avoid because of your bee allergy. [S2]\n\n"
        "## How Often to Change\n"
        "- Silver: every 2–3 days; Dermacyn Hydrogel: every other day. [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Likely needed — your wound is infected. See a clinician for a swab (culture & sensitivity). [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency referral, but get a prompt review for the infection. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse from the centre outward with saline or Dermacyn solution. [S3]\n"
        "2. Apply the chosen non-honey antimicrobial; cover with a pad. [S3][S4]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T3,3,"algorithm_anchor","WT3 antimicrobial dressing + antibiotic (C&S)"),
         rc(2,WCM_HONEY,3,"contraindication","honey = antimicrobial option being avoided (patient bee allergy)"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial (non-honey alternative)"),
         rc(4,DY_DERMACYN_GEL,2,"example_product","Dermacyn HOCl antimicrobial hydrogel (non-honey)"),
         rc(5,DY_WT3,2,"primary_protocol","DyaMed WT3 protocol")],
    allowed=["silver","hydrogel","iodine","tulle","hydrocolloid"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["honey"],
    antibiotic=True, referral=False,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD"},
    escalation_flags=["avoid honey (bee allergy)","wound swab (C&S)"],
    image_ref="ragas_testset/wound_images/WT03_wsnet_0096.png"))

# B5 — sloughy wound, patient on warfarin (anticoagulant) → autolytic (not sharp) debridement, bleeding caution
testset.append(make_case(
    "cat_b_anticoagulant_bleeding", "B", 5,
    {"necrotic_pct":0,"slough_pct":35,"granulation_pct":65,"infection":"Not infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I take warfarin (a blood thinner) and the wound bleeds very easily when the dressing is changed."},
    reference=(
        "## Your Wound\n"
        "Your wound has a moderate amount of dead tissue (slough) that needs lifting, but no signs of infection. Because you take a **blood thinner (warfarin)**, care must avoid causing bleeding. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a **hydrogel** to soften and lift the slough by **autolytic (self) debridement** — gentle and bloodless. [S1][S3]\n"
        "- If the wound **bleeds**, use a **haemostatic alginate** to help it stop. [S4]\n"
        "- Keep dressings **non-adherent/atraumatic** so removal does not tear the bed. [S2]\n\n"
        "## Example Products\n"
        "- Hydrogel: **Dermacyn WoundCare Hydrogel**; haemostatic primary if bleeding: alginate. [S3][S4]\n\n"
        "## Dressings to Avoid\n"
        "- **Sharp/surgical or aggressive mechanical debridement** at the bedside — bleeding risk on an anticoagulant; let the hydrogel deslough gently instead. [S2]\n"
        "- **Dressings that stick to the bed** — they can pull and bleed on removal. [S1]\n\n"
        "## How Often to Change\n"
        "- Hydrogel: every other day; soak the dressing off with saline rather than peeling it. [S3]\n\n"
        "## Antibiotics?\n"
        "Not needed — no signs of infection. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not urgent, but tell your clinician you are on warfarin so debridement is planned safely. Seek help if bleeding does not stop with pressure. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Soak the old dressing off with saline — do not peel it dry. [S2]\n"
        "2. Apply hydrogel to soften the slough; cover with a non-adherent pad. [S3]\n"
        "3. If oozing, apply a haemostatic alginate and light pressure. [S4]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T5,3,"algorithm_anchor","WT5 sloughy/non-viable — deslough, no infection"),
         rc(2,WCM_DEBRIDE,3,"primary_protocol","debridement methods — autolytic vs surgical/sharp (bleeding risk)"),
         rc(3,WCM_HYDROGEL,2,"primary_product","hydrogel rehydrate/deslough (autolytic)"),
         rc(4,SFP_ALGINATE,2,"primary_product","alginate haemostatic for bleeding"),
         rc(5,DY_DERMACYN_GEL,1,"example_product","Dermacyn Hydrogel monograph")],
    allowed=["hydrogel","alginate","hydrocolloid","foam","tulle"],
    example_products={"hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["sharp debridement","surgical debridement","adhesive"],
    antibiotic=False, referral=False,
    change_frequency={"hydrogel":"EOD"},
    escalation_flags=["bleeding caution (anticoagulant)","autolytic not sharp debridement"],
    image_ref="ragas_testset/wound_images/WT06_medetec_0298.png"))

# B6 — infected wound, patient allergic to silver/sulfa → non-silver antimicrobial
testset.append(make_case(
    "cat_b_silver_allergy", "B", 3,
    {"necrotic_pct":0,"slough_pct":12,"granulation_pct":88,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing",
     "notes":"I am allergic to silver and to sulfa medicines."},
    reference=(
        "## Your Wound\n"
        "Your wound is locally infected and needs an **antimicrobial** dressing. Silver is the usual first choice — but you are **allergic to silver (and sulfa)**, so silver (including silver sulfadiazine) must be avoided and a **non-silver antimicrobial** used instead. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a non-silver antimicrobial — an **antimicrobial hydrogel (HOCl)**, or **iodine** (only if you have no thyroid disorder). [S1][S3][S4]\n"
        "- **Secondary:** tulle or a simple pad. [S1]\n\n"
        "## Example Products\n"
        "- Antimicrobial hydrogel: **Dermacyn WoundCare Hydrogel** (HOCl — silver-free). [S3]\n\n"
        "## Dressings to Avoid\n"
        "- **Silver dressings**, including **silver sulfadiazine** — avoid due to your silver/sulfa allergy. [S2]\n"
        "- Iodine **if** you also have a thyroid disorder. [S4]\n\n"
        "## How Often to Change\n"
        "- Dermacyn Hydrogel: every other day; iodine per product guidance. [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Likely needed — the wound is infected. See a clinician for a swab (C&S). [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Not an emergency referral, but get a prompt review for the infection. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse with saline or Dermacyn solution from the centre outward. [S3]\n"
        "2. Apply the silver-free antimicrobial; cover with tulle/pad. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T3,3,"algorithm_anchor","WT3 antimicrobial + antibiotic (C&S)"),
         rc(2,WCM_SILVER,3,"contraindication","silver = agent being avoided (patient silver/sulfa allergy)"),
         rc(3,DY_DERMACYN_GEL,2,"example_product","Dermacyn HOCl antimicrobial hydrogel (silver-free)"),
         rc(4,SFP_IODINE,2,"primary_product","iodine alternative (with thyroid caveat)"),
         rc(5,DY_WT3,2,"primary_protocol","DyaMed WT3 protocol")],
    allowed=["hydrogel","iodine","honey","tulle","hydrocolloid"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["silver"],
    conditional_contraindications=["iodine (if thyroid disorder)"],
    antibiotic=True, referral=False,
    change_frequency={"hydrogel":"EOD"},
    escalation_flags=["avoid silver (allergy)","wound swab (C&S)"],
    image_ref="ragas_testset/wound_images/WT04_wsnet_0466.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY C (expansion) — escalation logic (note-driven, reused images)
# ═════════════════════════════════════════════════════════════════════════════

# C2 — CV says clean, but patient notes reveal infection → notes-driven ANTIBIOTIC escalation (subclinical)
testset.append(make_case(
    "cat_c_spreading_infection", "C", 2,
    {"necrotic_pct":0,"slough_pct":5,"granulation_pct":95,"infection":"Not infected","moisture":"High","edge":"Non-advancing",
     "notes":"The wound has become more painful and hot, is leaking yellow pus, and the redness is spreading around it."},
    reference=(
        "## Your Wound\n"
        "The assessment labelled this wound as **not infected**, but your description — **spreading redness, warmth, increasing pain, and pus** — points to an **infection developing**. We should treat it as infected. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** an **antimicrobial** dressing — **silver** or an antimicrobial hydrogel. [S1][S3]\n"
        "- **Secondary:** an absorbent pad. [S1]\n\n"
        "## Example Products\n"
        "- Antimicrobial: **silver** dressing, or **Dermacyn WoundCare Hydrogel**. [S3][S4]\n\n"
        "## Dressings to Avoid\n"
        "- Plain occlusive films alone — do not use over a wound that may be infected. [S1]\n\n"
        "## How Often to Change\n"
        "- Antimicrobial: every 2–3 days, or sooner if exudate increases. [S3]\n\n"
        "## Antibiotics?\n"
        "**Likely yes.** Your symptoms suggest infection despite the initial 'clean' reading — see a clinician promptly for a **swab (C&S)** and possible antibiotics. Do not self-medicate. [S1][S2]\n\n"
        "## Do You Need to See a Doctor?\n"
        "Yes — get a **prompt review** for the spreading infection. Go urgently if the redness keeps spreading or you develop fever. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse with saline or Dermacyn solution. [S3]\n"
        "2. Apply the antimicrobial dressing and arrange review. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T3,3,"algorithm_anchor","infected-wound dressing + antibiotic (C&S) — applies once infection recognised"),
         rc(2,GP_REFERRAL,3,"referral","escalation criteria — spreading infection needs review"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,DY_DERMACYN_GEL,1,"example_product","Dermacyn HOCl hydrogel"),
         rc(5,DY_WT3,2,"primary_protocol","DyaMed WT3 antimicrobial protocol")],
    allowed=["silver","hydrogel","iodine","tulle"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["film (alone)"],
    antibiotic=True, referral=False,
    change_frequency={"silver":"2-3 days"},
    escalation_flags=["notes reveal infection despite 'clean' label","wound swab (C&S)","watch for spreading/systemic signs"],
    image_ref="ragas_testset/wound_images/wsnet_0494.png"))

# C3 — severe wound + systemic symptoms (fever) → EMERGENCY red-flag (sepsis)
testset.append(make_case(
    "cat_c_sepsis_redflag", "C", 8,
    {"necrotic_pct":20,"slough_pct":40,"granulation_pct":40,"infection":"Locally infected","moisture":"High","edge":"Non-advancing",
     "notes":"I also have a fever, feel shivery and generally unwell, and the pain is getting much worse."},
    reference=(
        "## Your Wound\n"
        "You have a **severe, infected wound with a lot of dead tissue** — **and** you now have a **fever, chills, and feel unwell**. A wound infection plus these whole-body symptoms is a **medical emergency (possible sepsis)**. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- Dressings are **secondary to getting emergency care**. As an interim, cover with an **antimicrobial** dressing (silver) and go for help. [S1][S3]\n\n"
        "## Example Products\n"
        "- Interim antimicrobial: **silver** dressing. [S3]\n\n"
        "## Dressings to Avoid\n"
        "- Do not spend time on elaborate dressing choices — **do not delay emergency care**. [S2]\n\n"
        "## How Often to Change\n"
        "- Not the priority now — seek emergency care first. [S2]\n\n"
        "## Antibiotics?\n"
        "**Yes — urgent.** Systemic antibiotics are likely needed; this must be assessed emergently, not self-managed. [S1][S2]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**YES — go to the emergency department or call for urgent help NOW.** Fever and feeling unwell with an infected wound can mean the infection is spreading through your body. Do not wait. [S1][S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Cover the wound with a clean antimicrobial dressing. [S3]\n"
        "2. Go to emergency care immediately; bring your medication list. [S2]\n\n"
        "## ⚠️ Warning — Get Help Now\n"
        "Fever, shivering/chills, feeling very unwell, spreading redness, or confusion with a wound are **emergency red flags** — seek urgent medical care right away."
    ),
    rcs=[rc(1,GP_T8,3,"algorithm_anchor","WT8 wet infected >25% non-viable — antibiotic + referral"),
         rc(2,GP_REFERRAL,3,"referral","referral/emergency criteria for extensive/systemic infection"),
         rc(3,WCM_SILVER,2,"primary_product","interim silver antimicrobial")],
    allowed=["silver","alginate","hydrofiber","foam"],
    example_products={"antimicrobial":"silver dressing"},
    contraindicated=[],
    antibiotic=True, referral=True,
    change_frequency={"silver":"per hospital"},
    escalation_flags=["EMERGENCY — systemic infection/sepsis red-flag","urgent hospital / antibiotics"],
    image_ref="ragas_testset/wound_images/WT08_medetec_0175.png"))

# C4 — chronic non-healing wound → review for underlying cause + referral
testset.append(make_case(
    "cat_c_chronic_nonhealing", "C", 2,
    {"necrotic_pct":0,"slough_pct":10,"granulation_pct":90,"infection":"Not infected","moisture":"High","edge":"Non-advancing",
     "notes":"This wound has been here for over 4 months and is chronic — it just will not heal despite regular dressings."},
    reference=(
        "## Your Wound\n"
        "Your wound is **not infected**, but it has been **chronic (non-healing) for months**. A dressing alone will not fix a wound that will not heal — the **underlying cause** (circulation, pressure, nutrition, diabetes) must be found and treated. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** keep a **moist, absorbent** dressing appropriate to the exudate — e.g. **alginate/alginogel** or **foam** — while the cause is investigated. [S1][S3]\n\n"
        "## Example Products\n"
        "- Alginogel: **Flaminal Hydro**; foam: **RenoFoam**. [S3][S4]\n\n"
        "## Dressings to Avoid\n"
        "- No specific dressing is contraindicated — the key is investigating the cause, not just changing dressings. [S2]\n\n"
        "## How Often to Change\n"
        "- Per exudate: alginogel EOD–4 days; foam every 2–3 days. [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Not needed now — no signs of infection. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — get a review/referral** for a chronic non-healing wound: a full assessment (blood supply, nutrition, pressure offloading, diabetes control) is needed to find why it is not healing. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Continue moist wound care with an exudate-matched dressing. [S3]\n"
        "2. Arrange a clinical review to investigate and treat the underlying cause. [S2]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T2,3,"algorithm_anchor","WT2 clean/exudative dressing selection"),
         rc(2,AJGP_PRINCIPLES,3,"referral","dressings don't heal alone — treat the underlying cause; chronic wound needs review"),
         rc(3,WCM_ALGINATE,2,"primary_product","alginate for exudate"),
         rc(4,DY_RENOFOAM,1,"example_product","RenoFoam foam"),
         rc(5,DY_WT2,2,"primary_protocol","DyaMed WT2 protocol")],
    allowed=["alginate","hydrofiber","foam","hydrocolloid","polymeric_membrane"],
    example_products={"alginogel":"Flaminal Hydro","foam":"RenoFoam"},
    contraindicated=[],
    antibiotic=False, referral=True,
    change_frequency={"alginogel":"EOD up to 4 days","foam":"2-3 days"},
    escalation_flags=["chronic non-healing — investigate underlying cause","vascular/nutrition/offloading review"],
    image_ref="ragas_testset/wound_images/cat_f_medetec_0283.png"))

# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY D/E/F (expansion) — depth, complex-chronic, image-robustness (new images)
# ═════════════════════════════════════════════════════════════════════════════
ARTERIAL_ULCER="90610861be44"; VENOUS_ULCER="4b89a41a249d"; TIME_WBP="e905d7d38dad"

# D2 — large deep UNDERMINED infected cavity → NPWT + surgical referral
testset.append(make_case(
    "cat_d_deep_cavity_npwt", "D", 8,
    {"necrotic_pct":0,"slough_pct":30,"granulation_pct":70,"infection":"Locally infected","moisture":"High","edge":"Rolled/undermined",
     "notes":"This is a large deep wound with a pocket that undermines (tunnels) under the skin edges, and it produces a lot of fluid."},
    reference=(
        "## Your Wound\n"
        "You have a **large, deep cavity wound** that **undermines** (tunnels under the wound edges), with some dead tissue (slough), signs of infection, and heavy fluid. Deep undermined wounds need specialist management. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Best option:** **Negative Pressure Wound Therapy (NPWT / vacuum dressing)** — it fills the cavity, removes fluid, and closes undermining. This is arranged by a specialist. [S2][S3]\n"
        "- **Interim (until NPWT):** loosely fill the cavity with an **antimicrobial alginate/hydrofibre rope** — do **not** pack tightly — and cover with an absorbent secondary. [S1][S4]\n\n"
        "## Example Products\n"
        "- Interim cavity filler: alginate rope; antimicrobial: **silver**. [S4][S5]\n\n"
        "## Dressings to Avoid\n"
        "- Flat sheet dressings alone — they cannot fill a deep undermined cavity. [S1]\n\n"
        "## How Often to Change\n"
        "- NPWT: per specialist (typically every 2–3 days). Interim filler: daily–alternate days with high fluid. [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Likely needed — the wound is infected. A clinician will swab and prescribe. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — referral is needed.** Deep undermined cavity wounds requiring vacuum (NPWT) or surgical debridement must be referred for specialist wound care. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Irrigate the cavity gently with saline; probe/measure depth and undermining is done by the clinician. [S1]\n"
        "2. Until NPWT is set up, loosely fill with antimicrobial rope and an absorbent cover. [S4]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T8,3,"algorithm_anchor","WT8 wet infected >25% non-viable — antibiotic + referral"),
         rc(2,GP_REFERRAL,3,"referral","referral criteria — NPWT/vacuum + surgical debridement + extensive care"),
         rc(3,WCM_NPWT,3,"primary_protocol","Negative Pressure Wound Therapy for deep/cavity/undermined wounds"),
         rc(4,WCM_ALGINATE,2,"primary_product","alginate rope — interim cavity filler"),
         rc(5,WCM_SILVER,2,"primary_product","silver antimicrobial")],
    allowed=["npwt","alginate","hydrofiber","silver","foam"],
    example_products={"npwt":"Negative Pressure Wound Therapy","cavity_filler":"alginate rope"},
    contraindicated=["film (alone on cavity)"],
    antibiotic=True, referral=True,
    change_frequency={"npwt":"2-3 days (specialist)","alginate":"daily-EOD"},
    escalation_flags=["deep undermined cavity — NPWT / surgical referral","loose fill, do not over-pack"],
    wound_depth="cavity",
    image_ref="ragas_testset/wound_images/cat_d_deep_cavity_npwt.png"))

# D3 — extensive DRY necrotic eschar (unstageable) → debridement referral
testset.append(make_case(
    "cat_d_extreme_necrosis", "D", 5,
    {"necrotic_pct":100,"slough_pct":0,"granulation_pct":0,"infection":"Not infected","moisture":"Low","edge":"Non-advancing",
     "notes":"The whole wound is covered by a thick, hard, dry black scab (eschar) and I cannot tell how deep it goes."},
    reference=(
        "## Your Wound\n"
        "Your wound is completely covered by a **thick, hard, dry black scab (eschar)** — the whole surface is dead tissue, so the true depth cannot be seen (it is 'unstageable'). There are no signs of infection right now. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- The dead tissue needs **debridement**. Because the load is extensive, this needs a **clinician's plan**: often **surgical/sharp debridement**, or **autolytic** softening with a **hydrogel** if debridement is planned. [S1][S2][S3]\n"
        "- Keep the surrounding skin protected and the eschar dry until reviewed. [S3]\n\n"
        "## Example Products\n"
        "- Autolytic softener (if debridement planned): **Dermacyn WoundCare Hydrogel**. [S3]\n\n"
        "## Dressings to Avoid\n"
        "- Do not seal a large dry eschar under an occlusive dressing without a debridement plan. [S2]\n\n"
        "## How Often to Change\n"
        "- Per the debridement plan; if a hydrogel is started, every other day. [S3]\n\n"
        "## Antibiotics?\n"
        "Not needed now — no signs of infection. Watch for new redness, swelling, smell, or discharge. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — get a review.** An extensive, unstageable necrotic eschar needs a clinician to assess depth and plan **surgical debridement**. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Keep the eschar and surrounding skin clean and dry until reviewed. [S3]\n"
        "2. If a hydrogel is prescribed to soften it, apply to the eschar only, off the healthy skin. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T5,3,"algorithm_anchor","WT5 dry necrotic — debridement, no infection"),
         rc(2,WCM_DEBRIDE,3,"primary_protocol","debridement methods — surgical for extensive necrosis"),
         rc(3,GP_REFERRAL,2,"referral","surgical debridement = referral criterion"),
         rc(4,WCM_HYDROGEL,2,"primary_product","hydrogel rehydrate/soften eschar (autolytic)"),
         rc(5,DY_DERMACYN_GEL,1,"example_product","Dermacyn Hydrogel monograph")],
    allowed=["hydrogel","hydrocolloid"],
    example_products={"hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=[],
    antibiotic=False, referral=True,
    change_frequency={"hydrogel":"EOD"},
    escalation_flags=["extensive/unstageable necrosis — surgical debridement referral"],
    image_ref="ragas_testset/wound_images/cat_d_extreme_necrosis.png"))

# E2 — arterial (ischaemic) leg ulcer → compression CONTRAINDICATED + urgent vascular referral
testset.append(make_case(
    "cat_e_arterial_no_compression", "E", 7,
    {"necrotic_pct":10,"slough_pct":70,"granulation_pct":20,"infection":"Locally infected","moisture":"Low","edge":"Non-advancing",
     "notes":"This is on my lower leg; my foot is often cold and pale, the pulses feel weak, and it is very painful — I have poor blood supply (arterial)."},
    reference=(
        "## Your Wound\n"
        "You have a painful, sloughy leg ulcer with signs of infection **and features of poor arterial blood supply** (cold, pale foot, weak pulses). Arterial (ischaemic) ulcers heal only once blood flow is restored — the cause must be treated, not just the wound. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a gentle **antimicrobial** dressing for the infection; deslough **cautiously** (poor perfusion heals slowly). [S1][S3]\n"
        "- Keep dressings **non-adherent** and the limb protected. [S2]\n\n"
        "## Example Products\n"
        "- Antimicrobial: **silver**; antimicrobial hydrogel: **Dermacyn**. [S3][S4]\n\n"
        "## Dressings to Avoid\n"
        "- **COMPRESSION bandaging must NOT be used** — unlike a venous ulcer, compressing a limb with poor arterial supply cuts off blood flow and can cause tissue death. [S1][S2]\n"
        "- Aggressive sharp debridement of ischaemic tissue before vascular assessment. [S2]\n\n"
        "## How Often to Change\n"
        "- Antimicrobial: every 2–3 days; handle the limb gently. [S3]\n\n"
        "## Antibiotics?\n"
        "Likely needed — the wound is infected. See a clinician for a swab. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — urgent vascular referral.** Arterial ulcers need assessment of blood supply (ABPI/Doppler) and often **revascularisation** to heal. Do not apply compression in the meantime. [S1][S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse gently with saline; apply the antimicrobial dressing. [S3]\n"
        "2. Protect and elevate as advised (not compression); arrange urgent vascular review. [S2]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT + " Seek urgent care if the foot becomes more pale, blue, cold, numb, or the pain suddenly worsens — signs of critical loss of blood supply."
    ),
    rcs=[rc(1,GP_T7,3,"algorithm_anchor","WT7 dry infected >25% non-viable — antibiotic + referral"),
         rc(2,ARTERIAL_ULCER,3,"etiology","arterial/ischaemic ulcer — revascularisation, vascular referral, NOT compression"),
         rc(3,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(4,DY_DERMACYN_GEL,1,"example_product","Dermacyn HOCl hydrogel"),
         rc(5,GP_REFERRAL,2,"referral","referral for extensive/vascular care")],
    allowed=["silver","hydrogel","alginate","foam","tulle"],
    example_products={"antimicrobial":"silver dressing","antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel"},
    contraindicated=["compression","compression bandaging"],
    antibiotic=True, referral=True,
    change_frequency={"silver":"2-3 days"},
    escalation_flags=["arterial ulcer — NO compression","urgent vascular referral (ABPI/revascularisation)"],
    image_ref="ragas_testset/wound_images/cat_e_arterial.png"))

# E3 — chronic mixed-tissue wound (granulation + necrotic focus) → TIME wound-bed-preparation
testset.append(make_case(
    "cat_e_mixed_tissue_chronic", "E", 4,
    {"necrotic_pct":20,"slough_pct":0,"granulation_pct":80,"infection":"Locally infected","moisture":"High","edge":"Non-advancing",
     "notes":"This wound has been present for several months (chronic) and will not heal; it has a dark dead patch in the middle with healthy red tissue around it."},
    reference=(
        "## Your Wound\n"
        "You have a **chronic (months-long) wound with mixed tissue** — healthy red granulation around a **dark necrotic patch** — plus signs of infection. Mixed-tissue chronic wounds are managed with structured **wound-bed preparation (the T.I.M.E. approach)**. [S1][S2]\n\n"
        "## Dressing You Need\n"
        "- **T (Tissue):** **debride the necrotic patch** — autolytic **hydrogel** to soften it, or sharp debridement by a clinician. [S2][S3]\n"
        "- **I (Infection):** an **antimicrobial** (silver / antimicrobial hydrogel). [S1][S4]\n"
        "- **M (Moisture):** an **absorbent** secondary for the fluid; **E (Edge):** protect the margin. [S2]\n\n"
        "## Example Products\n"
        "- Autolytic + antimicrobial: **Dermacyn WoundCare Hydrogel**; antimicrobial: **silver**. [S3][S4]\n\n"
        "## Dressings to Avoid\n"
        "- Simply covering it without addressing the necrosis and the underlying cause — the wound will stay stuck. [S2]\n\n"
        "## How Often to Change\n"
        "- Antimicrobial/hydrogel: every 2–3 days (hydrogel EOD). [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Likely needed — the wound is infected. See a clinician for a swab. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "**Yes — get a review/referral** for a chronic non-healing wound: the **underlying cause** must be found and treated, and the necrotic tissue debrided. [S2]\n\n"
        "## Step-by-Step Care\n"
        "1. Cleanse; apply hydrogel over the necrotic patch to soften it. [S3]\n"
        "2. Add an antimicrobial and absorbent cover; arrange review to investigate the chronicity. [S2][S4]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_INFECT
    ),
    rcs=[rc(1,GP_T4,3,"algorithm_anchor","infected wound dressing + antibiotic (C&S)"),
         rc(2,TIME_WBP,3,"primary_protocol","TIME wound-bed-preparation framework (Tissue/Infection/Moisture/Edge) + treat underlying cause"),
         rc(3,WCM_HYDROGEL,2,"primary_product","hydrogel autolytic debridement of necrosis"),
         rc(4,WCM_SILVER,2,"primary_product","silver antimicrobial"),
         rc(5,GP_REFERRAL,2,"referral","chronic non-healing → review underlying cause")],
    allowed=["hydrogel","silver","alginate","hydrofiber","foam"],
    example_products={"antimicrobial_hydrogel":"Dermacyn WoundCare Hydrogel","antimicrobial":"silver dressing"},
    contraindicated=[],
    antibiotic=True, referral=True,
    change_frequency={"silver":"2-3 days","hydrogel":"EOD"},
    escalation_flags=["chronic mixed-tissue — TIME wound-bed-prep + debride necrosis","investigate underlying cause"],
    image_ref="ragas_testset/wound_images/cat_e_mixed_tissue.png"))

# F2 — image-robustness: clean superficial wound, darker skin tone
testset.append(make_case(
    "cat_f_clean_darkskin", "F", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing — healthy pink/red granulation tissue, little fluid, and no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a simple protective dressing — a **film** or **thin hydrocolloid**. [S1][S3]\n"
        "- **Secondary:** usually none needed. [S2]\n\n"
        "## Example Products\n"
        "- Hydrocolloid: **RenoCare Thin**. [S4]\n\n"
        "## Dressings to Avoid\n"
        "- **Silver / charcoal** — not needed on a clean, non-infected wound. [S1]\n\n"
        "## How Often to Change\n"
        "- Film/hydrocolloid: every 2–5 days (RenoCare Thin up to 7). [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Not needed — clean and not infected. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "No urgent referral. Keep caring for it at home and monitor. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean gently with saline from the centre outward. [S2]\n"
        "2. Apply the film or thin hydrocolloid smoothly. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 dressing list + silver/charcoal exclusion"),
         rc(2,DY_WT1,3,"primary_protocol","DyaMed WT1 protocol"),
         rc(3,WCM_FILM,2,"primary_product","film properties/application"),
         rc(4,DY_RENOCARE_THIN,2,"example_product","RenoCare Thin thin hydrocolloid")],
    allowed=["film","hydrocolloid","foam","tulle","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin","film":"transparent film dressing"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days"},
    escalation_flags=["monitor for new infection signs"],
    image_ref="ragas_testset/wound_images/cat_f_clean_darkskin.png"))

# F3 — image-robustness: clean superficial wound, limb (different anatomy)
testset.append(make_case(
    "cat_f_clean_limb", "F", 1,
    {"necrotic_pct":0,"slough_pct":0,"granulation_pct":100,"infection":"Not infected","moisture":"Low","edge":"Advancing","notes":""},
    reference=(
        "## Your Wound\n"
        "Your wound is clean and healing — healthy granulation tissue, minimal fluid, no signs of infection. [S1]\n\n"
        "## Dressing You Need\n"
        "- **Primary:** a **film** or **thin hydrocolloid** for protection. [S1][S3]\n"
        "- **Secondary:** usually none needed. [S2]\n\n"
        "## Example Products\n"
        "- Hydrocolloid: **RenoCare Thin**. [S4]\n\n"
        "## Dressings to Avoid\n"
        "- **Silver / charcoal** — unnecessary on a clean wound. [S1]\n\n"
        "## How Often to Change\n"
        "- Film/hydrocolloid: every 2–5 days. [S3][S4]\n\n"
        "## Antibiotics?\n"
        "Not needed — clean and not infected. [S1]\n\n"
        "## Do You Need to See a Doctor?\n"
        "No urgent referral. Monitor at home. [S1]\n\n"
        "## Step-by-Step Care\n"
        "1. Clean gently with saline. [S2]\n"
        "2. Apply the protective dressing smoothly. [S3]\n\n"
        "## ⚠️ Warning — Get Help Now\n" + WARN_CLEAN
    ),
    rcs=[rc(1,GP_T1,3,"algorithm_anchor","WT1 dressing list + silver/charcoal exclusion"),
         rc(2,DY_WT1,3,"primary_protocol","DyaMed WT1 protocol"),
         rc(3,WCM_HYDROCOLLOID,2,"primary_product","hydrocolloid properties/application"),
         rc(4,DY_RENOCARE_THIN,2,"example_product","RenoCare Thin thin hydrocolloid")],
    allowed=["film","hydrocolloid","foam","tulle","hydrogel"],
    example_products={"hydrocolloid":"RenoCare Thin","film":"transparent film dressing"},
    contraindicated=["silver","charcoal"],
    antibiotic=False, referral=False,
    change_frequency={"film":"2-5 days","hydrocolloid":"2-5 days"},
    escalation_flags=["monitor for new infection signs"],
    image_ref="ragas_testset/wound_images/cat_f_clean_limb.png"))

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION + EXPORT
# ═════════════════════════════════════════════════════════════════════════════
# Part 12 ground-truth matrix (antibiotic, referral) for Cat A sanity check
_MATRIX = {1:(False,False),2:(False,False),3:(True,False),4:(True,False),
           5:(False,False),6:(False,True),7:(True,True),8:(True,True)}

def _validate(ts):
    ids = [c["case_id"] for c in ts]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    for c in ts:
        # every reference has the mandatory warning section
        assert "## ⚠️ Warning — Get Help Now" in c["reference"], f"{c['case_id']}: missing Warning section"
        # ranks 1..n sequential
        ranks = [m["rank"] for m in c["reference_contexts_meta"]]
        assert ranks == list(range(1, len(ranks)+1)), f"{c['case_id']}: ranks not 1..n"
        # at least one binding (grade 3) context
        assert any(m["grade"] == 3 for m in c["reference_contexts_meta"]), f"{c['case_id']}: no grade-3 anchor"
        # reference_contexts strings align 1:1 with meta
        assert len(c["reference_contexts"]) == len(c["reference_contexts_meta"]), f"{c['case_id']}: ctx/meta mismatch"
        # Cat A obeys the MOH matrix
        if c["category"] == "A":
            ab, rf = _MATRIX[c["wound_type_expected"]]
            assert c["antibiotic_required"] == ab and c["referral_required"] == rf, \
                f"{c['case_id']}: antibiotic/referral != Part 12 matrix"
    print(f"Validation passed: {len(ts)} cases.")

_validate(testset)

OUT_JSON = os.path.join("ragas_testset", "wound_testset_v5.json")
OUT_CSV  = os.path.join("ragas_testset", "wound_testset_v5.csv")
os.makedirs("ragas_testset", exist_ok=True)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=2)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id","category","wound_type","antibiotic","referral","wound_depth",
                "image_ref","n_contexts","gold_chunk_ids(ranked)","reference_excerpt"])
    for c in testset:
        ids = " > ".join(f"{m['abbrev']}:{m['chunk_id'][:8]}(g{m['grade']})" for m in c["reference_contexts_meta"])
        w.writerow([c["case_id"], c["category"], c["wound_type_expected"], c["antibiotic_required"],
                    c["referral_required"], c["wound_depth"], c["image_ref"] or "",
                    len(c["reference_contexts_meta"]), ids, c["reference"][:160].replace("\n"," ")])

from collections import Counter
print(f"Wrote {OUT_JSON}  ({len(testset)} cases)")
print(f"Wrote {OUT_CSV}")
print("By category:", dict(Counter(c["category"] for c in testset)))
print("Cat A wound types:", sorted(c["wound_type_expected"] for c in testset if c["category"]=="A"))
print("Multimodal (image_ref) cases:", [c["case_id"] for c in testset if c["image_ref"]])
