"""
ingestion_ANZBA.py
==================
VerdaSense Wound RAG — ANZBA Burns Guidelines Ingestion
Cell-by-cell Python file. Paste each # ── CELL N ── block into a separate
Jupyter notebook cell. Comment blocks beginning with triple-quotes are Markdown cells.

Source documents (4 PDFs + 1 PNG image — same organisation):
  1. ANZBA_First_Aid_Consensus_Hydrogel_v3_Aug_2021.pdf
  2. ANZBA-Minor-BurnPoster-v2.pdf
  3. ANZBA-Severe-BurnPoster-v2-1.pdf
  4. ANZBA_Pharmacists_Advice_Poster_for_Burn_Injury.pdf
  5. ANZBA_Referral_Criteria.png  (image — content hardcoded from OCR)

All 5 merged under one SOURCE_NAME (same organisation/framework).
Output: ANZBA_burns_kept.json  (4 chunks, ChromaDB-ready)
"""

# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Title & Document Profile
# ══════════════════════════════════════════════════════════════════════════════
"""
# `ingestion_ANZBA.ipynb`
## Australian & New Zealand Burn Association (ANZBA) — Burns Guidelines

### Document profile

| Property | Detail |
|---|---|
| Organisation | Australian & New Zealand Burn Association (ANZBA) |
| Year | 2021 (Hydrogel consensus); posters undated but current edition |
| Documents | 4 PDFs + 1 PNG — Hydrogel Consensus, Minor Burns Poster, Severe Burns Poster, Pharmacist Advice Poster, Referral Criteria image |
| Layout | Single-page poster / flowchart layouts; text extraction via PyMuPDF + hardcoded reconstruction |
| Language | English |
| Wound category | Burns — acute burn injury management |

### Why 5 documents → 1 `_kept.json`
All documents are ANZBA publications covering the same clinical framework
(burn first aid → depth classification → dressing → referral). Merging them
avoids redundant embeddings while ensuring full coverage.

### What is duplicated across documents (collapsed into single content)
- Referral criteria (TBSA thresholds, special areas, mechanism) — appears in
  Minor Burns poster, Severe Burns poster, Pharmacist poster, and the PNG.
  Kept once in Chunk 2 (Referral Criteria).
- First aid "Cool for 20" rule — in Hydrogel, Pharmacist, Severe, Minor posters.
  Kept once in Chunk 1 (First Aid & Burn Classification).
- Depth classification table — in Minor and Pharmacist posters (same content).
  Kept once in Chunk 1.

### Document content map

| Document | Unique clinical content kept | Action |
|---|---|---|
| Hydrogel Consensus (2021) | Hydrogel role, limitations, hypothermia risk, first aid standards | ✅ Chunk 3 |
| Minor Burns Poster | Burn depth table (5 depths), initial dressing per depth, wound cleaning | ✅ Chunk 1 + Chunk 2 |
| Severe Burns Poster | Wound management for severe burns, cover options, severe referral criteria, transfer | ✅ Chunk 4 (severe — not primary for app) |
| Pharmacist Poster | Minor burn dressing advice (foam/silicone, no adhesive film), referral criteria | ✅ Chunk 1 + Chunk 2 |
| Referral Criteria PNG | Full referral criteria list | ✅ Chunk 2 |

### Chunk architecture (4 chunks)

| Chunk | Section | Primary source(s) |
|---|---|---|
| 1 | Burn Classification & Depth Assessment + Initial Dressing | Minor Burns Poster, Pharmacist Poster |
| 2 | Referral Criteria (when to refer burns to burn service) | Referral Criteria PNG + Minor + Severe + Pharmacist |
| 3 | First Aid, Hydrogel Use, and Wound Cover | Hydrogel Consensus + Minor + Severe |
| 4 | Minor Burn Dressing Guidance for Community/Pharmacist | Pharmacist Poster |

### Sections DROPPED

- Severe Burns ABCDE primary survey (airway, breathing, circulation, IV access,
  fluid formulae, morphine dosing) — hospital emergency department content,
  not relevant to community wound dressing recommendation for app users
- Transfer checklist for severe burns — hospital protocol
- Healed burn & scar advice — post-healing, not wound dressing
- ANZBA website/contact footers, logos
- Prevalence / general epidemiology filler text
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1 · Dependencies & paths ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Uncomment if any library is missing:
# !pip install pymupdf -q

import fitz          # PyMuPDF — native text layer extraction
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
PDF_HYDROGEL    = "../clinical_pdfs_v2/ANZBA_First_Aid_Consensus_Hydrogel_v3_Aug_2021.pdf"
PDF_MINOR       = "../clinical_pdfs_v2/ANZBA-Minor-BurnPoster-v2.pdf"
PDF_SEVERE      = "../clinical_pdfs_v2/ANZBA-Severe-BurnPoster-v2-1.pdf"
PDF_PHARMACIST  = "../clinical_pdfs_v2/ANZBA_Pharmacists_Advice_Poster_for_Burn_Injury.pdf"
# ANZBA_Referral_Criteria.png — image file, content hardcoded directly in chunk text

SOURCE_NAME   = "ANZBA_Burns_Guidelines.pdf"   # logical source key used in ChromaDB
OUT_DIR       = Path("../ingestion_output_ai")
OUT_DIR.mkdir(exist_ok=True)

MIN_CHUNK_CHARS = 60

print("✅ imports ok")
print(f"   Output directory: {OUT_DIR.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 1: Helpers
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 1 · Helpers — chunk_id generator & PDF verification
"""

# ── CELL 2 · Helpers ───────────────────────────────────────────────────────────

def make_chunk_id(source: str, section: str, idx: int = 0) -> str:
    """Deterministic 12-char MD5 hex ID — matches the pattern used by GP/AJGP/SFP/WCM/ISTAP."""
    raw = f"{source}::{section}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def clean_block_text(text: str) -> str:
    """
    Clean a raw PyMuPDF text block:
    - NFKC normalise (handles ligatures, private-use bullets \\uf0b7 → •)
    - Strip lone page-number strings
    - Collapse whitespace
    - Drop ANZBA boilerplate footer/header noise
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\uf0b7", "•")    # Wingdings bullet used in posters
    text = text.replace("\uf0fc", "✓")    # Wingdings checkmark
    text = text.strip()
    # Discard lone page numbers
    if re.fullmatch(r"\d{1,2}", text):
        return ""
    # Drop ANZBA boilerplate
    for noise in [
        "www.anzba.org.au",
        "www.anzba.org",
        "For more information go to ANZBA Website",
        "For more information go to",
        "For further information contact your local burn service",
        "Australian & New Zealand Burn Association",
        "Care • Prevention • Research • Education",
        "Pharmaceutical Society of Australia",
    ]:
        if noise in text:
            return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_page_blocks(doc: fitz.Document, pg_idx: int) -> list:
    """Return cleaned non-empty text blocks for a page, sorted by vertical position."""
    pg  = doc[pg_idx]
    raw = pg.get_text("blocks", sort=True)
    result = []
    for b in raw:
        if b[6] != 0:   # skip image blocks
            continue
        t = clean_block_text(b[4])
        if t:
            result.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": t})
    return result


# ── Verify all 4 PDFs open correctly ─────────────────────────────────────────
pdf_paths = {
    "Hydrogel":   PDF_HYDROGEL,
    "Minor":      PDF_MINOR,
    "Severe":     PDF_SEVERE,
    "Pharmacist": PDF_PHARMACIST,
}

for label, path in pdf_paths.items():
    try:
        d = fitz.open(path)
        print(f"✅ {label:12s} — {len(d)} page(s): {Path(path).name}")
        print(f"   First 8 text blocks (page 1):")
        for b in get_page_blocks(d, 0)[:8]:
            print(f"     [{b['y0']:.0f}] {repr(b['text'][:90])}")
        d.close()
    except Exception as e:
        print(f"❌ {label}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 2: Chunk 1 — Burn Classification & Initial Dressing
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 2 · Chunk 1 — Burn Classification, Depth Assessment & Initial Dressing

**Sources:** Minor Burns Poster (depth table, dressing table, wound cleaning),
Pharmacist Advice Poster (depth table cross-check, minor burn dressing advice)

**Why this is the primary retrieval chunk for dressing queries:**
The depth classification table directly maps burn depth → initial dressing →
secondary dressing → follow-up. This is what `cat_b_burns_hand` needs to retrieve.

**Content included:**
- 5 burn depth classifications with visual characteristics + capillary refill
- Initial dressing per depth (paraffin gauze, silicone, silver, moisturisers)
- Secondary dressing guidance (foam/gauze for exudate management)
- Wound cleaning protocol
- Key contraindication: no adhesive film dressings on burns (Pharmacist poster)

**Sections dropped from this chunk:**
- Severe burn ABCDE (not relevant to app users doing home dressing)
- Transfer checklist
"""

# ── CELL 3 · Chunk 1 — Burn Classification & Initial Dressing ──────────────────

# Diagnostic: print Minor Burns blocks for depth table section
doc_minor = fitz.open(PDF_MINOR)
print("Minor Burns raw blocks (all):")
for b in get_page_blocks(doc_minor, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:110])}")
doc_minor.close()
print()

CHUNK1_CLASSIFICATION = """\
ANZBA — Burn Classification, Depth Assessment and Initial Dressing
Source: ANZBA Initial Management of Minor Burns Poster;
        ANZBA Pharmacists Advice Poster for Burn Injury
Organisation: Australian & New Zealand Burn Association (ANZBA)

═══════════════════════════════════════════════════════════════
BURN WOUND CLEANING (before dressing)
═══════════════════════════════════════════════════════════════
  - Clean with 0.1% Aqueous Chlorhexidine OR Normal saline
  - Remove all foreign matter, loose and non-viable skin/tissue
  - Debride blisters if greater than 5 cm or located over joints
  - Shave hair in and around the wound to a 2 cm radius

IMPORTANT: Burns can continue to deepen over the first 3–5 days after injury.
  Reassess depth at each dressing change, especially in the first week.

═══════════════════════════════════════════════════════════════
BURN DEPTH CLASSIFICATION — Assessment & Initial Dressing
═══════════════════════════════════════════════════════════════

DEPTH 1 — EPIDERMAL BURN (Erythema / Superficial)
  Assessment:
    - Damage to epidermis only; skin is INTACT; no blisters present
    - Colour: Red / Erythema
    - Capillary refill: Brisk (< 2 seconds)
    - Sensation: Present / painful
  Healing: Heals spontaneously within 3–7 days
  Initial dressing: Simple moisturisers only
  Secondary dressing: NOT required
  Fixation: NOT required
  Follow up: Should not be required
  NOTE: Epidermal burns are NOT counted in the TBSA (body surface area) calculation.

DEPTH 2 — SUPERFICIAL DERMAL BURN (Superficial / Dermal)
  Assessment:
    - Damage to upper layer of dermis
    - Colour: Pink (pale pink); blisters present or absent
    - Capillary refill: Brisk (< 2 seconds), assessed under blister
    - Sensation: Painful
  Healing: Should heal within 7–10 days with minimal dressing requirements
  Initial dressing: Paraffin gauze; Silicone dressings;
                    Silver products if contaminated or infected
  Secondary dressing: Dermal burns produce a significant amount of exudate
    in the first 72 hours — absorbent secondary dressings such as gauze
    or foam should be considered to manage excess exudate
  Fixation: Tubular or crepe bandage, Tape
  Follow up: In 24–48 hours by GP or appropriate service

DEPTH 3 — MID DERMAL BURN
  Assessment:
    - Damage into mid dermis
    - Colour: Dark pink to red
    - Capillary refill: Sluggish (> 3 seconds)
    - Sensation: +/-  (variable / reduced)
  Healing: Should heal within 14 days.
    Deeper areas may need surgical intervention and referral.
    Possible scarring.
  Initial dressing: Silver products; Antimicrobial dressings; Silicone dressings
  Secondary dressing: Absorbent secondary dressings (gauze or foam) for exudate
  Fixation: Tubular or crepe bandage, Tape
  Follow up: In 24–48 hours by GP or appropriate service.
    Refer early to a surgeon if excision and skin grafting should be considered.

DEPTH 4 — DEEP DERMAL BURN
  Assessment:
    - Burns extend into deeper layers of dermis but not through entire dermis
    - Colour: Blotchy red/white (Pharmacist card: Blotchy Red / Cherry red / White)
    - Capillary refill: Very sluggish or absent
    - Sensation: Absent
  Healing: Generally needs surgical intervention — REFER TO SPECIALIST UNIT
    Significant scarring expected.
  Initial dressing: Silver products
  Secondary dressing: Absorbent secondary dressings for exudate
  Fixation: Tubular or crepe bandage
  Follow up: Refer early to a surgeon for excision and skin grafting.
    REFER APPROPRIATELY if wound is unhealed at 14 days.

DEPTH 5 — FULL THICKNESS BURN
  Assessment:
    - Destruction of entire dermis, sometimes with underlying tissue involvement
    - Colour: White, waxy, brown, black, or yellow (leathery appearance)
    - Capillary refill: NIL
    - Sensation: Absent
  Healing: Generally needs surgical intervention — REFER TO SPECIALIST UNIT
    Surgery and grafting required. Scarring expected.
  Initial dressing: Silver products
  Secondary dressing: Absorbent secondary dressings
  Follow up: Refer to specialist burn unit.

═══════════════════════════════════════════════════════════════
MINOR BURN DRESSING GUIDANCE (Community / Pharmacist / GP)
═══════════════════════════════════════════════════════════════
For minor burns NOT meeting referral criteria:

  ✅ USE:
    - A moist, protective dressing such as a foam dressing or silicone dressing
    - Non-stick / non-adherent dressings are preferred

  ❌ DO NOT USE:
    - Adhesive dressings directly over the burn wound itself
      (e.g. film dressings such as Opsite or Tegaderm — these are CONTRAINDICATED
      directly over burn wounds as they can cause further trauma on removal)
    - Ice — NEVER apply ice to burns (causes additional tissue damage)
    - Ointments or butter — NEVER apply to burns

  Pain management:
    - Advise simple pain relief (e.g. paracetamol, ibuprofen)

  General wound care advice:
    - Advise rest and elevation of affected limbs
    - Gentle range of motion exercises to prevent stiffness
    - Encourage patient NOT to smoke — smoking impairs wound healing
    - Explain signs of local infection requiring treatment:
        redness spreading beyond wound margins, increasing pain/warmth,
        purulent discharge, malodour, fever

  Febrile or unwell patient:
    If the patient is UNWELL OR FEBRILE, they require IMMEDIATE MEDICAL
    ATTENTION — especially in children.
    Infected burns require referral to burn service (via local ED).

(Reference: ANZBA Initial Management of Minor Burns Poster;
ANZBA Pharmacists Advice Poster for Burn Injury, www.anzba.org.au)
"""

print(f"\nChunk 1 length: {len(CHUNK1_CLASSIFICATION)} chars")
print(CHUNK1_CLASSIFICATION[:500])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 3: Chunk 2 — Referral Criteria
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 3 · Chunk 2 — Burns Referral Criteria (when to refer to Burn Service)

**Sources:** ANZBA_Referral_Criteria.png + Minor Burns Poster + Severe Burns Poster
+ Pharmacist Poster (all four contain same referral criteria — deduplicated here)

**Why this is the highest-priority safety chunk:**
The referral criteria are your `referral_required` safety check for `cat_b_burns_hand`.
Hands are explicitly listed as a special area requiring burn service referral.
This chunk must be retrieved for any burn query involving hands, face, feet,
children, electrical burns, or large TBSA.

**Content:**
- Full referral criteria list (TBSA thresholds, special areas, mechanism, person factors)
- Structured referral table (Size / Person / Area / Mechanism)
"""

# ── CELL 4 · Chunk 2 — Referral Criteria ──────────────────────────────────────

# Diagnostic: verify Severe Burns referral section
doc_severe = fitz.open(PDF_SEVERE)
print("Severe Burns raw blocks (referral section):")
for b in get_page_blocks(doc_severe, 0):
    if b["y0"] > 650:   # referral criteria are at the bottom of the poster
        print(f"  [{b['y0']:.0f}] {repr(b['text'][:110])}")
doc_severe.close()
print()

CHUNK2_REFERRAL = """\
ANZBA — Burns Referral Criteria (When to Refer to Burn Service)
Source: ANZBA Referral Criteria (www.anzba.org.au);
        ANZBA Initial Management of Minor Burns Poster;
        ANZBA Initial Management of Severe Burns Poster;
        ANZBA Pharmacists Advice Poster for Burn Injury
Organisation: Australian & New Zealand Burn Association (ANZBA)

═══════════════════════════════════════════════════════════════
RULE: If a burn meets ANY of the criteria below, it warrants
referral to a Burn Service / Burn Unit.
═══════════════════════════════════════════════════════════════

CRITERION 1 — SIZE (Total Body Surface Area, TBSA):
  • Burns greater than 10% TBSA in adults
  • Burns greater than 5% TBSA in children
  • Full Thickness burns greater than 5% TBSA (any age)

CRITERION 2 — BURNS OF SPECIAL AREAS (any size, any depth):
  These locations require specialist assessment regardless of size:
  • Face
  • Hands
  • Feet
  • Genitalia
  • Perineum
  • Major Joints
  • Circumferential limb burns
  • Circumferential chest burns

  CLINICAL NOTE ON HANDS: Burns to the hands — even small burns — warrant
  referral to a burn service because of the high functional and aesthetic
  importance of hand anatomy. Inappropriate dressing, infection, or scarring
  of hand burns can lead to long-term disability.

CRITERION 3 — MECHANISM OF INJURY:
  • Burns with inhalation injury (airway burns)
  • Electrical burns (any size — risk of deep tissue damage, cardiac arrhythmia,
    exit wound, compartment syndrome)
  • Chemical burns (any size — ongoing tissue damage, decontamination required)
  • Burns associated with major trauma (concurrent injuries)
  • Non-accidental burns (including suspected non-accidental injury in children)

CRITERION 4 — PERSON-RELATED FACTORS:
  • Burns with pre-existing illness (diabetes, immunosuppression, cardiac disease)
  • Burns in pregnant women
  • Burns at the extremes of age — young children and the elderly

CRITERION 5 — OTHER CLINICAL INDICATORS (from Pharmacist Poster):
  • Infected burns (local or spreading infection requiring specialist management)
  • Uncontrolled pain (inadequate pain control with available analgesia)

═══════════════════════════════════════════════════════════════
REFERRAL CRITERIA SUMMARY TABLE (from ANZBA Minor + Severe Posters)
═══════════════════════════════════════════════════════════════

  Size     │ >10% TBSA (adult)
           │ >5% TBSA (child)
           │ >5% TBSA full thickness (any age)
  ─────────┼──────────────────────────────────────────────────
  Person   │ Pre-existing illness
           │ Pregnancy
           │ Extremes of age
  ─────────┼──────────────────────────────────────────────────
  Area     │ Face / Hands / Feet / Perineum / Major Joints
           │ Circumferential (limb or chest)
           │ Lungs (inhalational)
  ─────────┼──────────────────────────────────────────────────
  Mechanism│ Chemical / Electrical
           │ Major Trauma
           │ Non-accidental injury (including suspected)

═══════════════════════════════════════════════════════════════
FOR SEVERE BURNS: Wound Cover During Emergency Referral / Transfer
═══════════════════════════════════════════════════════════════
  While awaiting burn service, for severe burns requiring transfer:
  - First aid: Cool running water for 20 minutes
  - Clean wound: Normal saline or 0.1% Chlorhexidine
  - Remove loose dermis or blisters >5 mm
  - Cover:
      If immediate transfer (<8 hours): Cling wrap longitudinally
      If transfer delayed: Paraffin gauze or silver dressing
      (discuss with local burn service)
  - Elevate limbs where circumferential burns present
  - Keep patient warm — 'Cool the Burn: Warm the Patient'
  - Administer tetanus immunoglobulin if required

(Reference: ANZBA Referral Criteria; ANZBA Minor and Severe Burns Posters;
www.anzba.org.au)
"""

print(f"\nChunk 2 length: {len(CHUNK2_REFERRAL)} chars")
print(CHUNK2_REFERRAL[:500])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 4: Chunk 3 — First Aid, Hydrogel & Wound Cover
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 4 · Chunk 3 — Burns First Aid, Hydrogel Use & Wound Cover

**Primary source:** ANZBA Consensus Statement — First Aid and Use of Hydrogels
(Revised August 2021) + Minor Burns Poster + Severe Burns Poster

**Why include for RAG dressing recommendation:**
- Hydrogel is a common dressing patients self-apply. The RAG system needs
  to know when hydrogel is appropriate (small burns, analgesia) and when
  it is contraindicated (large burns, hypothermia risk).
- This chunk grounds the recommendation with evidence-based ANZBA guidance
  rather than allowing the LLM to assume hydrogel = good for all burns.
- The 20-minute cool water rule is the first-line intervention before any
  dressing is applied — relevant context for any burn dressing query.

**Key clinical rules captured:**
- Cool running water 20 min = gold standard (not ice, not butter, not ointments)
- Hydrogel for small burns (<10% TBSA) before definitive dressing: acceptable
- Hydrogel contraindicated for large burns (>20% TBSA adults, >10% TBSA children)
  due to hypothermia risk
- After first aid: minimise evaporative heat loss; cling wrap for transfer
"""

# ── CELL 5 · Chunk 3 — First Aid, Hydrogel & Cover ────────────────────────────

# Diagnostic: verify Hydrogel PDF blocks
doc_hydrogel = fitz.open(PDF_HYDROGEL)
print("Hydrogel PDF raw blocks (all):")
for b in get_page_blocks(doc_hydrogel, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:110])}")
doc_hydrogel.close()
print()

CHUNK3_FIRSTAID = """\
ANZBA — Burn First Aid, Hydrogel Use and Initial Wound Cover
Source: ANZBA Consensus Statement — First Aid and the Use of Hydrogels (Revised August 2021);
        ANZBA Initial Management of Minor Burns Poster;
        ANZBA Initial Management of Severe Burns Poster
Organisation: Australian & New Zealand Burn Association (ANZBA)

═══════════════════════════════════════════════════════════════
BURN FIRST AID — GOLD STANDARD: 'COOL FOR 20'
═══════════════════════════════════════════════════════════════
  STEP 1 — STOP: Remove the person from danger
  STEP 2 — REMOVE: Remove clothing and jewellery from the burn area
  STEP 3 — COOL: Apply 20 minutes of COOL RUNNING WATER to the burn
    - Apply as soon as possible after injury
    - Most effective within 1 hour; still beneficial up to 3 hours after injury
    - Especially important for dermal thickness wounds — cooling may decrease
      burn wound progression and improve healing outcomes
  STEP 4 — COVER: Cover loosely with clean cloth or cling wrap (non-stick dressing)
  STEP 5 — SEEK: Seek medical attention as soon as possible

  ❌ NEVER:
    - Use ice on a burn — causes vasoconstriction and additional tissue injury
    - Apply ointments, butter, toothpaste, or oils to burns
    - Use very cold water (cold water risks hypothermia)

  HYPOTHERMIA RISK — 'COOL THE BURN: WARM THE PATIENT':
    - Keep the REST of the patient's body warm during burn cooling
    - Use heated blankets and warmed environments
    - Stop cooling immediately if hypothermia develops
    - Patients with extensive burns (>20% TBSA adults or >10% TBSA children)
      are at HIGH RISK of hypothermia, especially children and elderly

═══════════════════════════════════════════════════════════════
HYDROGEL DRESSINGS — EVIDENCE-BASED GUIDANCE (ANZBA 2021)
═══════════════════════════════════════════════════════════════
Evidence indicates that the cooling function of hydrogel dressing products
is NOT as effective as cool running water for first aid.

WHEN HYDROGEL / WET DRESSINGS MAY BE APPROPRIATE:
  ✅ Small burns (<10% TBSA) in the first few hours after injury:
    - May assist with cooling via evaporative heat loss (when exposed to air)
    - Can provide good analgesia for dermal thickness burns after first aid
      and before definitive dressings are applied
  ✅ When no water is available:
    - Hydrogel or other wet dressings may be used as an analgesic
    - Must be replaced by cool running water as soon as water is available
      (if within 3 hours of injury)

WHEN HYDROGEL IS CONTRAINDICATED OR SHOULD BE AVOIDED:
  ❌ Large burns (>20% TBSA in adults; >10% TBSA in children):
    - Wet dressings can cause hypothermia if left in place for prolonged periods
    - This risk is increased in the elderly, in children, and in burns exposed to air
    - Hydrogels should be AVOIDED in extensive burn injuries
  ❌ Alternative water is available within 3 hours:
    - Running water is always preferred over hydrogel for cooling/first aid

HYDROGEL COOLING MECHANISM NOTE:
  - Moist dressings rely on exposure to air for their cooling effect via
    evaporative heat loss — this is less efficient than direct water cooling
  - After first aid is completed, patients with large burns should be covered
    in dressings that MINIMISE evaporative heat loss (by excluding air from
    the wound and the outer surface of the dressing)

═══════════════════════════════════════════════════════════════
INITIAL WOUND COVER OPTIONS (after first aid, before definitive dressing)
═══════════════════════════════════════════════════════════════
  FOR TRANSFER TO HOSPITAL or BURN SERVICE:
    Immediate transfer (<8 hours from injury):
      - Plastic cling film (wrap longitudinally — do NOT wrap circumferentially
        as this restricts swelling)
    Transfer delayed (>8 hours) or definitive dressing:
      - Paraffin gauze (tulle), OR
      - Silver dressing (if contamination or infection risk)
      - Discuss with local burn service

  CLING FILM NOTES:
    - Suitable for transfer if less than 8 hours since injury
    - Do not use cling film over large areas in young children (hypothermia risk)
    - Do not wrap circumferentially — wrap longitudinally only

(Reference: ANZBA Consensus Statement — First Aid and the Use of Hydrogels,
Revised August 2021; ANZBA Minor and Severe Burns Posters; www.anzba.org.au)
"""

print(f"\nChunk 3 length: {len(CHUNK3_FIRSTAID)} chars")
print(CHUNK3_FIRSTAID[:500])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 5: Chunk 4 — Burn Depth to Dressing Quick Reference
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 5 · Chunk 4 — Burn Depth to Dressing Quick Reference + Wound Size

**Source:** Pharmacist Advice Poster (depth table cross-check), Minor Burns Poster

**Why a separate chunk from Chunk 1:**
Chunk 1 contains the full depth classification narrative. This chunk is a concise
dressing selection lookup structured for retrieval by clinical decision queries
("what dressing for mid dermal burn", "when is silver used for burns"), plus
the TBSA size estimation method.

This chunk is structured differently from Chunk 1 — it is a decision matrix
format optimised for retrieval of dressing-specific queries rather than
depth assessment queries.
"""

# ── CELL 6 · Chunk 4 — Dressing Quick Reference ───────────────────────────────

doc_pharmacist = fitz.open(PDF_PHARMACIST)
print("Pharmacist Poster raw blocks (dressing + depth section):")
for b in get_page_blocks(doc_pharmacist, 0):
    if b["y0"] > 600:
        print(f"  [{b['y0']:.0f}] {repr(b['text'][:110])}")
doc_pharmacist.close()
print()

CHUNK4_DRESSING_REFERENCE = """\
ANZBA — Burn Dressing Selection Reference & Wound Size Estimation
Source: ANZBA Pharmacists Advice Poster for Burn Injury;
        ANZBA Initial Management of Minor Burns Poster
Organisation: Australian & New Zealand Burn Association (ANZBA)

═══════════════════════════════════════════════════════════════
BURN DRESSING SELECTION BY DEPTH (quick reference)
═══════════════════════════════════════════════════════════════

EPIDERMAL (Superficial) — Red, no blisters, brisk capillary refill:
  Initial dressing  : Simple moisturisers only
  Secondary dressing: Not required
  Antimicrobials    : Not required
  Referral          : Not required; no follow-up needed

SUPERFICIAL DERMAL — Pale pink, blisters, brisk capillary refill, painful:
  Initial dressing  : Paraffin gauze (tulle); Silicone dressings
                      Silver dressings if contaminated or signs of infection
  Secondary dressing: Foam or gauze (absorbent) — significant exudate in first 72hrs
  Fixation          : Tubular or crepe bandage; Tape
  Follow-up         : 24–48 hours by GP

MID DERMAL — Dark pink to red, sluggish capillary refill, +/- sensation:
  Initial dressing  : Silver products (antimicrobial); Silicone dressings
  Secondary dressing: Foam or gauze (absorbent)
  Fixation          : Tubular or crepe bandage
  Follow-up         : 24–48 hours by GP; consider early surgical referral
  Healing time      : ~14 days; deeper areas may need surgery

DEEP DERMAL — Blotchy red/white/cherry red, absent capillary refill, absent sensation:
  Initial dressing  : Silver products
  Secondary dressing: Absorbent foam or gauze
  Follow-up         : Refer to surgeon early for excision and skin grafting
  Refer unhealed    : If unhealed at 14 days → refer to burn service

FULL THICKNESS — White/waxy/brown/black/yellow, no sensation, no capillary refill:
  Initial dressing  : Silver products
  Secondary dressing: Absorbent secondary dressing
  Follow-up         : Refer to specialist burn unit; surgical intervention required

═══════════════════════════════════════════════════════════════
MINOR BURNS: KEY DRESSING DO's AND DON'Ts
═══════════════════════════════════════════════════════════════

  ✅ USE for minor burns:
    - Foam dressing (moist, protective)
    - Silicone dressing (non-adherent, atraumatic)
    - Paraffin gauze (tulle) for superficial dermal burns

  ❌ DO NOT USE directly over the burn wound:
    - FILM DRESSINGS (e.g. Opsite, Tegaderm) — adhesive film dressings
      must NOT be applied directly over the burn wound itself
    - Ice (causes tissue damage)
    - Ointments, butter, or oil-based products

  Signs of infection in a burn requiring escalation:
    - Increasing pain, redness, warmth, swelling
    - Purulent or malodorous discharge
    - Fever or patient feeling unwell
    - Failure to heal at expected timeline
    → These require referral: infected burns are an ANZBA referral criterion

═══════════════════════════════════════════════════════════════
BURN SIZE ESTIMATION — TBSA RULE
═══════════════════════════════════════════════════════════════
  Rule of Nines (adults): Head 9%, each arm 9%, each leg 18%, trunk front 18%,
    trunk back 18%, perineum 1%
  Pharmacist 'Hand Rule': The PATIENT's own hand (including fingers) = ~1% TBSA
  Rule for assessment in community:
    - Epidermal burns (surface erythema only) are NOT counted in TBSA
    - Only partial and full thickness burns are counted
  Paediatric TBSA: Use Lund-Browder chart or contact burn service

(Reference: ANZBA Pharmacists Advice Poster for Burn Injury;
ANZBA Initial Management of Minor Burns Poster; www.anzba.org.au)
"""

print(f"\nChunk 4 length: {len(CHUNK4_DRESSING_REFERENCE)} chars")
print(CHUNK4_DRESSING_REFERENCE[:500])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 6: Assemble all chunks
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 6 · Assemble all 4 chunks into the final list
"""

# ── CELL 7 · Assemble chunk list ───────────────────────────────────────────────

def make_chunk(
    section: str,
    parent_section: str,
    text: str,
    chunk_index: int = 0,
) -> dict:
    """Build a chunk dict matching the schema used by GP / AJGP / SFP / WCM / ISTAP."""
    return {
        "chunk_id":       make_chunk_id(SOURCE_NAME, section, chunk_index),
        "source":         SOURCE_NAME,
        "section":        section,
        "parent_section": parent_section,
        "chunk_index":    chunk_index,
        "char_count":     len(text),
        "text":           text,
        "ai_summary":     text,   # overwritten by LLM enrichment cell if enabled
    }


chunks: list = []

# ── Chunk 1: Classification & Initial Dressing ────────────────────────────────
chunks.append(make_chunk(
    section        = "ANZBA Burns — Burn Classification, Depth Assessment and Initial Dressing",
    parent_section = "ANZBA Burns Assessment and Treatment",
    text           = CHUNK1_CLASSIFICATION,
))

# ── Chunk 2: Referral Criteria ────────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ANZBA Burns — Referral Criteria (When to Refer to Burn Service)",
    parent_section = "ANZBA Burns Assessment and Treatment",
    text           = CHUNK2_REFERRAL,
))

# ── Chunk 3: First Aid, Hydrogel & Cover ──────────────────────────────────────
chunks.append(make_chunk(
    section        = "ANZBA Burns — First Aid, Hydrogel Use and Initial Wound Cover",
    parent_section = "ANZBA Burns First Aid",
    text           = CHUNK3_FIRSTAID,
))

# ── Chunk 4: Dressing Quick Reference ────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ANZBA Burns — Dressing Selection Reference and Wound Size Estimation",
    parent_section = "ANZBA Burns Assessment and Treatment",
    text           = CHUNK4_DRESSING_REFERENCE,
))

print(f"Total chunks assembled: {len(chunks)}")
for i, c in enumerate(chunks, 1):
    print(f"  {i}. '{c['section'][:65]:65s}' chars={c['char_count']:5d}")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 7: Quality validation
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 7 · Quality validation — char counts, chunk_id uniqueness, keyword coverage
"""

# ── CELL 8 · Quality checks ────────────────────────────────────────────────────

print("═" * 70)
print("QUALITY VALIDATION")
print("═" * 70)

# ── 1. All chunks have sufficient content ─────────────────────────────────────
short_chunks = [c for c in chunks if c["char_count"] < MIN_CHUNK_CHARS]
if short_chunks:
    print(f"❌ {len(short_chunks)} chunk(s) below minimum {MIN_CHUNK_CHARS} chars:")
    for c in short_chunks:
        print(f"   {c['section']}: {c['char_count']} chars")
else:
    print(f"✅ All chunks above minimum {MIN_CHUNK_CHARS} chars")

# ── 2. chunk_ids are unique ───────────────────────────────────────────────────
ids = [c["chunk_id"] for c in chunks]
if len(ids) == len(set(ids)):
    print(f"✅ All {len(ids)} chunk_ids are unique")
else:
    dupes = [id_ for id_ in ids if ids.count(id_) > 1]
    print(f"❌ Duplicate chunk_ids found: {set(dupes)}")

# ── 3. No empty text or ai_summary ───────────────────────────────────────────
empty = [c for c in chunks if not c["text"].strip() or not c["ai_summary"].strip()]
if empty:
    print(f"❌ {len(empty)} chunk(s) with empty text or ai_summary")
else:
    print(f"✅ All chunks have non-empty text and ai_summary")

# ── 4. Totals ──────────────────────────────────────────────────────────────────
total_chars = sum(c["char_count"] for c in chunks)
print(f"\n   Total characters across {len(chunks)} chunks: {total_chars:,}")
print(f"   Average chars per chunk:                {total_chars // len(chunks):,}")
print(f"   Source name key:                         {SOURCE_NAME}")

# ── 5. Clinical keyword coverage ──────────────────────────────────────────────
combined = "\n".join(c["text"] for c in chunks).lower()
keywords = [
    ("silver",                  "silver dressings for burns"),
    ("silicone",                "silicone dressings for minor burns"),
    ("paraffin",                "paraffin gauze for superficial dermal"),
    ("foam",                    "foam dressing for minor burns"),
    ("film dressing",           "film dressing contraindication"),
    ("opsite",                  "Opsite film dressing contraindicated"),
    ("tegaderm",                "Tegaderm film dressing contraindicated"),
    ("hydrogel",                "hydrogel guidance / contraindication"),
    ("hypothermia",             "hypothermia risk with large burns"),
    ("ice",                     "ice contraindicated on burns"),
    ("20 minutes",              "20 min cool water gold standard"),
    ("hands",                   "hands = special area referral"),
    ("10% tbsa",                "TBSA threshold for adults"),
    ("5% tbsa",                 "TBSA threshold for children"),
    ("electrical",              "electrical burns referral criteria"),
    ("chemical",                "chemical burns referral criteria"),
    ("full thickness",          "full thickness classification"),
    ("mid dermal",              "mid dermal classification"),
    ("referral",                "referral criteria"),
    ("silver allergy",          "silver allergy contraindication"),
    ("infected",                "infected burns require referral"),
    ("circumferential",         "circumferential burns referral"),
]

print("\n   Clinical keyword coverage:")
all_present = True
for kw, desc in keywords:
    found = kw.lower() in combined
    status = "✅" if found else "❌ MISSING"
    print(f"   {status} '{kw}' — {desc}")
    if not found:
        all_present = False

if all_present:
    print("\n✅ All clinical keywords present — chunks are clinically complete")
else:
    print("\n⚠️  Some keywords missing — review chunk content above")

# Note: silver allergy is not explicitly in ANZBA documents — it is covered
# by the general caution in SFP/WCM chunks. ANZBA doesn't explicitly list it.
# This is acceptable — the silver allergy contraindication for skin tears
# is covered by ISTAP chunks; for burns, silver products are standard.


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 8: Spot-check
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 8 · Spot-check individual chunks
"""

# ── CELL 9 · Spot-check ────────────────────────────────────────────────────────

def preview_chunk(idx: int):
    c = chunks[idx]
    print(f"\n{'─' * 65}")
    print(f"[{idx}] chunk_id    : {c['chunk_id']}")
    print(f"     section      : {c['section']}")
    print(f"     parent       : {c['parent_section']}")
    print(f"     chars        : {c['char_count']}")
    print("TEXT (first 700 chars):")
    print(c["text"][:700])
    if len(c["text"]) > 700:
        print("... [truncated]")

# Spot-check: Referral (most critical for safety checker) and Dressing Reference
for idx in [0, 1, 2, 3]:
    preview_chunk(idx)


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 9: (Optional) LLM ai_summary enrichment
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 9 · (Optional) LLM `ai_summary` enrichment

Set `ENABLE_AI_SUMMARY = True` when your OpenAI API key is available.
The ai_summary is what gets embedded into ChromaDB as `page_content`.
The raw `text` is stored as metadata and used for evidence display in the app.
"""

# ── CELL 10 · LLM ai_summary ──────────────────────────────────────────────────

ENABLE_AI_SUMMARY = False   # ← set True when OpenAI key is available

if ENABLE_AI_SUMMARY:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    SYSTEM_PROMPT = (
        "You are a medical summarisation assistant. "
        "Rewrite the following burn wound care guideline text as a clear, complete, "
        "self-contained clinical summary suitable for retrieval-augmented generation. "
        "Preserve ALL clinical facts including: burn depth classifications, dressing names, "
        "TBSA thresholds, referral criteria (especially special areas: hands, face, feet, "
        "genitalia, perineum, major joints, circumferential burns), contraindications "
        "(film dressings, ice, hydrogel in large burns), and the 20-minute cool water rule. "
        "Return only the summary text — no preamble."
    )

    print(f"Running AI summaries for {len(chunks)} chunks...")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {c['section'][:60]}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": c["text"]},
            ],
        )
        c["ai_summary"] = resp.choices[0].message.content.strip()
    print("✅ AI summaries done")
else:
    print("ℹ️  AI summary disabled — ai_summary == text (raw chunk)")
    print("   Set ENABLE_AI_SUMMARY = True to enrich with GPT-4o-mini")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 10: Export JSON
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 10 · Export `ANZBA_burns_kept.json`
"""

# ── CELL 11 · Export ChromaDB-ready JSON ──────────────────────────────────────
# Format mirrors GP/AJGP/SFP/WCM/ISTAP _kept.json for uniform loading
# in ingestion_full.ipynb.

output = {
    "meta": {
        "total_chunks":    len(chunks),
        "kept_count":      len(chunks),
        "ai_summarised":   sum(1 for c in chunks if c["ai_summary"] != c["text"]),
        "extraction":      (
            "Hardcoded text reconstruction from 4 ANZBA PDFs + 1 PNG image. "
            "PyMuPDF used for block-level verification. Content from all 5 documents "
            "is merged and deduplicated — referral criteria and first aid rules that "
            "appear in multiple documents are retained once."
        ),
        "chunking":        "manual section-aware — one chunk per clinical domain",
        "source_pdfs": [
            "ANZBA_First_Aid_Consensus_Hydrogel_v3_Aug_2021.pdf",
            "ANZBA-Minor-BurnPoster-v2.pdf",
            "ANZBA-Severe-BurnPoster-v2-1.pdf",
            "ANZBA_Pharmacists_Advice_Poster_for_Burn_Injury.pdf",
            "ANZBA_Referral_Criteria.png",
        ],
        "sections_used": [
            "Burn depth classification table (5 depths)",
            "Initial dressing per depth",
            "Wound cleaning protocol",
            "Minor burn dressing guidance (foam/silicone, no film dressings)",
            "Referral criteria (TBSA, special areas, mechanism, person factors)",
            "Burns first aid 'Cool for 20' rule",
            "Hydrogel guidance and contraindications",
            "Wound cover for transfer (cling wrap, paraffin, silver)",
            "Dressing selection quick reference",
        ],
        "sections_dropped": [
            "Severe burn ABCDE primary survey (airway, breathing, circulation, IV access, "
            "fluid Parkland formula, morphine dosing) — hospital emergency content only",
            "Transfer checklist (hospital protocol)",
            "Healed burn and scar advice (post-healing)",
            "ANZBA contact details, website footers, logos",
        ],
        "chunk_params": {
            "min_characters": MIN_CHUNK_CHARS,
        },
        "wound_category":  "burns",
        "note": (
            "Use ai_summary field for ChromaDB page_content and RAGAS reference_contexts. "
            "wound_category=burns enables metadata filtering in v4 sub-query A. "
            "All 5 ANZBA documents are merged under SOURCE_NAME='ANZBA_Burns_Guidelines.pdf'. "
            "Chunk 2 (Referral Criteria) is critical for the referral_required safety check — "
            "especially for hand burns (cat_b_burns_hand). "
            "Chunk 1 (Classification) drives dressing selection by depth. "
            "Chunk 3 (Hydrogel) provides evidence-based guidance on hydrogel use/avoidance."
        ),
    },
    "kept_ids_by_source": {
        SOURCE_NAME: [c["chunk_id"] for c in chunks]
    },
    "kept_chunks": [
        {
            "chunk_id":       c["chunk_id"],
            "source":         c["source"],
            "section":        c["section"],
            "parent_section": c["parent_section"],
            "chunk_index":    c["chunk_index"],
            "char_count":     c["char_count"],
            "text":           c["text"],
            "ai_summary":     c["ai_summary"],
        }
        for c in chunks
    ]
}

out_path = OUT_DIR / "ANZBA_burns_kept.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Exported {len(chunks)} chunks → {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")


# ── CELL 12 · Final summary table ─────────────────────────────────────────────

with open(OUT_DIR / "ANZBA_burns_kept.json") as f:
    exported = json.load(f)

print("═" * 72)
print("INGESTION COMPLETE — ANZBA Burns Guidelines")
print("═" * 72)

hdr = f"{'#':>3}  {'Chunk ID':14}  {'Section':55}  {'Chars':>5}  {'AI?':8}"
print(hdr)
print("-" * len(hdr))
for i, c in enumerate(exported["kept_chunks"], 1):
    ai  = "yes" if c["ai_summary"] != c["text"] else "no (raw)"
    sec = c["section"][:55]
    print(f"{i:3d}  {c['chunk_id']:14}  {sec:55s}  {c['char_count']:5d}  {ai}")

print()
print(f"Output JSON     : {OUT_DIR / 'ANZBA_burns_kept.json'}")
print(f"Source key      : {SOURCE_NAME}")
print(f"wound_category  : burns")
print()
print("Next steps:")
print("  1) Review chunk text — edit CHUNK1–CHUNK4 constants if content needs adjustment")
print("  2) Set ENABLE_AI_SUMMARY = True (Cell 10) to enrich ai_summary with GPT-4o-mini")
print("  3) Add ANZBA_burns_kept.json to ingestion_full.ipynb's CHUNK_FILES dict:")
print(f"       '{SOURCE_NAME}': '../ingestion_output_ai/ANZBA_burns_kept.json',")
print("  4) Add ANZBA to GUIDELINE_METADATA in ingestion_full.ipynb:")
print(f"       '{SOURCE_NAME}': {{")
print("           'guideline_type': 'specialist_burns',")
print("           'authority':      'ANZBA',")
print("           'year':           '2021',")
print("           'focus':          'burns_first_aid_classification_dressing_referral',")
print("           'wound_category': 'burns',")
print("       }")
print("  5) Re-run ingestion_full.ipynb to rebuild db_wound_care_v4")
print()
print("RAGAS testset updates needed:")
print("  • cat_b_burns_hand: update reference_contexts to include ANZBA chunks")
print("    referral_required should be True (hands = ANZBA special area)")
print()
print("CHUNK ID MAP (add to wound_testset_builder_v3.py):")
print("─" * 55)
alias_map = {
    0: "ANZBA_CLASSIFICATION",
    1: "ANZBA_REFERRAL",
    2: "ANZBA_FIRSTAID_HYDROGEL",
    3: "ANZBA_DRESSING_REFERENCE",
}
for i, c in enumerate(exported["kept_chunks"]):
    print(f'{alias_map.get(i, "ANZBA_CHUNK_" + str(i)):30s} = "{c["chunk_id"]}"')

# RAGAS reference-context lookup map
ref_ctx_by_section = defaultdict(list)
for c in exported["kept_chunks"]:
    ref_ctx_by_section[c["section"]].append(c["ai_summary"])
print(f"\nRAGAS reference_context lookup ready — {len(ref_ctx_by_section)} sections")
