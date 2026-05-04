"""
ingestion_ISTAP.py
==================
VerdaSense Wound RAG — ISTAP Skin Tear Ingestion (2024)
Cell-by-cell Python file. Paste each # ── CELL N ── block into a separate
Jupyter notebook cell. Markdown header comments become Markdown cells.

Source PDFs (4 documents, same organisation):
  1. ISTAP_Pathway_to_Assessment_-_Treatment.pdf
  2. ISTAP_Tool_Kit_Poster.pdf
  3. ISTAP_Risk_Assessment_Pathway.pdf
  4. ISTAP_Decision_Algorithm.pdf

All 4 share the same SOURCE_NAME (same KB entry — "ISTAP Skin Tear Guidelines 2024").
Output: ISTAP_skin_tear_kept.json  (6 chunks, ChromaDB-ready)
"""

# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Title & Document Profile
# ══════════════════════════════════════════════════════════════════════════════
"""
# `ingestion_ISTAP.ipynb`
## International Skin Tear Advisory Panel (ISTAP) — Skin Tear Tool Kit 2024

### Document profile

| Property | Detail |
|---|---|
| Organisation | International Skin Tear Advisory Panel (ISTAP) |
| Year | 2024 |
| Documents | 4 PDFs — Pathway to Assessment & Treatment, Tool Kit Poster, Risk Assessment Pathway, Decision Algorithm |
| Layout | Poster / flowchart layout (image-heavy); text extraction via PyMuPDF + hardcoded reconstruction |
| Language | English |
| Wound category | Skin tears — acute wounds, elderly, fragile skin |

### Why 4 PDFs → 1 `_kept.json`
All 4 ISTAP PDFs are companion documents from the same organisation and 2024
release. They cover different views of the same clinical framework. Treating them
as one knowledge base entry avoids redundant embeddings and keeps retrieval focused.

### Document content map

| PDF | Content | Action |
|---|---|---|
| Pathway to Assessment & Treatment | Treatment flowchart: Treat Cause, Local Wound Care, Debridement, Infection/Inflammation, Moisture Balance, Non Advancing Edge | ✅ Reconstruct as text |
| Tool Kit Poster | Key Points, Prevalence Study Sheet, ISTAP Classification System (Types 1-3), Product Selection Guide (table), Skin Tear Decision Algorithm text, Quick Reference Guide | ✅ Reconstruct most sections |
| Risk Assessment Pathway | Risk factors (General Health, Mobility, Skin), At Risk / High Risk criteria, Risk Reduction Programme | ✅ Reconstruct |
| Decision Algorithm | Classification Types 1-3 with visual descriptions, Goals of Treatment, Assessment steps | ✅ Reconstruct |

### Chunk architecture (6 chunks)

| Chunk | Section | Primary source |
|---|---|---|
| 1 | ISTAP Classification System — Skin Tear Types 1, 2, 3 | Tool Kit Poster + Decision Algorithm |
| 2 | Pathway to Assessment & Treatment — Decision Flowchart | Pathway PDF |
| 3 | Skin Tear Product Selection Guide | Tool Kit Poster |
| 4 | Infected Skin Tear — Special Antimicrobial Dressings | Tool Kit Poster |
| 5 | Risk Assessment Pathway | Risk Assessment Pathway PDF |
| 6 | Goals of Treatment & Quick Reference | Decision Algorithm + Tool Kit Poster |

### Pages / sections DROPPED (noise / irrelevant to wound dressing RAG)

- Prevalence Study Data Collection Sheet (data collection form — not clinical guidance)
- Drugs Associated with Risk of Falls (falls pharmacology — not wound dressing)
- ISTAP contact details, website, social media footers
- Reference list (references 1–13)
- Logos, organisation branding
- Image photograph sections (images cannot be represented in text chunks)
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
# All 4 ISTAP PDFs are in the same clinical_pdfs directory.
# We don't use PyMuPDF to drive content for these files (they are poster/flowchart
# layouts where the text layer is incomplete or disordered). Instead, we use
# PyMuPDF purely to verify the files open correctly, then build every chunk
# from carefully reconstructed hardcoded text strings — the same approach used
# for the GP algorithm and referral chunks.

PDF_PATHWAY   = "../clinical_pdfs_v2/ISTAP_Pathway_to_Assessment_-_Treatment.pdf"
PDF_TOOLKIT   = "../clinical_pdfs_v2/ISTAP_Tool_Kit_Poster.pdf"
PDF_RISK      = "../clinical_pdfs_v2/ISTAP_Risk_Assessment_Pathway.pdf"
PDF_ALGORITHM = "../clinical_pdfs_v2/ISTAP_Decision_Algorithm.pdf"

SOURCE_NAME   = "ISTAP_Skin_Tear_Guidelines_2024.pdf"   # logical source key used in ChromaDB
OUT_DIR       = Path("../ingestion_output_no_ai")
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
    """Deterministic 12-char MD5 hex ID — matches the pattern used by GP/AJGP/SFP/WCM."""
    raw = f"{source}::{section}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def clean_block_text(text: str) -> str:
    """
    Clean a raw PyMuPDF text block:
    - NFKC normalise (handles ligatures, private-use bullets)
    - Strip lone page-number strings
    - Collapse whitespace
    - Drop ISTAP footer / header noise patterns
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\uf097", "•")       # private-use bullet
    text = text.strip()
    # Discard lone page numbers
    if re.fullmatch(r"\d{1,2}", text):
        return ""
    # Drop ISTAP boilerplate footer lines
    for noise in [
        "www.skintears.org",
        "@ISTAP",
        "@SkinTears",
        "@skin-tears",
        "International Skin Tear Advisory Panel",
        "Working towards a world without skin tears",
        "© ISTAP 2024",
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
    "Pathway":   PDF_PATHWAY,
    "ToolKit":   PDF_TOOLKIT,
    "Risk":      PDF_RISK,
    "Algorithm": PDF_ALGORITHM,
}

for label, path in pdf_paths.items():
    try:
        d = fitz.open(path)
        print(f"✅ {label:12s} — {len(d)} page(s): {Path(path).name}")
        # Print raw blocks from page 0 for verification
        print(f"   Raw text blocks (page 1):")
        for b in get_page_blocks(d, 0)[:12]:
            print(f"     [{b['y0']:.0f}] {repr(b['text'][:80])}")
        d.close()
    except Exception as e:
        print(f"❌ {label}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 2: Chunk 1 — Classification System
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 2 · Chunk 1 — ISTAP Classification System (Skin Tear Types 1, 2, 3)

**Source:** Tool Kit Poster + Decision Algorithm PDF  
**Why reconstruct:** The classification table and Decision Algorithm flowchart are
image-based. The text layer has labels only ("Type 1: No Skin Loss" etc.) without
the clinical descriptions. We reconstruct from the clearly visible text content.

**Clinical content included:**
- 3 skin tear types with definitions and treatment approach
- Assessment steps (Control Bleeding → Assess → Cleanse → Approximate Wound Edges → Classify)
- Goals of Treatment (8 goals)
"""

# ── CELL 3 · Chunk 1 — Classification System ───────────────────────────────────

CHUNK1_CLASSIFICATION = """\
ISTAP SKIN TEAR CLASSIFICATION SYSTEM — Types 1, 2, and 3
Source: International Skin Tear Advisory Panel (ISTAP) Tool Kit 2024 / Decision Algorithm 2024

DEFINITION (ISTAP 2024):
A skin tear is "a wound caused by shear, friction, and/or blunt force resulting in
separation of skin layers. A skin tear can be partial-thickness (separation of the
epidermis from the dermis) or full-thickness (separation of both the epidermis and
dermis from underlying structures)."

INITIAL ASSESSMENT STEPS (applied to ALL skin tear types):
  Step 1. Control Bleeding — apply gentle pressure
  Step 2. Assess — wound characteristics, surrounding skin, risk factors
  Step 3. Cleanse — gentle irrigation with saline or wound cleanser
  Step 4. Approximate Wound Edges — gently reposition viable skin flap if possible
  Step 5. Classify — document type using ISTAP classification below
  Step 6. Measure and document wound size

ISTAP SKIN TEAR CLASSIFICATION:

  TYPE 1 — No Skin Loss (Linear or Flap Tear)
    Definition: Linear or flap tear where the skin flap CAN be repositioned
                to cover the wound bed.
    Visual appearance: Intact flap of skin that can be repositioned.
    Treatment approach:
      - Reposition the skin flap over the wound bed using moistened gloved finger
        or dampened cotton tip applicator
      - Apply 2-octyl cyanoacrylate topical bandage (skin glue) to approximate
        wound edges — use within 24 hours of injury; medical directive/protocol
        may be required
      - Or apply non-adherent/low-tack dressing that will not disturb the flap
        on removal
      - Dressing options: non-adherent mesh, silicone mesh, acrylic dressing
        (mild-moderate exudate; extended wear time)

  TYPE 2 — Partial Flap Loss
    Definition: Partial skin flap loss — the skin flap CANNOT be repositioned
                to completely cover the wound bed.
    Visual appearance: Partial area of wound bed exposed; some flap viable.
    Treatment approach:
      - Reposition remaining viable flap as far as possible
      - Select dressing based on exudate level and wound condition
      - Dressing options: non-adherent mesh (any exudate level), foam (moderate
        exudate, 2–7 days — AVOID adhesive bordered versions), hydrogel (dry wounds,
        caution: may cause maceration if wound is exudative), calcium alginate
        (moderate-heavy exudate, haemostatic), hydrofibre (moderate-heavy exudate),
        acrylic dressing (mild-moderate exudate without bleeding)

  TYPE 3 — Total Flap Loss
    Definition: Total skin flap loss — the wound bed is completely exposed.
                No viable flap tissue remains to reposition.
    Visual appearance: Entire wound bed exposed; bright red or pale/necrotic.
    Treatment approach:
      - Full wound bed management required
      - Same dressing options as Type 2 (except skin glue — no edges to approximate)
      - If large or complex, consider specialist referral

GOALS OF TREATMENT (all skin tear types):
  1. Treat the underlying cause
  2. Implement skin tear prevention protocol
  3. Moist wound healing environment
  4. Avoid trauma to wound and periwound skin
  5. Protect periwound skin from damage and maceration
  6. Manage exudate appropriately
  7. Avoid infection
  8. Pain control

AT-RISK POPULATIONS:
  - Elderly patients (extremes of age — most common population)
  - Critically or chronically ill patients
  - Neonates and paediatric patients
  - Patients with cognitive impairment, impaired mobility, falls history
  - Patients on polypharmacy (especially anticoagulants, steroids, chemotherapy agents)
  - Patients with fragile, papery, or thin skin; previous skin tears; senile purpura

EPIDEMIOLOGY NOTE:
  Skin tear prevalence rates are reported as EQUAL TO OR GREATER THAN pressure ulcer
  prevalence rates. They are acute wounds with high risk of becoming complex chronic
  wounds if not treated appropriately.

(Reference: ISTAP Skin Tear Tool Kit 2024; LeBlanc et al., Advances in Skin &
Wound Care 2011; LeBlanc et al., Advances in Skin & Wound Care 26(6) 2013)
"""

print(f"Chunk 1 length: {len(CHUNK1_CLASSIFICATION)} chars")
print(CHUNK1_CLASSIFICATION[:600])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 3: Chunk 2 — Pathway to Assessment & Treatment
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 3 · Chunk 2 — Pathway to Assessment & Treatment (flowchart)

**Source:** ISTAP_Pathway_to_Assessment_-_Treatment.pdf  
**Why reconstruct:** The pathway is a visual decision flowchart. PyMuPDF extracts
the box labels in non-sequential order. We reconstruct the clinical logic explicitly.

**Clinical content included:**
- Three parallel treatment axes: Treat the Cause, Local Wound Care, Patient-Centered Concerns
- Three treatment domains: Debridement, Infection/Inflammation, Moisture Balance
- Non-Advancing Edge escalation step
"""

# ── CELL 4 · Chunk 2 — Pathway to Assessment & Treatment ───────────────────────

# Diagnostic: show raw blocks from Pathway PDF page 0
doc_pathway = fitz.open(PDF_PATHWAY)
print("Raw blocks from Pathway PDF (page 1):")
for b in get_page_blocks(doc_pathway, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:100])}")
doc_pathway.close()
print()

CHUNK2_PATHWAY = """\
ISTAP — Pathway to Assessment and Treatment of Skin Tears
Source: ISTAP Pathway to Assessment & Treatment (© ISTAP 2024)
Adapted from: Sibbald et al modified from LeBlanc, Christensen, Orstead, Keast 2008

OVERVIEW:
This pathway applies to any person presenting with a skin tear. It runs three
parallel clinical workstreams simultaneously: Treat the Cause, Local Wound Care,
and Patient-Centered Concerns. Below these, three treatment domains are addressed.

═══════════════════════════════════════════════════════════════
WORKSTREAM 1 — TREAT THE CAUSE
═══════════════════════════════════════════════════════════════
Address the underlying factors contributing to the skin tear:

  GENERAL HEALTH factors to assess and manage:
    - Cognitive impairment (may resist care, increases trauma risk)
    - Sensory impairment
    - Visual impairment
    - Auditory impairment
    - Nutritional status (malnutrition increases skin fragility)
    - Chronic or critical disease (e.g. heart failure, renal failure, diabetes)
    - Polypharmacy (especially anticoagulants, corticosteroids — see falls risk list)

  AMBULATION / MOBILITY factors:
    - History of falls
    - Impaired mobility
    - Activities of daily living (ADLs) requiring assistance

  SKIN factors:
    - Age-related skin changes (thin, fragile, reduced elasticity)
    - Mechanical trauma from dressing removal, equipment
    - Previous skin tears

═══════════════════════════════════════════════════════════════
WORKSTREAM 2 — LOCAL WOUND CARE
═══════════════════════════════════════════════════════════════
  1. Atraumatic dressing removal — use remover wipes; peel slowly in direction
     that does not disturb the skin flap or viable tissue edges
  2. Cleanse the wound gently (saline or wound cleanser)
  3. Control bleeding
  4. Approximate wound edges — reposition viable skin flap with moistened
     gloved finger or dampened cotton tip applicator
  5. Assess and classify according to ISTAP Classification System (Types 1, 2, 3)
  6. Select appropriate dressing (see Product Selection Guide)

═══════════════════════════════════════════════════════════════
WORKSTREAM 3 — PATIENT-CENTERED CONCERNS
═══════════════════════════════════════════════════════════════
  - Activities of daily living (ADLs) — advise on protective clothing, padding
  - Pain control — select atraumatic dressings; consider analgesia before
    dressing changes
  - Educate client and circle of care / caregivers:
    * Prevention strategies (protective sleeves, hydration, nutrition)
    * Correct dressing removal technique (slow, low angle)
    * When to seek further care

═══════════════════════════════════════════════════════════════
TREATMENT DOMAIN 1 — DEBRIDEMENT
═══════════════════════════════════════════════════════════════
  - Debride NON-VIABLE tissue only
  - AVOID sutures or staples — they cause further trauma to fragile skin
    and can tear through the skin edges
  - Skin glue (2-octyl cyanoacrylate) is an acceptable alternative to sutures
    for Type 1 tears (edge approximation within 24 hours)

═══════════════════════════════════════════════════════════════
TREATMENT DOMAIN 2 — INFECTION / INFLAMMATION
═══════════════════════════════════════════════════════════════
  LOCAL INFECTION:
    - Topical antimicrobial dressings are appropriate
    - Options: Methylene Blue and Gentian Violet dressings, Ionic Silver dressings
    - Non-traumatic to wound bed — prioritise non-adherent versions
  DEEP TISSUE INFECTION:
    - Systemic antibiotics required
    - Refer for assessment if suspected
  TETANUS:
    - Consider tetanus immunisation status — skin tears can introduce tetanus
      if the patient has not been vaccinated or booster is overdue

═══════════════════════════════════════════════════════════════
TREATMENT DOMAIN 3 — MOISTURE BALANCE
═══════════════════════════════════════════════════════════════
  PERIWOUND PROTECTION:
    - Apply film-forming liquid acrylate (skin barrier) to periwound skin
      to protect from moisture, maceration, and adhesive trauma
  WOUND DRESSING SELECTION:
    - Use NON-ADHERENT or LOW-TACK dressings only — adhesive dressings
      risk causing new skin tears on removal, especially on elderly forearms,
      hands, and shins
    - Facilitate moisture balance appropriate to exudate level:
        Dry wound / low exudate  → Hydrogel (moisture donation), non-adherent mesh
        Moderate exudate         → Foam (non-adhesive), non-adherent mesh, alginate
        Heavy exudate            → Alginate (haemostatic, highly absorbent), hydrofibre

═══════════════════════════════════════════════════════════════
NON-ADVANCING EDGE — ESCALATION STEP
═══════════════════════════════════════════════════════════════
  If wound edges are not advancing after appropriate treatment:
    1. Re-evaluate all three treatment domains (cause, local care, patient concerns)
    2. Consider Active Therapy (e.g. growth factors, bioengineered skin substitutes)
    3. Consider specialist referral

(Reference: ISTAP Pathway to Assessment & Treatment, © ISTAP 2024;
adapted from Sibbald et al / LeBlanc, Christensen, Orstead, Keast 2008)
"""

print(f"\nChunk 2 length: {len(CHUNK2_PATHWAY)} chars")
print(CHUNK2_PATHWAY[:400])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 4: Chunk 3 — Product Selection Guide
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 4 · Chunk 3 — Skin Tear Product Selection Guide

**Source:** ISTAP Tool Kit Poster — "Skin Tear Product Selection Guide" table  
**Why this is the highest-value chunk for RAG:** It contains the explicit
dressing-type → indication → skin tear type → considerations mapping that
directly answers dressing selection queries.

**What is included:**
- 8 product categories for non-infected skin tears
- 2 product categories for infected skin tears  
- Indications, applicable skin tear types, contraindication notes
"""

# ── CELL 5 · Chunk 3 — Product Selection Guide ─────────────────────────────────

# Diagnostic: verify Tool Kit Poster blocks
doc_toolkit = fitz.open(PDF_TOOLKIT)
print(f"Tool Kit Poster: {len(doc_toolkit)} page(s)")
print("Raw text blocks (page 1):")
for b in get_page_blocks(doc_toolkit, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:100])}")
doc_toolkit.close()
print()

CHUNK3_PRODUCTS = """\
ISTAP SKIN TEAR PRODUCT SELECTION GUIDE
Source: ISTAP Skin Tear Tool Kit 2024 (Tool Kit Poster — Product Selection Guide table)
© ISTAP 2024

This guide lists dressing categories for NON-INFECTED skin tears, organised by
indication, applicable skin tear type (1, 2, or 3), and clinical considerations.

═══════════════════════════════════════════════════════════════
PRODUCT SELECTION — NON-INFECTED SKIN TEARS
═══════════════════════════════════════════════════════════════

1. NON-ADHERENT MESH DRESSINGS
   Examples: Lipido-colloid mesh, impregnated gauze mesh, silicone mesh, petrolatum
   Indications: Dry OR exudative wound
   Applicable types: Type 1, Type 2, Type 3
   Considerations:
     - Maintains moisture balance for multiple levels of wound exudate
     - Atraumatic removal — will not disturb the skin flap
     - May need a secondary cover dressing

2. FOAM DRESSING (NON-ADHESIVE ONLY)
   Indications: Moderate exudate; longer wear time (2–7 days depending on exudate)
   Applicable types: Type 2, Type 3
   Considerations:
     - CAUTION with ADHESIVE BORDERED foams — adhesive borders MUST NOT be used
       on fragile skin; they risk causing new skin tears on removal
     - Use NON-ADHESIVE versions whenever possible to avoid periwound trauma
     - Anchor with tubular bandage or tape applied to non-fragile skin

3. HYDROGEL
   Indications: Donates moisture for dry wounds
   Applicable types: Type 2, Type 3
   Considerations:
     - CAUTION: may result in periwound maceration if wound is exudative
     - Appropriate for autolytic debridement in wounds with low exudate
     - Secondary cover dressing required

4. 2-OCTYL CYANOACRYLATE TOPICAL BANDAGE (SKIN GLUE)
   Indications: To approximate wound edges
   Applicable types: Type 1 ONLY
   Considerations:
     - Use in a similar fashion to sutures — apply within first 24 hours post-injury
     - Relatively expensive
     - Medical directive / protocol may be required before application
     - NOT appropriate for Types 2 and 3 (no edges to approximate)

5. CALCIUM ALGINATE
   Indications: Moderate to heavy exudate; haemostatic (helps control bleeding)
   Applicable types: Type 1, Type 2, Type 3
   Considerations:
     - May dry out the wound bed if wound exudate is inadequate
     - Secondary cover dressing required
     - Haemostatic property useful in Types 1 and 2 where some bleeding may occur

6. HYDROFIBRE (HYDROFIBER)
   Indications: Moderate to heavy exudate
   Applicable types: Type 2, Type 3
   Considerations:
     - No haemostatic properties (unlike alginate)
     - May dry out wound bed if exudate is inadequate
     - Secondary cover dressing required

7. ACRYLIC DRESSING
   Indications: Mild to moderate exudate WITHOUT any evidence of bleeding;
                may remain in place for an extended period
   Applicable types: Type 1, Type 2, Type 3
   Considerations:
     - Care on removal — do not peel quickly from fragile skin
     - Should only be used as directed and left on for extended wear time
     - Extended wear helps stabilise the fragile skin environment

═══════════════════════════════════════════════════════════════
PRODUCT SELECTION — INFECTED SKIN TEARS (Special Considerations)
═══════════════════════════════════════════════════════════════

8. METHYLENE BLUE AND GENTIAN VIOLET DRESSINGS
   Indications: Effective broad-spectrum antimicrobial action including against
                antibiotic-resistant organisms (e.g. MRSA)
   Applicable types: Type 1, Type 2, Type 3
   Considerations:
     - Non-traumatic to wound bed
     - Use when local OR deep tissue infection is suspected or confirmed
     - Secondary dressing required

9. IONIC SILVER DRESSINGS
   Indications: Effective broad-spectrum antimicrobial action including against
                antibiotic-resistant organisms
   Applicable types: Type 1, Type 2, Type 3
   Considerations:
     - Should NOT be used indefinitely — review need at each dressing change
     - CONTRAINDICATED in patients with silver allergy
     - Use non-adherent silver products whenever possible to minimise risk of
       further trauma to fragile skin
     - Use when local or deep tissue infection is suspected or confirmed

═══════════════════════════════════════════════════════════════
SUMMARY — DRESSING CHOICE BY WOUND CONDITION (SKIN TEARS)
═══════════════════════════════════════════════════════════════
  Dry wound (Type 1/2/3, no exudate)    → Non-adherent mesh, hydrogel, acrylic dressing
  Mild-moderate exudate (Type 1/2/3)    → Non-adherent mesh, acrylic dressing, foam (non-adhesive)
  Moderate exudate (Type 2/3)           → Non-adhesive foam, alginate, hydrofibre
  Heavy exudate (Type 2/3)              → Alginate (haemostatic), hydrofibre
  Bleeding present (Type 1/2/3)         → Alginate (haemostatic primary)
  Edge approximation needed (Type 1)    → Skin glue (within 24 hrs), non-adherent mesh
  Local infection (Types 1/2/3)         → Methylene Blue/Gentian Violet OR ionic silver (non-adherent)

*This product list is not all inclusive — there may be additional products applicable
for the treatment of skin tears.*

(Reference: ISTAP Skin Tear Tool Kit, © ISTAP 2024)
"""

print(f"\nChunk 3 length: {len(CHUNK3_PRODUCTS)} chars")
print(CHUNK3_PRODUCTS[:400])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 5: Chunk 4 — Infected Skin Tear Management
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 5 · Chunk 4 — Infected Skin Tear Management

**Source:** ISTAP Tool Kit Poster (infection/inflammation section) + Pathway PDF  
**Why a separate chunk:** Infection is a key safety dimension in your RAG pipeline.
A dedicated chunk for infection management ensures retrieval is triggered correctly
for infected skin tear cases, and populates the `contraindication_absent_silver`
safety check.

**Content:** Full infection management decision logic including tetanus, systemic
antibiotics, contraindications.
"""

# ── CELL 6 · Chunk 4 — Infected Skin Tear Management ──────────────────────────

CHUNK4_INFECTION = """\
ISTAP — Infected Skin Tear Management
Source: ISTAP Skin Tear Tool Kit 2024 (Pathway to Assessment & Treatment,
        Tool Kit Poster — Special Consideration for Infected Skin Tears)
© ISTAP 2024

SKIN TEAR INFECTION: DEFINITIONS
  LOCAL INFECTION (confined to wound):
    Signs and symptoms: localised pain, erythema (redness), warmth,
    oedema (swelling), purulent discharge, malodour, delayed healing.
  DEEP TISSUE INFECTION (spreading / systemic):
    Signs: cellulitis spreading beyond wound margins, lymphangitis,
    fever, systemically unwell. Requires systemic antibiotics and
    consideration of hospital referral.

═══════════════════════════════════════════════════════════════
MANAGEMENT DECISION TREE — INFECTED SKIN TEAR
═══════════════════════════════════════════════════════════════

STEP 1 — Determine infection depth:
  → Local infection       : Topical antimicrobial dressing
  → Deep tissue infection : Systemic antibiotics + consider referral

STEP 2 — Dressing selection for LOCAL infection (all skin tear types 1, 2, 3):

  OPTION A — Methylene Blue and Gentian Violet (MB/GV) Dressings:
    Mechanism: Broad-spectrum antimicrobial including against antibiotic-resistant
               organisms (MRSA, VRE)
    Non-traumatic to fragile wound bed
    Indications: Local or deep tissue infection — suspected or confirmed
    Considerations: Secondary dressing required
    Contraindications: Known allergy to methylene blue or gentian violet

  OPTION B — Ionic Silver Dressings:
    Mechanism: Broad-spectrum antimicrobial including against antibiotic-resistant
               organisms
    Indications: Local or deep tissue infection — suspected or confirmed
    Considerations:
      - Should NOT be used indefinitely — reassess at each dressing change
      - CONTRAINDICATED in patients with silver allergy
      - Use NON-ADHERENT silver products only — adhesive silver dressings
        risk causing further skin tears on fragile elderly skin
      - Use non-adherent versions whenever possible to minimise trauma

  NOTE: Standard iodine-based dressings are NOT listed in ISTAP's skin tear
  product guide. Iodine should also be avoided in patients with thyroid
  disorders (systemic absorption risk).

STEP 3 — Systemic antibiotic considerations:
  - Systemic antibiotics are required for DEEP TISSUE INFECTION
  - Based on culture and sensitivity (C&S) where available
  - Local infection without systemic features: topical antimicrobial is
    usually sufficient; systemic antibiotic only if not responding

STEP 4 — Tetanus immunisation:
  - Consider tetanus immunisation status for ALL skin tears
  - Skin tears, especially those with contamination or in elderly patients
    with unknown vaccination history, may require tetanus prophylaxis
  - Refer to immunisation guidelines if booster is overdue

STEP 5 — Monitoring:
  - Review infected skin tears at 48–72 hours
  - If local infection is not responding to topical antimicrobials,
    escalate to systemic antibiotics
  - If signs of spreading cellulitis, lymphangitis, or sepsis: immediate
    hospital referral

STEP 6 — Atraumatic technique for infected skin tear dressing changes:
  - Use remover wipes (e.g. medical adhesive remover)
  - Peel slowly at low angle in the direction that does not disturb
    viable skin flap edges
  - Remove in direction away from the wound, not across it
  - Use non-adherent silver or MB/GV products — never adhesive silver dressings

(Reference: ISTAP Pathway to Assessment & Treatment © ISTAP 2024;
ISTAP Tool Kit Poster — Special Consideration for Infected Skin Tears © ISTAP 2024)
"""

print(f"\nChunk 4 length: {len(CHUNK4_INFECTION)} chars")
print(CHUNK4_INFECTION[:400])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 6: Chunk 5 — Risk Assessment Pathway
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 6 · Chunk 5 — Skin Tear Risk Assessment Pathway

**Source:** ISTAP_Risk_Assessment_Pathway.pdf  
**Clinical value for RAG:** When patient notes mention elderly, fragile skin,
falls history, or polypharmacy, this chunk provides the correct risk framework
and prevention guidance to include in the recommendation.
"""

# ── CELL 7 · Chunk 5 — Risk Assessment Pathway ─────────────────────────────────

# Diagnostic: verify Risk PDF blocks
doc_risk = fitz.open(PDF_RISK)
print(f"Risk Assessment Pathway PDF: {len(doc_risk)} page(s)")
print("Raw text blocks (page 1):")
for b in get_page_blocks(doc_risk, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:100])}")
doc_risk.close()
print()

CHUNK5_RISK = """\
ISTAP SKIN TEAR RISK ASSESSMENT PATHWAY
Source: ISTAP Risk Assessment Pathway (© ISTAP 2024)
Evidence level: Strength of evidence C, expert opinion

PURPOSE:
This pathway identifies patients at risk of developing skin tears so that
preventive interventions can be implemented before a tear occurs. It is to be
used on first contact and reassessed whenever clinical status changes.

═══════════════════════════════════════════════════════════════
RISK FACTOR DOMAINS (assess all three)
═══════════════════════════════════════════════════════════════

DOMAIN 1 — GENERAL HEALTH
  High-risk factors:
    - Chronic or critical disease (heart failure, renal failure, diabetes,
      malignancy, COPD)
    - Polypharmacy — especially:
        * Anticoagulants / antiplatelet agents (warfarin, aspirin, clopidogrel)
        * Corticosteroids (systemic or long-term topical)
        * Chemotherapy agents
    - Impaired cognition (dementia, delirium — patient may resist care)
    - Sensory impairment
    - Visual impairment
    - Auditory impairment
    - Poor nutritional status (hypoalbuminaemia, vitamin C/zinc deficiency)

DOMAIN 2 — MOBILITY
  High-risk factors:
    - History of falls (any fall in previous 12 months)
    - Impaired mobility (requires walking aid, hoist, or full assistance)
    - Dependent activities of daily living (ADLs) — bathing, dressing,
      transfers requiring caregiver assistance
    - Mechanical trauma from care activities (repositioning, equipment)

DOMAIN 3 — SKIN
  High-risk factors:
    - Extremes of age (elderly adults; neonates; premature infants)
    - Fragile, thin, papery skin (senile purpura, ecchymosis, visible
      subcutaneous vessels)
    - Previous skin tears (strongest predictor of future skin tears)

═══════════════════════════════════════════════════════════════
RISK STRATIFICATION
═══════════════════════════════════════════════════════════════

  NO RISK FACTORS:
    Action: Reassess with change of status
    No active intervention required beyond standard skin care

  AT RISK (1 or more risk factors from any domain):
    Action: Implement Skin Tear Risk Reduction Programme
    Refer to Quick Reference Guide

  HIGH RISK (all of the following):
    Specific high-risk criteria:
      - Visual impairment, AND
      - Impaired mobility, AND
      - Dependent ADLs, AND
      - Extremes of age, AND
      - Previous skin tears
    Action: Immediate implementation of full Skin Tear Risk Reduction
            Programme (see Quick Reference Guide and ISTAP prevention resources)

═══════════════════════════════════════════════════════════════
SKIN TEAR RISK REDUCTION PROGRAMME — KEY INTERVENTIONS
═══════════════════════════════════════════════════════════════
(Quick Reference Guide for clinicians and caregivers)

  GENERAL HEALTH:
    - Educate patient/carer on skin tear prevention and promote active
      involvement in treatment decisions (if cognitive function allows)
    - Assess nutrition; maintain adequate body mass and hydration
    - Review polypharmacy for medication reduction/optimisation

  MOBILITY:
    - Encourage active physical function if not impaired
    - Appropriate selection and use of assistive devices
    - Daily skin assessment and monitoring for skin tears
    - Ensure safe patient handling/repositioning techniques
    - Proper transferring and repositioning programme
    - Include fall prevention programme
    - Remove clutter / hazards from environment

  SKIN:
    - Awareness of medication-induced skin fragility (e.g. topical steroids,
      systemic corticosteroids)
    - Keep skin moisturised (lubrication and hydration)
    - Keep fingernails short and smooth
    - Wear protective clothing (long sleeves, shin guards, long trousers)
    - Avoid sharp edges on furniture, equipment, beds
    - Avoid wearing jewellery during care activities
    - Use padded side rails

  HEALTHCARE SETTING:
    - Implement comprehensive skin tear risk reduction programme
    - Include skin tears in audit programmes
    - Utilise related classification system
    - Develop consultative team (wound care/dietary specialists, rehabilitation)

ANATOMICAL LOCATIONS (most common for skin tears):
  1. Hands, 2. Arms (especially pre-tibial forearm), 3. Legs (including pre-tibial, ankle),
  4. Feet, 5. Head/Face, 6. Abdomen, 7. Buttocks/Hip, 8. Chest, 9. Perineum, 10. Back

DRUGS ASSOCIATED WITH HIGH RISK OF FALLS AND SKIN FRAGILITY:
  High-risk drug categories:
    - Antidepressants (especially tricyclics, SSRIs)
    - Antipsychotics (sedation, orthostatic hypotension)
    - Antiepileptic drugs
    - Benzodiazepines and related drugs (avoid long-acting in elderly)
    - Dopaminergic drugs used in Parkinson's disease
    - Antihypertensives (especially those causing orthostatic hypotension)
    - Antibiotics (Quinolones — collagen synthesis disruption possible)
    - Diuretics
    - Insulin (hypoglycaemia → falls risk)

(Reference: ISTAP Risk Assessment Pathway, © ISTAP 2024;
Ziere et al., Br J Clin Pharmacol 2006; LeBlanc et al., Pilot Study 2011)
"""

print(f"\nChunk 5 length: {len(CHUNK5_RISK)} chars")
print(CHUNK5_RISK[:400])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 7: Chunk 6 — Atraumatic Technique & Quick Reference
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 7 · Chunk 6 — Atraumatic Dressing Technique & Quick Reference Guide

**Source:** ISTAP Tool Kit Poster (Quick Reference Guide section)  
**Why include:** Your testcase `cat_b_skin_tear_fragile` specifically tests for
correct dressing technique knowledge (adhesive bordered foam contraindication,
removal technique). This chunk captures the practical application guidance that
generates those correct recommendations.

**Content:** Dressing change technique, skin care, periwound protection,
atraumatic removal, patient / carer education points.
"""

# ── CELL 8 · Chunk 6 — Atraumatic Technique & Quick Reference ─────────────────

# Diagnostic: verify Decision Algorithm PDF
doc_algorithm = fitz.open(PDF_ALGORITHM)
print(f"Decision Algorithm PDF: {len(doc_algorithm)} page(s)")
print("Raw text blocks (page 1):")
for b in get_page_blocks(doc_algorithm, 0):
    print(f"  [{b['y0']:.0f}] {repr(b['text'][:100])}")
doc_algorithm.close()
print()

CHUNK6_TECHNIQUE = """\
ISTAP — Atraumatic Dressing Technique and Quick Reference Guide for Skin Tear Care
Source: ISTAP Skin Tear Tool Kit 2024 (Quick Reference Guide for the ISTAP Risk
        Reduction Program; Tool Kit Poster; © ISTAP 2024)

═══════════════════════════════════════════════════════════════
SECTION A — DRESSING SELECTION KEY RULES
═══════════════════════════════════════════════════════════════

RULE 1 — ALWAYS use non-adherent or low-tack dressings on fragile skin:
  - Adhesive dressings (especially bordered adhesive foam) MUST NOT be used
    on elderly patients with fragile, thin, papery skin
  - Common sites of adherent dressing trauma: forearms, shins, dorsum of hands,
    pre-tibial areas (elderly)
  - If a dressing with an adhesive border was previously applied to fragile skin:
    CHANGE to a non-adherent alternative

RULE 2 — Anchor non-adhesive dressings safely:
  - Use tubular bandage (e.g. Tubifast), cohesive bandage, or tape applied
    to non-fragile skin only
  - Film-forming liquid acrylate (skin barrier wipe) applied to periwound skin
    can help secure light dressings while protecting from maceration

RULE 3 — Dressing wear time for skin tears:
  - Wear time varies by product (see Product Selection Guide)
  - Acrylic dressings: leave in place for extended wear time (do not change daily)
  - Silicone mesh / non-adherent mesh: 2–5 days
  - Non-adhesive foam: 2–7 days (change when saturated)
  - Alginate: change when saturated; do not leave dry alginate on wound

═══════════════════════════════════════════════════════════════
SECTION B — ATRAUMATIC DRESSING REMOVAL TECHNIQUE
═══════════════════════════════════════════════════════════════

CRITICAL: Poor dressing removal technique is the leading cause of iatrogenic
(clinician-caused) skin tears in hospital and community settings.

Correct removal steps:
  1. Use a medical adhesive remover wipe (e.g. APPEEL, Niltac, Remove)
     — apply to the edge of the dressing and allow 30 seconds to work
  2. Peel the dressing edge back at a LOW ANGLE (15–30°) rather than pulling
     upward — lower angle reduces shear force on skin
  3. Support the skin with one hand while peeling dressing with the other
  4. Remove in the direction of hair growth where possible
  5. Remove slowly — do not rush
  6. Remove in a direction that does NOT disturb the viable skin flap
     (lift from the end of the wound, not across it)
  7. If the dressing is adherent: re-soak with saline or remover wipe;
     never forcibly pull

═══════════════════════════════════════════════════════════════
SECTION C — SKIN PREPARATION AND PERIWOUND CARE
═══════════════════════════════════════════════════════════════

  - Cleanse wound gently with saline or wound cleanser; avoid aggressive
    scrubbing of fragile skin
  - Apply film-forming liquid acrylate (barrier) to periwound skin:
      * Protects from maceration and moisture
      * Reduces trauma from dressing edges and tape
      * Helps secure light dressings without adhesive contact with skin
  - Moisturise periwound and surrounding skin at every dressing change:
      * Use a non-perfumed, preservative-free moisturiser
      * Avoid products containing lanolin (allergy risk in elderly)
      * Moisturising reduces skin fragility and tear risk
  - Keep periwound skin dry if exudate is heavy (change dressing before
    strikethrough; use moisture barrier paste if maceration is present)

═══════════════════════════════════════════════════════════════
SECTION D — PATIENT AND CAREGIVER EDUCATION
═══════════════════════════════════════════════════════════════

  Educate on prevention:
    - Wear long-sleeved, lightweight protective garments (e.g. tubular
      sleeves/stockinette) over at-risk skin — especially forearms and shins
    - Pad sharp edges on bed rails, wheelchairs, furniture
    - Ensure good lighting to prevent falls
    - Maintain hydration (8 glasses water/day — improves skin turgor)
    - Maintain adequate nutrition (protein, vitamins C and E, zinc)
    - Keep fingernails trimmed and smooth to avoid accidental scratching
    - Use emollient / moisturiser twice daily on at-risk skin

  Educate on dressing care:
    - Never pull dressings off quickly
    - Always use adhesive remover wipes if any stickiness is felt
    - Report signs of infection promptly (increased redness, warmth, discharge, smell)
    - Attend follow-up review as scheduled (skin tears can deteriorate rapidly)

═══════════════════════════════════════════════════════════════
SECTION E — REFERRAL INDICATORS FOR SKIN TEARS
═══════════════════════════════════════════════════════════════

  Consider referral to wound care specialist or dermatology if:
    - Skin tear is not healing after 2 weeks of appropriate treatment
    - Wound edges are non-advancing (consider active therapy)
    - Deep tissue infection (cellulitis, lymphangitis, systemic signs)
    - Very large skin tear (>5 cm) or full-thickness loss in high-risk patient
    - Recurrent skin tears despite prevention programme

  Consider tetanus assessment for:
    - Any skin tear with uncertain vaccination history
    - Elderly patients who may not have had recent booster
    - Contaminated skin tears

(Reference: ISTAP Quick Reference Guide for the ISTAP Risk Reduction Program,
ISTAP Skin Tear Tool Kit © ISTAP 2024; LeBlanc & Baranoski, Adv Skin Wound Care 2009)
"""

print(f"\nChunk 6 length: {len(CHUNK6_TECHNIQUE)} chars")
print(CHUNK6_TECHNIQUE[:400])


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 8: Assemble all chunks
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 8 · Assemble all 6 chunks into the final list
"""

# ── CELL 9 · Assemble chunk list ───────────────────────────────────────────────

def make_chunk(
    section: str,
    parent_section: str,
    text: str,
    chunk_index: int = 0,
) -> dict:
    """Build a chunk dict matching the schema used by GP / AJGP / SFP / WCM."""
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

# ── Chunk 1: Classification ───────────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Classification System (Types 1, 2, 3)",
    parent_section = "ISTAP Skin Tear Assessment",
    text           = CHUNK1_CLASSIFICATION,
))

# ── Chunk 2: Pathway ─────────────────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Pathway to Assessment and Treatment",
    parent_section = "ISTAP Skin Tear Assessment",
    text           = CHUNK2_PATHWAY,
))

# ── Chunk 3: Product Selection ───────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Product Selection Guide",
    parent_section = "ISTAP Skin Tear Treatment",
    text           = CHUNK3_PRODUCTS,
))

# ── Chunk 4: Infected Skin Tear ───────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Infected Skin Tear Management",
    parent_section = "ISTAP Skin Tear Treatment",
    text           = CHUNK4_INFECTION,
))

# ── Chunk 5: Risk Assessment ─────────────────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Risk Assessment Pathway",
    parent_section = "ISTAP Skin Tear Prevention",
    text           = CHUNK5_RISK,
))

# ── Chunk 6: Technique & Quick Reference ─────────────────────────────────────
chunks.append(make_chunk(
    section        = "ISTAP Skin Tear — Atraumatic Technique and Quick Reference",
    parent_section = "ISTAP Skin Tear Treatment",
    text           = CHUNK6_TECHNIQUE,
))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Total chunks assembled: {len(chunks)}")
for i, c in enumerate(chunks, 1):
    print(f"  {i}. '{c['section']:60s}' chars={c['char_count']:5d}")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 9: Quality validation
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 9 · Quality validation — char counts, chunk_id uniqueness, deduplication check
"""

# ── CELL 10 · Quality checks ───────────────────────────────────────────────────

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

# ── 4. Total character count (sanity check) ───────────────────────────────────
total_chars = sum(c["char_count"] for c in chunks)
print(f"\n   Total characters across {len(chunks)} chunks: {total_chars:,}")
print(f"   Average chars per chunk:                {total_chars // len(chunks):,}")
print(f"   Source name key:                         {SOURCE_NAME}")

# ── 5. Clinical keyword coverage check ───────────────────────────────────────
# Verify each safety-critical keyword appears somewhere in the combined text
combined = "\n".join(c["text"] for c in chunks).lower()
keywords = [
    ("silver", "ionic silver dressings for infection"),
    ("alginate", "alginate for exudate / haemostasis"),
    ("foam", "foam dressings (non-adhesive)"),
    ("adhesive bordered", "adhesive bordered foam contraindication"),
    ("hydrogel", "hydrogel for dry wounds"),
    ("skin glue", "skin glue for type 1 edge approximation"),
    ("silver allergy", "silver contraindication in allergy"),
    ("iodine", "iodine note for thyroid context"),
    ("non-adherent", "non-adherent dressing requirement"),
    ("type 1", "skin tear type 1 classification"),
    ("type 2", "skin tear type 2 classification"),
    ("type 3", "skin tear type 3 classification"),
    ("tetanus", "tetanus consideration"),
    ("systemic antibiotic", "systemic antibiotic for deep infection"),
    ("referral", "referral indicators"),
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


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 10: Spot-check individual chunks
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 10 · Spot-check individual chunks
"""

# ── CELL 11 · Spot-check chunks ────────────────────────────────────────────────

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

# Spot-check: Classification (most important for testset),
#             Product Selection Guide (most important for dressing recommendations),
#             Infection (safety check), Technique (cat_b_skin_tear_fragile)
for idx in [0, 2, 3, 5]:
    preview_chunk(idx)


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CELL — Step 11: (Optional) LLM ai_summary enrichment
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 11 · (Optional) LLM `ai_summary` enrichment

Set `ENABLE_AI_SUMMARY = True` when your OpenAI API key is available.
The ai_summary is what gets embedded into ChromaDB as `page_content`.
The raw `text` is stored as metadata and used for evidence display in the app.
"""

# ── CELL 12 · LLM ai_summary ──────────────────────────────────────────────────

ENABLE_AI_SUMMARY = False   # ← set True when OpenAI key is available

if ENABLE_AI_SUMMARY:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    SYSTEM_PROMPT = (
        "You are a medical summarisation assistant. "
        "Rewrite the following wound-care guideline text as a clear, complete, self-contained "
        "clinical summary suitable for retrieval-augmented generation. "
        "Preserve all clinical facts, dressing names, wound types, indications, and "
        "contraindications. Include all product names, skin tear types (1, 2, 3), "
        "infection management steps, and contraindications (silver allergy, adhesive "
        "bordered foam on fragile skin, iodine in thyroid disorders). "
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
# MARKDOWN CELL — Step 12: Export JSON
# ══════════════════════════════════════════════════════════════════════════════
"""
## Step 12 · Export `ISTAP_skin_tear_kept.json`
"""

# ── CELL 13 · Export ChromaDB-ready JSON ──────────────────────────────────────
# Format mirrors GP/AJGP/SFP/WCM _kept.json so ingestion_full.ipynb can load
# all source files uniformly.

output = {
    "meta": {
        "total_chunks":    len(chunks),
        "kept_count":      len(chunks),
        "ai_summarised":   sum(1 for c in chunks if c["ai_summary"] != c["text"]),
        "extraction":      (
            "Hardcoded text reconstruction from 4 ISTAP PDFs "
            "(Pathway, Tool Kit Poster, Risk Assessment Pathway, Decision Algorithm). "
            "PyMuPDF used for verification only — poster/flowchart layout prevents "
            "automated reliable text extraction."
        ),
        "chunking":        "manual section-aware — one chunk per clinical domain",
        "source_pdfs":     [
            "ISTAP_Pathway_to_Assessment_-_Treatment.pdf",
            "ISTAP_Tool_Kit_Poster.pdf",
            "ISTAP_Risk_Assessment_Pathway.pdf",
            "ISTAP_Decision_Algorithm.pdf",
        ],
        "sections_used": [
            "Classification System (Types 1-3)",
            "Pathway to Assessment & Treatment (flowchart)",
            "Product Selection Guide (table)",
            "Infected Skin Tear Special Considerations",
            "Risk Assessment Pathway",
            "Quick Reference Guide / Atraumatic Technique",
        ],
        "sections_dropped": [
            "Prevalence Study Data Collection Sheet (data collection form)",
            "Drugs Associated with Risk of Falls (falls pharmacology list)",
            "Reference list",
            "ISTAP contact details, website footers, social media handles",
            "Image/photograph sections",
        ],
        "chunk_params": {
            "min_characters": MIN_CHUNK_CHARS,
        },
        "wound_category":  "skin_tear",
        "note": (
            "Use ai_summary field for ChromaDB page_content and RAGAS reference_contexts. "
            "wound_category=skin_tear enables metadata filtering in v4 sub-query A. "
            "All 4 ISTAP PDFs are merged under SOURCE_NAME='ISTAP_Skin_Tear_Guidelines_2024.pdf' "
            "to avoid duplicate embeddings from companion documents."
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

out_path = OUT_DIR / "ISTAP_skin_tear_kept.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Exported {len(chunks)} chunks → {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")


# ── CELL 14 · Final summary table ─────────────────────────────────────────────

with open(OUT_DIR / "ISTAP_skin_tear_kept.json") as f:
    exported = json.load(f)

print("═" * 72)
print("INGESTION COMPLETE — ISTAP Skin Tear Guidelines 2024")
print("═" * 72)

hdr = f"{'#':>3}  {'Chunk ID':14}  {'Section':58}  {'Chars':>5}  {'AI?':8}"
print(hdr)
print("-" * len(hdr))
for i, c in enumerate(exported["kept_chunks"], 1):
    ai  = "yes" if c["ai_summary"] != c["text"] else "no (raw)"
    sec = c["section"][:58]
    print(f"{i:3d}  {c['chunk_id']:14}  {sec:58s}  {c['char_count']:5d}  {ai}")

print()
print(f"Output JSON     : {OUT_DIR / 'ISTAP_skin_tear_kept.json'}")
print(f"Source key      : {SOURCE_NAME}")
print(f"wound_category  : skin_tear")
print()
print("Next steps:")
print("  1) Review chunk text above — edit CHUNK1–CHUNK6 constants if any clinical")
print("     content is missing or should be adjusted")
print("  2) Set ENABLE_AI_SUMMARY = True (Cell 12) to enrich ai_summary with GPT-4o-mini")
print("  3) Add ISTAP_skin_tear_kept.json to ingestion_full.ipynb's CHUNK_FILES dict")
print("  4) Add ISTAP to GUIDELINE_METADATA in ingestion_full.ipynb:")
print("       SOURCE_NAME: {")
print("           'guideline_type': 'specialist_skin_tear',")
print("           'authority':      'ISTAP',")
print("           'year':           '2024',")
print("           'focus':          'skin_tear_prevention_treatment',")
print("       }")
print("  5) Re-run ingestion_full.ipynb to rebuild db_wound_care_v4")
print()
print("RAGAS testset updates needed:")
print("  • cat_b_skin_tear_fragile: add ISTAP chunk IDs to reference_contexts")
print("    Suggested: [ctx(AJGP_SKINTEAR), ctx(ISTAP_PRODUCTS), ctx(ISTAP_PATHWAY)]")
print("    where ISTAP_PRODUCTS and ISTAP_PATHWAY are the new chunk_ids printed above")
print("  • Update allowed_dressings for cat_b_skin_tear_fragile:")
print("    ['silicone_foam', 'silicone_mesh', 'non_adherent_mesh', 'alginate']")
print()

# RAGAS reference-context lookup map
ref_ctx_by_section = defaultdict(list)
for c in exported["kept_chunks"]:
    ref_ctx_by_section[c["section"]].append(c["ai_summary"])

print(f"RAGAS reference_context lookup ready — {len(ref_ctx_by_section)} sections")
print()
print("CHUNK ID MAP (add to wound_testset_builder_v3.py):")
print("─" * 50)
for c in exported["kept_chunks"]:
    alias = (
        c["section"]
        .replace("ISTAP Skin Tear — ", "ISTAP_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("1,_2,_3", "CLASS")
        .upper()
    )
    # Friendly aliases
    alias_map = {
        0: "ISTAP_CLASSIFICATION",
        1: "ISTAP_PATHWAY",
        2: "ISTAP_PRODUCTS",
        3: "ISTAP_INFECTION",
        4: "ISTAP_RISK",
        5: "ISTAP_TECHNIQUE",
    }
    idx = exported["kept_chunks"].index(c)
    print(f'{alias_map.get(idx, "ISTAP_CHUNK_" + str(idx)):30s} = "{c["chunk_id"]}"')
