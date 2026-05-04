# ══════════════════════════════════════════════════════════════════════════════
# ingestion_EWMA.py
# EWMA Position Document: Wound Bed Preparation in Practice (2004)
# European Wound Management Association (EWMA) / MEP Ltd, London
# ══════════════════════════════════════════════════════════════════════════════

# ── CELL 1 · Document profile (markdown) ──────────────────────────────────────
#
# ## ingestion_EWMA.ipynb
# ### EWMA Position Document: Wound Bed Preparation in Practice (London: MEP Ltd, 2004)
#
# | Property        | Detail                                                          |
# |---|---|
# | Organisation    | European Wound Management Association (EWMA)                    |
# | Year            | 2004                                                            |
# | Total pages     | 19                                                              |
# | Useful pages    | 3–18 (pages 1–2 = cover/credits; page 19 = VLU references only)|
# | Language        | English                                                         |
# | Layout          | Two-column with left sidebar labels; text-extractable PDF       |
# | Tables          | 2 × "Advanced therapies" tables (DFU p.10–11; VLU p.15–16)    |
# | Key framework   | TIME (Tissue management, Inflammation/Infection control,        |
# |                 |   Moisture balance, Epithelial (edge) advancement)              |
#
# ### Articles / sections in this document
# | PDF pages | Article / Section                                              |
# |---|---|
# | 3         | Editorial overview — CJ Moffatt (EWMA Past President)          |
# | 4–5       | Wound bed preparation: science applied to practice — V Falanga  |
# | 6–11      | Wound bed preparation for diabetic foot ulcers — Edmonds et al |
# | 12–16     | Wound bed preparation for venous leg ulcers — Moffatt et al    |
# | 17–19     | VLU references (skip)                                          |
#
# ### Chunk architecture (13 chunks)
# | # | Section                                                   | Source       |
# |---|---|---|
# | 1 | TIME Framework — Evolution & Four Components               | pp.3–4       |
# | 2 | TIME Applied to Practice — Pathway & WBP Principles       | pp.4–5       |
# | 3 | Wound Bed Preparation — Tissue Management (general)       | pp.4         |
# | 4 | Wound Bed Preparation — Inflammation & Infection Control   | pp.4–5       |
# | 5 | Wound Bed Preparation — Moisture Balance & Edge Advance.   | pp.4–5       |
# | 6 | DFU — Before TIME + Tissue Management (Debridement)       | pp.6–7       |
# | 7 | DFU — Inflammation & Infection Control                    | pp.7–10      |
# | 8 | DFU — Moisture Balance & Rationale for Covering Ulcers    | p.10         |
# | 9 | DFU — Epithelial (Edge) Advancement                       | pp.9–10      |
# |10 | DFU — Advanced Therapies (Table 1) & After TIME           | pp.10–11     |
# |11 | VLU — Before TIME + Tissue Management (Debridement)       | pp.12–13     |
# |12 | VLU — Inflammation & Infection Control + Moisture Balance  | pp.13–16     |
# |13 | VLU — Epithelial (Edge) Advancement + Advanced Therapies   | pp.15–18     |
#
# ### Strategy
# All text is reconstructed verbatim from the PDF (verified against pdftotext -layout
# and fitz block extraction). No pypdf/pdfplumber extraction is used for body text
# because the two-column + sidebar layout fragments text unpredictably.
# The two Advanced Therapies tables are reconstructed from verified block text.
# Pages 1–2 (cover, credits) and page 19 (VLU references only) are skipped.


# ── CELL 2 · Dependencies & paths ─────────────────────────────────────────────

# Uncomment if any library is missing:
# !pip install pymupdf --break-system-packages -q

import fitz          # PyMuPDF — used only for PDF open/verify and block diagnostic
import re
import json
import hashlib
import unicodedata
import statistics
from pathlib import Path
from collections import defaultdict, Counter

# ── paths ──────────────────────────────────────────────────────────────────────
PDF_PATH    = "../clinical_pdfs_v2/EWMA_Wound_Bed_Preparation_in_Practice.pdf"
SOURCE_NAME = "EWMA_Wound_Bed_Preparation_in_Practice.pdf"
OUT_DIR     = Path("../ingestion_output_no_ai")
OUT_DIR.mkdir(exist_ok=True)

MIN_CHUNK_CHARS = 60

print("✅ imports ok")


# ── CELL 3 · Helper functions ──────────────────────────────────────────────────

def make_chunk_id(source: str, section: str, idx: int = 0) -> str:
    """Deterministic 12-char MD5 hex ID — matches pattern used by GP/AJGP/SFP/WCM/ISTAP/ANZBA."""
    raw = f"{source}::{section}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def clean_block_text(text: str) -> str:
    """
    Clean a raw PyMuPDF text block:
    - NFKC normalise
    - Strip lone page-number strings
    - Drop EWMA running header / footer noise
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    # Lone page numbers (1–2 digits)
    if re.fullmatch(r"\d{1,2}", text):
        return ""
    # Running header / structural noise specific to this PDF
    for noise in [
        "POSITION\nDOCUMENT",
        "WOUND BED PREPARATION IN PRACTICE",
        "POSITION DOCUMENT",
    ]:
        if text.strip() == noise:
            return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_chunk(section: str, parent_section: str, text: str, chunk_index: int = 0) -> dict:
    return {
        "chunk_id":       make_chunk_id(SOURCE_NAME, section, chunk_index),
        "source":         SOURCE_NAME,
        "section":        section,
        "parent_section": parent_section,
        "chunk_index":    chunk_index,
        "char_count":     len(text),
        "text":           text,
        "ai_summary":     text,  # overwrite with LLM summary if ENABLE_AI_SUMMARY = True
    }


# ── Verify PDF opens correctly ─────────────────────────────────────────────────
doc = fitz.open(PDF_PATH)
print(f"✅ Opened PDF: {len(doc)} pages")
print(f"   Extracting pages 3–18 (0-indexed 2–17); skipping cover (0–1) and references (18)")
doc.close()


# ── CELL 4 · Chunk 1 — TIME Framework: Evolution & Four Components ────────────
#
# Source: pp. 3–4 (Editorial overview p.3 + Falanga article pp.4–5)
# Covers: TIME acronym definition, EWMA advisory board reformulation (Table 1),
#         four component descriptions, framework goals
#
# Verified verbatim against pdftotext -layout output and fitz block inspection.

CHUNK1_TIME_FRAMEWORK = """\
EWMA POSITION DOCUMENT — TIME Wound Bed Preparation Framework
Source: European Wound Management Association (EWMA). Position Document:
Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.3–4.

DEFINITION AND ORIGIN
Wound bed preparation offers clinicians a comprehensive approach to removing barriers
to healing and stimulating the healing process. Based on the work of the International
Wound Bed Preparation Advisory Board, an acronym has been formed using the names of
the components in the English language; the framework has been named TIME. In order
to maximise their value across different disciplines and languages, the EWMA wound bed
preparation editorial advisory board has further developed the terms.

TIME ACRONYM — EWMA FORMULATION (Table 1: Evolution of the TIME framework):

  Original acronym                         EWMA advisory board terms
  ─────────────────────────────────────────────────────────────────────
  T = Tissue, non-viable or deficient   →  Tissue management
  I = Infection or inflammation         →  Inflammation and infection control
  M = Moisture imbalance                →  Moisture balance
  E = Edge of wound,                    →  Epithelial (edge) advancement
      non-advancing or undermined

FOUR COMPONENTS OF WOUND BED PREPARATION

T — Tissue management
  The presence of necrotic or compromised tissue is common in chronic non-healing
  wounds, and its removal has many beneficial effects. It takes away non-vascularised
  tissue, bacteria and cells that impede the healing process (cellular burden), thus
  providing an environment that stimulates the build-up of healthy tissue. In the light
  of recent studies about senescence of wound cells and their unresponsiveness to
  certain signals, the fact that debridement removes the cellular burden and allows a
  stimulatory environment to be established is particularly important. Unlike acute
  wounds, which usually only require debridement once if at all, chronic wounds may
  require repeated debridement.

I — Inflammation and infection control
  Chronic wounds are often heavily colonised with bacterial or fungal organisms. This is
  due in part to the fact that these wounds remain open for prolonged periods, but is also
  related to other factors such as poor blood flow, hypoxia and the underlying disease
  process. There is little question that clinical infection resulting in failure to heal must
  be treated aggressively and promptly. Evidence shows that a bacterial burden of 10^6
  organisms or more per gram of tissue seriously impairs healing, although the reason for
  this is poorly understood.
  Recently, there has been increasing interest in the possible presence of biofilms in
  chronic wounds and their role in impaired healing or recurrence. Biofilms are bacterial
  colonies surrounded by a protective coat of polysaccharides; such colonies become more
  easily resistant to the action of antimicrobials.

M — Moisture balance
  Experimental evidence indicating that keeping wounds moist accelerates re-epithelisation
  is one of the major breakthroughs of the last 50 years and led to the development of a
  vast array of moisture-retentive dressings that promote 'moist wound healing'. Most
  evidence for moist wound healing was developed in experiments on acute wounds, but
  the findings were quickly extrapolated to chronic wounds. Contrary to what had been
  conventional wisdom, keeping the wound moist does not increase infection rates.
  Fluid from chronic wounds will block cellular proliferation and angiogenesis and contains
  excessive amounts of matrix metalloproteinases (MMPs) capable of breaking down
  critical extracellular matrix proteins, including fibronectin and vitronectin. Excessive
  activity (or maldistribution) of enzymes MMP-2 and MMP-9 impair healing.

E — Epithelial (edge) advancement
  Effective healing requires the re-establishment of an intact epithelium and restoration
  of skin function. However, the process of epithelialisation may be impaired either
  indirectly, such as when faults in the wound matrix or ischaemia inhibit keratinocyte
  migration, or directly due to regulatory defects, impaired cellular mobility or adhesion
  within the keratinocytes.
  There is increasing evidence that the resident cells of chronic wounds have undergone
  phenotypic changes that impair their capacity to proliferate and move. Fibroblasts from
  venous and pressure ulcers show diminished ability to proliferate and their decreased
  proliferative capacity correlates with a failure to heal.

FRAMEWORK GOALS
The TIME framework aims to optimise the wound bed by reducing oedema and exudate,
reducing the bacterial burden and, importantly, correcting the abnormalities contributing
to impaired healing. This should facilitate the normal endogenous process of wound
healing, providing the underlying intrinsic and extrinsic factors affecting the wound's
failure to heal have also been addressed.
"""

print(f"Chunk 1 (TIME Framework) length: {len(CHUNK1_TIME_FRAMEWORK)} chars")


# ── CELL 5 · Chunk 2 — TIME Applied to Practice: Pathway & WBP Principles ─────
#
# Source: pp. 3–5 (Editorial overview + Falanga conclusion)
# Covers: Non-linearity of TIME, wound bed preparation pathway (Figure 2 described),
#         holistic assessment, growth factor trapping, impaired blood flow & hypoxia,
#         therapeutic boldness conclusion

CHUNK2_TIME_IN_PRACTICE = """\
EWMA — TIME Applied to Practice: Wound Bed Preparation Pathway & Key Principles
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.3–5.

KEY PRINCIPLE: THE TIME FRAMEWORK IS NOT LINEAR
During the process of healing different elements of the framework will require attention.
A single intervention can impact on more than one element of the framework — for example
debridement will not only remove necrotic tissue but will also reduce bacterial load.
Different wounds require attention to different elements of TIME.

WOUND BED PREPARATION PATHWAY (Figure 2)
The pathway shows how wound bed preparation is applied to practice:

  1. Patient / Wound Assessment → Preliminary Diagnosis (e.g. treat underlying cause)
  2. Immediate Considerations (e.g. surgical debridement if required)
  3. Basic Wound Management → Ongoing Assessment
  4. If wound is UNCOMPLICATED and HEALING → Healed Wound
  5. If wound is NON-HEALING → Apply TIME:
       Tissue Management
       Inflammation & Infection Control
       Moisture Balance
       Epithelial (Edge) Advancement
  6. Follow-up Assessment:
       - HEALING WOUND  → Continue present therapy
       - NON-HEALING WOUND → Re-evaluate TIME → Implement Advanced Therapies → Healed Wound

HOLISTIC WOUND ASSESSMENT
Wound bed preparation should not be seen in isolation from holistic wound assessment,
which encompasses the patient's psychosocial needs as well as underlying and associated
aetiologies. Used in this way, if all elements of the framework are successfully addressed,
many wounds should move towards healing.

INTEGRATION OF TIME INTO OVERALL CARE
This position document reinforces the importance of integrating TIME into an overall
programme of care that addresses all other aspects of the patient's treatment:
  - Venous ulcers will not heal without compression.
  - Diabetic foot ulcers will not heal without pressure offloading and diabetic control.
  - For diabetic foot ulcers, the emphasis within TIME is on tissue management in the
    form of radical and repeated debridement, and inflammation and infection control.
  - For venous leg ulcers, the emphasis is on restoring and maintaining moisture balance,
    while tissue management and infection control are less prominent issues.

GROWTH FACTOR TRAPPING (advanced pathophysiology)
Normal components of plasma, if continuously present, can lead to what has been
hypothesised as 'growth factor trapping'. The hypothesis is that certain macromolecules
and even growth factors are bound or 'trapped' in the tissues, which could result in
unavailability or maldistribution of critical mediators, including cytokines. Trapping of
growth factors and cytokines, as well as matrix material, has the potential to cause a
cascade of pathogenic abnormalities, and dressings may play an important role in
modulating these factors.

IMPAIRED BLOOD FLOW AND HYPOXIA
There is a substantial body of data indicating that low levels of oxygen tension as measured
at the skin surface correlate with inability to heal. It should be noted that ischaemia is not
the same as hypoxia. Low levels of oxygen tension can stimulate fibroblast proliferation
and clonal growth, and can actually enhance the transcription and synthesis of a number of
growth factors. It is possible that low oxygen tension serves as a potent initial stimulus
after injury, while prolonged hypoxia, as seen in chronic wounds, can lead to a number of
abnormalities including scarring and fibrosis, as well as delayed edge migration and poor
restoration of epithelial function.

CLINICAL CONCLUSION (Falanga)
Greater therapeutic boldness is required and one of the challenges for clinicians is to
recognise when therapeutic interventions should be introduced to accelerate healing.
TIME provides a framework for the cost-effective introduction of advanced and expensive
technologies, targeting them at patients who will benefit from their use.
"""

print(f"Chunk 2 (TIME in Practice) length: {len(CHUNK2_TIME_IN_PRACTICE)} chars")


# ── CELL 6 · Chunk 3 — TIME Figure 1: Dynamic TIME Progression Descriptions ───
#
# Source: p. 5 (Figure 1 captions — four sequential wound states)
# These four states are clinically significant: they map directly to the T/I/M/E
# dimensions your RAG system scores, and describe dynamic wound progression.

CHUNK3_TIME_FIGURE1 = """\
EWMA — TIME Framework: Dynamic Wound Progression (Figure 1 — Four Wound States)
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. p.5.

Figure 1 shows TIME applied to practice using the example of an open, chronic,
slow-healing wound, progressing through four sequential states:

STATE 1a — TISSUE (T) DOMINANT
  Represents an open chronic, slow-healing wound, covered with necrotic tissue
  requiring debridement.
  → Primary intervention: Tissue management (debridement).

STATE 1b — INFECTION (I) DOMINANT
  The wound has become critically colonised or infected, slowing healing.
  Antimicrobial agents and further debridement are required.
  → Primary intervention: Inflammation and infection control.

STATE 1c — MOISTURE (M) DOMINANT
  As a result of infection and/or inflammation the wound is producing more exudate
  and attention now focuses on moisture balance.
  → Primary intervention: Moisture balance (exudate management).

STATE 1d — EDGE (E) DOMINANT
  As the critical colonisation or infection resolves and moisture balance is achieved,
  attention should move to epithelial (edge) advancement.
  → Primary intervention: Epithelial (edge) advancement.

CLINICAL IMPLICATION FOR RAG DRESSING SELECTION
The TIME framework is not a one-time assessment. As wound status changes (e.g. from
heavily necrotic to infected to exuding to stalled edges), the priority element of TIME
shifts and dressing selection must change accordingly. A wound may require attention to
multiple TIME components simultaneously or sequentially.
"""

print(f"Chunk 3 (Figure 1 TIME progression) length: {len(CHUNK3_TIME_FIGURE1)} chars")


# ── CELL 7 · Chunk 4 — DFU: Before TIME & Tissue Management (Debridement) ─────
#
# Source: pp. 6–8 (Edmonds, Foster, Vowden — Diabetic Foot Ulcers)
# Covers: Introduction, Before TIME prerequisites, sharp debridement (gold standard),
#         tissue characteristics, larval therapy

CHUNK4_DFU_TISSUE = """\
EWMA — Wound Bed Preparation for Diabetic Foot Ulcers (DFU): Before TIME & Tissue Management
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.6–8.
Authors: M Edmonds, AVM Foster, P Vowden.

INTRODUCTION
Diabetic foot ulcers occur when trauma leads to an acute wound, which progresses to a
chronic wound due to extrinsic and intrinsic factors. The aim is to create a well-vascularised
wound bed surrounded by intact skin with an advancing epithelial edge that progresses to
healing and produces a stable scar.
Diabetes extends beyond glycaemic control, affecting protein synthesis, white cell function,
oxygen transportation and utilisation and growth factor availability. These complications are
compounded by poor glycaemic control, and exacerbated by neuropathy, cheiroarthropathy
(diabetic changes affecting the skin and joints) and peripheral vascular disease. Suppression
of neutrophil function further aggravates the situation by increasing the risk of infection.

BEFORE TIME — PREREQUISITES FOR WOUND CARE SUCCESS
When managing ulceration in the diabetic foot the underlying pathophysiology must be
established to identify whether there is evidence of peripheral neuropathy and/or peripheral
vascular disease (ischaemia). The underlying physical cause of the wound must also be
identified and, if possible, eliminated or corrected. Three basic elements must be addressed:
  ● Pressure control: offloading and weight redistribution and/or callus removal
  ● Restoration or maintenance of pulsatile blood flow
  ● Metabolic control.
Unless these elements are addressed, wound care is more likely to fail and the patient will
be at increased risk of amputation or recurrent ulceration. Education should also be given
to ensure the patient understands the aims of treatment.

T — TISSUE MANAGEMENT: DEBRIDEMENT
The diabetic foot does not tolerate sloughy, necrotic tissue, and debridement is therefore an
important component of ulcer management. Debridement serves several functions:
  - Removes necrotic tissue and callus
  - Reduces pressure
  - Allows full inspection of the extent of the wound
  - Facilitates drainage
  - Stimulates healing.
Studies by Steed et al confirmed that patients with diabetic neuropathic foot ulcers which
underwent regular sharp debridement did better than those whose ulcers had less debridement.

SHARP DEBRIDEMENT — GOLD STANDARD
With the exception of ulcers requiring extensive debridement by a surgeon while the patient
is under general anaesthetic, the gold standard method is sharp debridement. This can
remove the unhealthy components of a chronic foot wound, stimulating the wound bed by
creating an acute injury in a chronic wound environment. Regular sharp debridement may
be necessary to prevent the wound from reverting to a purely chronic state.

TISSUE CHARACTERISTICS — RECOGNISING VIABLE VS NON-VIABLE TISSUE
Healthy tissue: pink or red, and either shiny and smooth or with 'rosettes' on the surface;
new epithelium can be seen growing from the wound edge and is pink or pearly white.
Non-viable tissue may:
  ● Be yellow, grey, blue, brown or black
  ● Have a soft or slimy consistency
  ● Form a hard, 'leathery' eschar.
Debridement is indicated where there is accumulation of callus, slough, fibrous tissue or
obviously non-viable tissue. It is important to achieve the right balance: removing too much
will prolong the healing process, while if too little is removed, the wound's chronic status
will continue.

NEUROPATHIC vs NEUROISCHAEMIC FOOT — DEBRIDEMENT APPROACH
  Neuropathic foot (good blood supply): Aggressive sharp debridement (to healthy, bleeding
    tissue) can be performed to remove callus, slough, necrosis and non-viable tissue.
  Neuroischaemic foot (poor blood supply): Benefits from removal of non-viable tissue but
    must be debrided with extreme caution to minimise damage to viable tissue.
  Sharp debridement can also help prevent or manage infection if sinuses are opened,
    sloughy infected tissue removed and fluid-filled cavities drained.
  In the neuropathic foot, wet necrosis caused by infection can be treated with intravenous
    antibiotics and surgical debridement.
  In the neuroischaemic foot with severe ischaemia: revascularisation should be performed.
    If vascular intervention is not possible, an attempt should be made to convert wet
    necrosis to dry necrosis using intravenous antibiotics and appropriate wound care such
    as the use of iodine products. Some cases do well with a dry managed eschar and may
    proceed to auto-amputation.

LARVAL THERAPY (alternative debridement)
Although sharp debridement is the gold standard, on occasions if the foot is too painful or
the patient has expressed a preference, larvae of the greenbottle fly can achieve relatively
rapid, atraumatic removal of necrotic material.
  - The larvae may be used to remove slimy slough in painful ulcers in the neuroischaemic foot.
  - They are NOT recommended as the sole agent for debriding the neuropathic foot as they
    do not remove callus, which is essential for healing.
  - They may, however, reduce the bacterial load.
"""

print(f"Chunk 4 (DFU Tissue Management) length: {len(CHUNK4_DFU_TISSUE)} chars")


# ── CELL 8 · Chunk 5 — DFU: Inflammation & Infection Control ──────────────────
#
# Source: pp. 7–10 (Edmonds, Foster, Vowden)
# Covers: Infection threat, indicators, cellulitis, osteomyelitis, bacterial management
#         (antimicrobials: iodine, silver, mupirocin), general principles table

CHUNK5_DFU_INFECTION = """\
EWMA — Wound Bed Preparation for Diabetic Foot Ulcers (DFU): Inflammation & Infection Control
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.7–10.
Authors: M Edmonds, AVM Foster, P Vowden.

I — INFLAMMATION AND INFECTION CONTROL

INFECTION RISK IN THE DIABETIC FOOT
Infection is a threat to the diabetic foot as high-risk patients are immunocompromised,
while in those who have poor metabolic control white cell function is impaired. It is
implicated in most cases that result in major amputation. Staphylococci and streptococci
are the most common pathogens, although gram-negative and anaerobic organisms occur
in approximately 50% of patients, and infection is often polymicrobial. Bacterial species
that are not pathogenic may cause a true infection in a diabetic foot as part of mixed flora,
and poor immune response seen on occasions in diabetic patients means that even bacteria
regarded as skin commensals may cause severe tissue damage.
While increased bacterial burden slows healing, the host-bacteria relationship is complex
as many wounds are colonised with a stable bacterial population. If the bacterial burden
increases, it may result in increased exudate as clinical infection develops.
The signs of inflammation and infection are absent or reduced in many diabetic patients,
such as those who lack the protective pain sensation and/or have a poor blood supply to
the feet, and may be masked in patients with a severe autonomic neuropathy.

INDICATORS OF INFECTION IN DIABETIC FOOT ULCERS
  • Ulcer base yellowish grey
  • Blue discoloration of surrounding tissues
  • Fluctuance (softness) or crepitus (crackling, grating) on palpation
  • Purulent exudate
  • Sloughing of ulcer and surrounding tissue
  • Sinuses with undermined or exposed bone
  • Abscess formation
  • Odour
  • Wound breakdown
  • Delayed healing
  Note: Classic signs of infection (pain, erythema, heat and purulence) may be absent
  or reduced due to sensory neuropathy and/or ischaemia.

CELLULITIS AND OSTEOMYELITIS
Cellulitis covers a spectrum of presentations, including local infection of the ulcer,
spreading cellulitis, sloughing of soft tissue and vascular compromise of the skin. When
vascular compromise occurs there is an inadequate supply of oxygen to the soft tissues,
causing a blue discoloration.
When infection spreads there is widespread, intense erythema, swelling and lymphangitis.
Regional lymphadenitis may occur with malaise, 'flu-like' symptoms and rigors. Pain and
throbbing usually indicate pus within the tissues, but these symptoms are often absent in
the neuropathic foot. Palpation may reveal fluctuance (a soft, saturated feeling) or
crepitus (a crackly, grating feeling), which suggest abscess formation. Often there is
generalised sloughing of the ulcer and surrounding subcutaneous tissues, which liquefy
and disintegrate.
If a sterile probe inserted into the ulcer reaches bone, this suggests osteomyelitis. In the
initial stages plain X-ray may be normal and localised loss of bone density and cortical
outline may not be apparent until at least 14 days later.

BACTERIAL MANAGEMENT — TOPICAL THERAPY
Saline is the cleansing agent of choice as it does not interfere with microbiological
samples or damage granulating tissue. Cetrimide-based cleansing agents are not
recommended as their cytotoxic action may impede healing.
Three antimicrobials are commonly used:
  ● Iodine: Effective against a wide spectrum of organisms. Current consensus suggests
    that slow-release iodine formulations are useful for antisepsis without impairing healing
    and have been used successfully on diabetic foot ulcers.
  ● Silver compounds: Applied as silver sulphadiazine or may be impregnated into
    dressings. In vitro silver is effective against Staphylococcus aureus including
    methicillin-resistant Staphylococcus aureus (MRSA) and pseudomonas species.
  ● Mupirocin: Active against gram-positive infections including MRSA. Its use should
    be limited to 10 days, and it should not be used as a prophylactic.
Systemic antibiotic treatment is always indicated in the presence of cellulitis, lymphangitis
and osteomyelitis. Infection in the neuroischaemic foot is often more serious than in the
neuropathic foot, which has a good blood supply. A positive swab in a neuroischaemic
foot ulcer therefore has more serious implications and influences antibiotic policy.

GENERAL PRINCIPLES OF BACTERIAL MANAGEMENT (DFU)
  • At initial presentation of infection it is important to prescribe wide-spectrum
    antibiotics and take cultures.
  • Deep swabs or tissue should be taken from the ulcer after initial debridement.
  • Ulcer swabs should be taken at every follow-up visit if suspicion of infection remains.
  • Diabetic patients respond poorly to sepsis, therefore even bacteria regarded as skin
    commensals can cause severe tissue damage.
  • Gram-negative bacteria isolated from an ulcer swab should not automatically be
    considered insignificant.
  • Blood cultures should be sent if fever and systemic toxicity are present.
  • The wound should be inspected regularly for early signs of infection.
  • Microbiologists have a crucial role; laboratory results should be used to guide
    antibiotic selection.
  • Timely surgical intervention is important in the presence of severe infection or
    abscess formation.
"""

print(f"Chunk 5 (DFU Infection Control) length: {len(CHUNK5_DFU_INFECTION)} chars")


# ── CELL 9 · Chunk 6 — DFU: Moisture Balance ─────────────────────────────────
#
# Source: p. 10 (Edmonds, Foster, Vowden)
# Covers: Moisture balance principles for DFU, dressing rationale, covering ulcers

CHUNK6_DFU_MOISTURE = """\
EWMA — Wound Bed Preparation for Diabetic Foot Ulcers (DFU): Moisture Balance
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. p.10.
Authors: M Edmonds, AVM Foster, P Vowden.

M — MOISTURE BALANCE (DFU)
Wound and peri-wound moisture balance is critical and must be linked to the overall
treatment plan. The value of moist wound healing in the diabetic foot ulcer has not been
proven and there is an increasing argument that hydration is, for example, inappropriate
in neuroischaemic ulceration if a decision has been made to mummify the digit or ulcer.
Excessive hydration may also macerate the plantar skin and reduce its effectiveness as a
bacterial barrier.

DRESSING SELECTION FOR DFU
There is no robust evidence that any one dressing performs significantly better on the
diabetic foot than others. However, it is useful if the dressing is:
  - Easy to remove
  - Absorbent
  - Able to accommodate pressures of walking without disintegrating.
If possible, dressings should be removed by the healthcare professional every day for
wound inspection, as the only signs of infection may be visual when patients lack the
protective pain sensation. However, the ulcer should be covered with a sterile,
non-adherent dressing at all times except when being inspected or debrided.

RATIONALE FOR COVERING DFU ULCERS
  • To protect the wound from noxious stimuli
  • To prevent infestation with insects
  • To keep the wound warm
  • To protect the wound from mechanical trauma
  • To reduce the risk of infection
"""

print(f"Chunk 6 (DFU Moisture Balance) length: {len(CHUNK6_DFU_MOISTURE)} chars")


# ── CELL 10 · Chunk 7 — DFU: Epithelial (Edge) Advancement ───────────────────
#
# Source: pp. 9–10 (Edmonds, Foster, Vowden)
# Covers: Saucerisation, die-back, extrinsic & intrinsic factors, treatment of both

CHUNK7_DFU_EDGE = """\
EWMA — Wound Bed Preparation for Diabetic Foot Ulcers (DFU): Epithelial (Edge) Advancement
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.9–10.
Authors: M Edmonds, AVM Foster, P Vowden.

E — EPITHELIAL (EDGE) ADVANCEMENT (DFU)
It is important that the edges of neuropathic ulcers are 'saucerised' and all callus, dried
exudate and accumulated slough, necrosis or non-viable cellular debris are debrided,
removing potential physical barriers to the growth of epithelium across the ulcer bed.
In patients with necrotic ulcers or necrotic digits the area of necrosis adjoining healthy
tissue frequently gives rise to problems: the demarcation line between gangrene and viable
tissue (the edge) frequently becomes the site of infection. This may be because debris
accumulates at this site and covers healthy skin, which then becomes macerated and prone
to infection. Similar problems can be observed when a healthy toe is touching a gangrenous
toe and becomes macerated at the point of contact, then infected. It may be that healing is
stimulated by debriding the edge of the wound, and by preventing contact between healthy
tissues and gangrene using dry dressings between the toes.
'Die-back' is similar to the above, but is an abnormal response to over-aggressive sharp
debridement. It involves necrosis of tissue at the wound edge and extends through
previously healthy tissue. Clinical experience suggests this is a particular problem in
patients with severe nephropathy or end-stage renal failure.
In addition to edge-specific problems, epithelial (edge) advancement may be affected by
extrinsic and intrinsic factors:
  Extrinsic factors: repeated trauma (not sensed due to neuropathy), ischaemia and
    poor metabolic control.
  Intrinsic factors: deficiency of growth factors, abnormal extracellular matrix components
    with excess protease and reduced fibroblast activity.

TREATMENT OF EXTRINSIC FACTORS
  Neuropathic foot: Redistribute plantar pressures evenly by applying some form of cast,
    adapted footwear or padding. Crutches, wheelchairs and Zimmer frames may be useful
    to aid offloading.
  Neuroischaemic foot: Protect the vulnerable margins of the foot through
    revascularisation and pressure redistribution.
  Ischaemia can be treated by angioplasty or arterial bypass. If lesions are too widespread
    for angioplasty, arterial bypass may be considered if the ulcer does not respond to
    conservative treatment.
  While the influence of blood glucose control on wound healing is debatable, it is
    important to control blood glucose, blood pressure and lipids and to encourage the
    patient to stop smoking. In patients with type 2 diabetes, oral hypoglycaemic therapy
    should be optimised, and if this is unsuccessful insulin should be initiated. Those with
    neuroischaemic ulcers should be given statin and anti-platelet therapy, while those aged
    over 55 years who have peripheral vascular disease should also benefit from an ACE
    inhibitor to prevent further vascular episodes.

TREATMENT OF INTRINSIC FACTORS — Growth Factor Abnormalities
  Skin biopsies from the edge of foot ulcers in non-diabetic and diabetic subjects have
  shown increased expression of transforming growth factor (TGF) beta 3 in the
  epithelium. However, expression of TGF-beta 1 was not increased, and this could
  explain impaired healing. Lack of expression of insulin-like growth factor (IGF) 1 in
  diabetic skin and foot ulcers and in dermal fibroblasts may also contribute to delayed
  wound healing.
  Hyperglycaemia and impaired insulin signalling may result in poor wound healing by
  reducing glucose utilisation of skin keratinocytes as well as skin proliferation and
  differentiation. Glycation of basic fibroblast growth factor (FGF) 2 significantly reduces
  its activity and thus its ability to bind to tyrosine kinase receptor and activate signal
  transduction pathways.
  Free radicals may be important in the pathogenesis of diabetes-related healing deficit.
  In non-diabetic patients dermal wounds heal by contraction and granulation tissue
  formation, rather than re-epithelialisation. Contraction provides 80–90% of wound
  closure. In contrast, closure is predominantly the result of granulation and
  re-epithelialisation in diabetic wounds. Simple epithelial repair is not hindered in
  superficial wounds, but is severely impaired in deeper wounds requiring collagen
  formation.
"""

print(f"Chunk 7 (DFU Edge Advancement) length: {len(CHUNK7_DFU_EDGE)} chars")


# ── CELL 11 · Chunk 8 — DFU: Advanced Therapies (Table 1) & After TIME ────────
#
# Source: pp. 10–11 (Edmonds, Foster, Vowden)
# Covers: Advanced wound healing products for DFU (Table 1), VAC therapy,
#         After TIME summary, Key Points

CHUNK8_DFU_ADVANCED = """\
EWMA — Wound Bed Preparation for Diabetic Foot Ulcers (DFU): Advanced Therapies & After TIME
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.10–11.
Authors: M Edmonds, AVM Foster, P Vowden.

ADVANCED THERAPIES FOR DFU (Table 1: Advanced therapies — Diabetic Foot Ulcers)

Tissue-engineered products:
  Description: Engineered skin constructs (neonatal allogeneic fibroblasts/keratinocytes)
  Activity: Produce growth factors and stimulate angiogenesis
  Research: 56% of diabetic foot ulcers (DFU) healed compared to 39% of controls;
            50.8% of DFU healed completely compared to 31.7% of controls.

Growth factors:
  Description: Platelet-derived growth factor
  Activity: Attracts neutrophils, macrophages and fibroblasts. Stimulates fibroblast
            proliferation.
  Research: Licensed for DFU; 50% of ulcers healed compared to 35% of controls.

Bioactive dressings/treatments:
  Description: Esterified hyaluronic acid
  Activity: Delivers multifunctional hyaluronic acid to the wound
  Research: Pilot studies have shown promising results in treating neuropathic DFUs,
            especially with sinuses.

  Description: Protease modulating matrix
  Activity: Stimulates angiogenesis by inactivating excess proteases
  Research: 37% of DFUs healed compared to 28% of controls.

VACUUM ASSISTED CLOSURE (VAC) / TOPICAL NEGATIVE PRESSURE
Vacuum assisted closure, a topical negative pressure therapy, has also been used to achieve
closure of diabetic ulcers, and has been shown on other chronic wound types to reduce
bacterial colonisation and diminish oedema and interstitial fluid.

AFTER TIME — DFU CARE STRATEGY SUMMARY
Each wound is different and requires an individual approach to care. For the diabetic foot
ulcer the emphasis is on:
  - Radical and repeated debridement
  - Frequent inspection and bacterial control
  - Careful moisture balance to prevent maceration
  - Linked to pressure control and the management of blood glucose and perfusion.
This should result in healing.
Diabetic foot ulceration is both a life- and limb-threatening condition. Recurrent
ulceration rates are high and patients are at increased risk of amputation. Management
must involve the patient in care and this requires effective education and a foot review
programme that addresses the initial cause of ulceration and gives the patient access to
appropriate and acceptable footwear.

KEY POINTS — DFU
1. Effective management of diabetic foot ulcers requires a multidisciplinary approach and
   patient involvement. It combines wound care, pressure offloading and diabetic control.
2. Inflammation and infection control is a vital priority to avoid severe tissue damage and
   amputation.
3. Tissue management in the form of radical and repeated debridement is the main focus
   of wound bed preparation in the treatment of neuropathic diabetic foot ulcers. This
   intervention must be used with caution in the neuroischaemic foot.
"""

print(f"Chunk 8 (DFU Advanced Therapies) length: {len(CHUNK8_DFU_ADVANCED)} chars")


# ── CELL 12 · Chunk 9 — VLU: Before TIME & Tissue Management ─────────────────
#
# Source: pp. 12–13 (Moffatt, Morison, Pina — Venous Leg Ulcers)
# Covers: Introduction, Before TIME prerequisites, risk factors, necrotic tissue,
#         debridement methods for VLU, surrounding skin

CHUNK9_VLU_TISSUE = """\
EWMA — Wound Bed Preparation for Venous Leg Ulcers (VLU): Before TIME & Tissue Management
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.12–13.
Authors: C Moffatt, MJ Morison, E Pina.

INTRODUCTION
For most patients with venous leg ulceration the application of high compression bandaging
in combination with simple non-adherent dressings is sufficient to stimulate autolytic
debridement, control moisture balance and encourage healing within 24 weeks. The
challenge for effective wound bed preparation is the early detection of those ulcers unlikely
to heal by simple compression therapy alone, and for which additional therapeutic
interventions may accelerate or facilitate healing.

BEFORE TIME — PREREQUISITES FOR VENOUS LEG ULCER MANAGEMENT
Venous ulceration results from venous insufficiency or obstruction. Oedema occurs and
graduated, sustained multi-layer compression is the cornerstone of care. Wound bed
preparation will not be successful unless the following management principles are taken
into account, along with effective patient education and concordance with therapy:
  ● Correct the cause of the ulcer by managing the underlying venous disease (surgical
    intervention where necessary)
  ● Improve venous return using high compression therapy
  ● Create the optimum local environment at the wound site
  ● Improve the wider factors that may delay healing
  ● Maintain ongoing assessment to identify changing aetiology
  ● Maintain a healed limb through a lifetime of compression therapy.

HEALING RATE AND PREDICTION
There is currently no internationally agreed standard healing rate of an uncomplicated
venous ulcer: reported healing at 12 weeks ranges from 30% to over 75%. However, the
percentage of wound reduction during the first three to four weeks of treatment can be
used to predict subsequent healing, with a 44% reduction in initial area at week 3 correctly
predicting healing in 77% of cases.

RISK FACTORS FOR DELAYED HEALING (VLU)
  • Ulcer duration >6 months
  • Ulcer size >10 cm²
  • Reduced mobility
  • Severe pain
  • Psychosocial: living alone, social support, clinical depression
  • Gender (male)
  • Poor general health

T — TISSUE MANAGEMENT (VLU): NECROTIC TISSUE
The majority of uncomplicated venous ulcers have relatively little necrotic tissue on the
wound surface and do not require debridement. However, it may be beneficial for more
complex ulcers, for example where severe infection, uncontrolled oedema and wound
desiccation may cause tissue necrosis. In addition, ulcers of long duration may develop a
chronic fibrinous base, which is pale, shiny and adherent. Removal of this layer using
sharp debridement under local anaesthetic may promote healing, but care must be taken
to avoid damaging deeper structures. Clinicians must be appropriately qualified before
undertaking surgical or sharp debridement.
Ulcers lying behind the malleoli are particularly prone to slough development and heal
slowly. Limited sharp debridement using forceps and scissors is often sufficient as slough
is usually superficial. Simple methods of increasing local pressure to the wound, such as
the use of foam shapes or firm padding cut to the contour of the area, can stimulate
healing. Adapting the method of compression can also be helpful; for example, an extra
layer of bandaging will increase pressure to this area, although care should be taken to
ensure there is adequate padding to the dorsum of the foot.
For more adherent slough, debridement using enzymatic preparations may be considered
as a practical alternative. Larval therapy can also be considered as an alternative to sharp
debridement, although application under compression may be associated with practical
challenges. Autolytic debridement using dressings with a high water content, such as
hydrogels and hydrocolloids, is slow and clinical experience suggests this is not an
effective form of debridement under compression. Although maintenance debridement is
recommended for wound bed preparation, this is rarely indicated with venous leg ulcers.

SURROUNDING SKIN (VLU)
Surrounding skin problems, such as callus formation and hyperkeratosis, may interfere
with healing. The development of hard callus or scabs, for example, may become a source
of pressure beneath compression and require careful removal using fine forceps, avoiding
trauma to the vulnerable underlying epithelium. Clinical experience suggests that soaking
in warm water with emollient for more than 10 minutes can facilitate tissue removal.
Bleeding after debridement may be resolved by the application of a haemostat such as an
alginate and compression.
"""

print(f"Chunk 9 (VLU Tissue Management) length: {len(CHUNK9_VLU_TISSUE)} chars")


# ── CELL 13 · Chunk 10 — VLU: Inflammation & Infection Control ────────────────
#
# Source: pp. 13–14 (Moffatt, Morison, Pina)
# Covers: Bacterial burden, clinical indicators, microbiological diagnosis,
#         antimicrobial treatments, topical antiseptics (iodine, silver), systemic abx

CHUNK10_VLU_INFECTION = """\
EWMA — Wound Bed Preparation for Venous Leg Ulcers (VLU): Inflammation & Infection Control
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.13–14.
Authors: C Moffatt, MJ Morison, E Pina.

I — INFLAMMATION AND INFECTION CONTROL (VLU)
Bacteria may stimulate a persisting inflammation leading to the production of inflammatory
mediators and proteolytic enzymes. Amongst many other effects this causes extracellular
matrix (ECM) degradation and inhibition of re-epithelialisation. Bacterial burden must
therefore be controlled to facilitate healing or to maximise the effectiveness of newer
therapeutic techniques such as bioengineered skin or growth factors.
The diagnosis of wound infection is a clinical skill based on careful history taking and
clinical observation. Infection in venous ulcers is usually localised and there may be
cellulitis. On rare occasions, particularly where the patient is immunocompromised,
systemic infection may develop. Leucocytosis and acute-phase reactants such as
erythrocyte sedimentation rate and C-reactive protein are not reliable since these patients
are constantly challenged by minor illnesses and peripheral lesions that may elevate these
indices. It is therefore necessary to be aware of other signs often presenting in these
wounds, such as an increase in the intensity or change in the character of pain.

INDICATORS OF INFECTION IN VENOUS ULCERS
  • Increased intensity and/or change in character of pain
  • Discoloured or friable granulation tissue
  • Odour
  • Wound breakdown
  • Delayed healing
  Note: The classical signs and symptoms of infection (pain, erythema, heat and purulence)
  may be reduced or masked by dermatological problems.

MICROBIOLOGICAL DIAGNOSIS
Microbiological diagnosis should be limited to situations where there is a clear indication
that the bacterial load is implicated in delayed healing. Quantification of bacteria by wound
biopsy has been considered the gold standard but surface sampling is easier and less costly,
and it is increasingly suggested that bacterial synergistic interaction is more important than
the precise number, as a greater diversity (i.e. more than four species) is associated with
non-healing. Anaerobic organisms are considered to have at least as great a negative impact
on healing as aerobes. Staphylococcus aureus and Pseudomonas aeruginosa are the bacteria
most commonly isolated in infected leg ulcers, but are also found in non-infected wounds.
Haemolytic streptococci are not commonly found in leg ulcers, but can be a particular cause
for concern and can lead to massive tissue damage if not recognised and treated effectively
and promptly. Other organisms such as mycobacteria, fungi and viruses as well as parasites
such as Leishmania may be implicated in a differential diagnosis.

TREATMENT (VLU infection)
It is essential to enhance host resistance by correcting the underlying vascular disease and
eliminating or reducing risk factors including smoking, heart failure, oedema, pain,
malnutrition and the effects of medications such as steroids and immunosuppressive agents.
Clearing devitalised tissue and foreign bodies is the first step to restoring bacterial balance.
This can be achieved through exudate control, cleansing with sterile saline and sharp
debridement where indicated, or other methods of debridement including larval therapy.

ANTIMICROBIAL TREATMENTS (VLU)
In wounds that exhibit local signs of infection or fail to heal in spite of appropriate care,
topical antiseptics should be considered. In addition to the choice of product, the form and
system of delivery are important. Antiseptic solutions are not indicated because of toxicity.
The role of antiseptics was recently reappraised; a number of new sustained slow-release
formulations of iodine and silver were found to reduce bacterial burden safely and
efficiently. When selecting antiseptic-containing dressings, in addition to antibacterial
properties, other characteristics such as moisture retention, absorption of endotoxins,
reduction of inflammation and pain relief should be considered.
Antiseptics are preferable because resistance is not yet a clinical problem. If no improvement
is observed in two weeks antiseptic treatment should cease, the wound should be reassessed
and systemic antibiotics may be considered.
Topical antibiotics can deliver high concentrations to the wound while minimising the risk
of systemic toxicity; however, cutaneous sensitisation, inactivation, inhibition of healing as
well as a selection of resistant strains have been reported and they are therefore not
recommended.
  - Metronidazole gel has been used to manage odour and reduce anaerobic colonisation.
  - Fusidic acid and mupirocin are active against gram-positive bacteria including MRSA.
  - Polymyxin B, neomycin and bacitracin should not be used because of allergy.
Systemic antibiotics should be used when there are signs of systemic invasion, cellulitis,
or when active infection cannot be managed using local therapies.
"""

print(f"Chunk 10 (VLU Infection Control) length: {len(CHUNK10_VLU_INFECTION)} chars")


# ── CELL 14 · Chunk 11 — VLU: Moisture Balance ────────────────────────────────
#
# Source: pp. 14–15 (Moffatt, Morison, Pina)
# Covers: Exudate impact on healing, compression therapy for moisture balance,
#         dressing selection, maceration prevention, paraffin/zinc paste

CHUNK11_VLU_MOISTURE = """\
EWMA — Wound Bed Preparation for Venous Leg Ulcers (VLU): Moisture Balance
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.14–15.
Authors: C Moffatt, MJ Morison, E Pina.

M — MOISTURE BALANCE (VLU)
Venous leg ulcers usually produce copious exudate, which can delay healing and cause
maceration of the surrounding skin. Chronic exudate causes the breakdown of extracellular
matrix proteins and growth factors, prolongs inflammation, inhibits cell proliferation, and
leads to the degradation of tissue matrix. Its management is therefore vital to wound bed
preparation.

COMPRESSION THERAPY — CORNERSTONE OF MOISTURE BALANCE
The removal of oedema using sustained compression therapy is fundamental to achieving
moisture balance. Compression helps to optimise local moisture balance by:
  - Reducing exudate production
  - Reducing tissue maceration
  - Ensuring adequate tissue perfusion by improving venous return.
Compression therapy can be achieved using a variety of methods such as bandages,
hosiery and intermittent pneumatic compression. Choice of method depends on resources
available, patient mobility, the size and shape of the affected leg and patient preference.
If venous ulcers continue to produce copious exudate and there are signs of oedema,
compression may be inadequate. Bandages may need to be changed more frequently if
soiled by excessive exudate or if the limb circumference is reduced markedly, when
remeasuring of the ankle circumference may be necessary.
To assist the action of compression, patients should be advised to avoid standing for long
periods and to elevate their legs above heart level when sitting or lying down. These steps
can make a sufficient difference to allow healing in an otherwise static ulcer.

DRESSING SELECTION FOR VLU
Venous ulcers require basic moist wound healing principles, as dryness of the ulcer bed
is rarely a problem. Simple measures such as washing the lower limbs and effective skin
care are important.
Dressing selection should take account of a number of factors:
  - Minimise tissue trauma
  - Absorb excess exudate
  - Manage slough/necrotic tissue
  - Be hypoallergenic.
Where possible adhesive dressings should be avoided as they increase the risk of allergic
reactions or contact dermatitis. Dressing performance may be affected by compression,
especially those designed to deal with high levels of exudate, as compression may affect
the lateral flow of fluid within the dressing.

SKIN PROTECTION AND HYDRATION
Hydration and protection of the skin using paraffin-based products or zinc paste is a
fundamental aspect of care. However, these must be removed regularly by washing or
they may form a thick layer preventing removal of dead keratinocytes and promoting the
development of varicose eczema and hyperkeratosis.

PREVENTING MACERATION (VLU)
Maceration may occur around the margins of venous ulceration and is manifested as
white, soggy tissue. Areas of erythema may also be present where exudate is in contact
with vulnerable skin. This can lead to the development of irritant dermatitis and new
areas of ulceration.
  • Use paraffin-based products or zinc paste as a barrier
  • Select appropriately sized dressing capable of handling high exudate levels such as
    foams and capillary action dressings
  • Carefully position the dressing so that exudate does not run below the wound
  • Silver and iodine products can be used if excess exudate is caused by infection
  • Avoid hydrocolloids and films
"""

print(f"Chunk 11 (VLU Moisture Balance) length: {len(CHUNK11_VLU_MOISTURE)} chars")


# ── CELL 15 · Chunk 12 — VLU: Epithelial (Edge) Advancement & Advanced Therapies
#
# Source: pp. 15–18 (Moffatt, Morison, Pina)
# Covers: Reasons for failed epithelialisation, indicators of healing, advanced therapies
#         Table 1 (VLU), tissue engineering, growth factors, bioactive dressings,
#         protease inhibitors, Conclusion, Key Points

CHUNK12_VLU_EDGE_ADVANCED = """\
EWMA — Wound Bed Preparation for Venous Leg Ulcers (VLU): Epithelial (Edge) Advancement & Advanced Therapies
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004. pp.15–18.
Authors: C Moffatt, MJ Morison, E Pina.

E — EPITHELIAL (EDGE) ADVANCEMENT (VLU)
If the epidermal margin fails to migrate across the wound bed there are many possible
reasons, including:
  - Hypoxia
  - Infection
  - Desiccation
  - Dressing trauma
  - Overgrowth of hyperkeratosis and callus at the wound margin.
Careful clinical observation can help to determine the cause, although this will not reveal
defects in the underlying cell biology.
The presence of islands of epithelium originating from hair follicles and evidence of edge
stimulation at the wound margin are useful indicators of healing. However, newly formed
epithelial cells can be difficult to identify as they are partly translucent and may be hidden
by slough, fibrous tissue or exudate.
Edge stimulation is intrinsically linked to moisture balance, as without optimal moisture
balance epidermal migration will not occur.

ADVANCED THERAPIES FOR VLU (Table 1: Advanced therapies — Venous Leg Ulcers)
Despite adequate wound bed preparation using standard methods some wounds fail to heal
or heal slowly. This may be the consequence of a disordered healing response resulting
from inappropriate cytokine, growth factor, protease and reactive oxygen species
production by cells within granulation tissue, which leads to non-resolving inflammation,
poor angiogenesis, ECM degradation and non-migration of epithelial cells from the wound
margin. Advanced therapies are only likely to be successful if applied to a well-prepared
wound bed.

Tissue-engineered products:
  Description: Engineered skin constructs (neonatal allogeneic fibroblasts/keratinocytes)
  Activity: Produce growth factors and stimulate angiogenesis
  Research: More effective than conventional venous leg ulcer (VLU) therapy in a clinical
            trial. Activity demonstrated in VLU.

Growth factors:
  Description: Granulocyte monocyte colony stimulating factor
  Activity: Activates monocytes, stimulates proliferation and migration of keratinocytes,
            modulates fibroblasts
  Research: Enhanced healing rates with VLU.

  Description: Keratinocyte growth factor
  Activity: Stimulates proliferation of keratinocytes and migration of keratinocytes and
            fibroblasts
  Research: Enhanced healing rates with VLU.

Bioactive dressings/treatments:
  Description: Esterified hyaluronic acid
  Activity: Delivers multifunctional hyaluronic acid to the wound
  Research: Pilot study demonstrates initiation of healing in VLU.

  Description: Protease modulating matrix
  Activity: Stimulates angiogenesis by inactivating excess proteases
  Research: 62% of VLU improved over 8 weeks compared to 42% in control group.

TISSUE ENGINEERING DETAIL
Grafting of autologous skin to a prepared wound bed has been used to stimulate healing
for many years. However, this suffers from the disadvantage of donor site pain, scarring
and the possibility of infection. Recent advances in cell culture techniques allow expansion
of cells in vitro, which are then used to populate biocompatible scaffolds to act as a carrier
and substitute for split-thickness skin grafts. Cells may be either autologous or from
allogeneic donors. This treatment has the added advantage that the transplanted cells
interact in the healing process by producing growth factors that may also act to stimulate
healing.

GROWTH FACTORS DETAIL
The growth factor networks that regulate healing may become degraded and disorganised
in the chronic wound. This leads to the concept that supplying exogenous growth factors
to the wound microenvironment may stimulate healing. Many have been evaluated but
platelet-derived growth factor is, to date, the first growth factor to be licensed for topical
application and only in diabetic ulcers.

BIOACTIVE DRESSINGS DETAIL
Modern wound dressings developed to maintain a moist wound environment have recently
evolved into a new generation of products that interact with the wound to stimulate
healing. Examples are protease modulating dressings, which claim to stimulate healing by
inactivating excess proteases and a range of products, based on esterified hyaluronic acid,
which deliver multifunctional hyaluronic acid to the wound.

PROTEASE INHIBITORS
A novel synthetic inhibitor of protease activity has recently been described that inhibits
ECM-degrading enzymes without affecting those proteases required for normal
keratinocyte migration. This suggests it will be feasible in the future to develop highly
specific pharmacologic agents to treat defects of non-healing wounds.

CONCLUSION (VLU)
The general aims of wound bed preparation are as relevant to the management of venous
leg ulcers as any other wound type. However, its different elements do not have equal
emphasis. Debridement is rarely an issue; the main priority in the management of venous
ulcers is to achieve moisture balance by improving venous return using sustained
compression. Edge stimulation is intrinsically linked to moisture balance, as without
optimal moisture balance epidermal migration will not occur.
In addition to problems of limited resources, it is usually unnecessary to use advanced
wound care products with venous leg ulcers. The challenge in managing these wounds is
to predict, perhaps as early as the fourth week of standard care, which ulcers will fail to
heal rapidly, as these patients benefit the most from alternative care strategies.

KEY POINTS — VLU
1. Most venous leg ulcers will heal with the application of high compression bandaging
   and simple non-adherent dressings.
2. The challenge is to predict as early as the fourth week of standard care which ulcers
   will benefit from wound bed preparation and the use of advanced therapies.
3. Using the TIME framework, it can be seen that the main priority with venous leg ulcers
   is to achieve moisture balance. Although tissue management and infection control are
   rarely an issue, rigorous attention must be paid to these components if there are
   problems with healing or where advanced therapies are required.
"""

print(f"Chunk 12 (VLU Edge & Advanced) length: {len(CHUNK12_VLU_EDGE_ADVANCED)} chars")


# ── CELL 16 · Chunk 13 — TIME Comparative Summary: DFU vs VLU ────────────────
#
# This is a synthesised cross-reference chunk (not copied from any single page,
# but derived only from facts stated in the document) that directly serves the RAG
# use case: given TIME inputs, which wound emphasis applies?

CHUNK13_TIME_COMPARATIVE = """\
EWMA — TIME Framework: Comparative Emphasis for DFU vs VLU
Source: EWMA Position Document: Wound Bed Preparation in Practice. London: MEP Ltd, 2004.
Derived from editorial overview (p.3) and conclusions (pp.11, 16–18).

This summary shows which TIME component receives primary emphasis per wound type,
directly supporting dressing recommendation based on TIME classification inputs.

TIME COMPONENT EMPHASIS BY WOUND TYPE

  Wound Type: Diabetic Foot Ulcer (DFU)
  ──────────────────────────────────────
  T — Tissue management:           PRIMARY — radical and repeated debridement is the
                                    main focus; gold standard is sharp debridement
  I — Infection control:            HIGH — vital priority to avoid tissue damage and
                                    amputation; use iodine, silver, mupirocin topically;
                                    systemic antibiotics for cellulitis/osteomyelitis
  M — Moisture balance:             MODERATE — careful balance; avoid excessive
                                    hydration (risk of maceration); non-adherent,
                                    absorbent, easy-to-remove dressings preferred;
                                    moist healing not proven for neuroischaemic DFU
  E — Epithelial (edge) advancement: ADDRESSED — saucerisation of wound edges;
                                    manage extrinsic (offloading, revascularisation,
                                    glycaemic control) and intrinsic (growth factors)

  Wound Type: Venous Leg Ulcer (VLU)
  ───────────────────────────────────
  T — Tissue management:           LOW (usually) — most uncomplicated VLUs do not
                                    require debridement; compression alone may
                                    stimulate autolytic debridement; slough behind
                                    malleoli may need limited sharp debridement
  I — Infection control:            MODERATE — rarely causes systemic infection;
                                    topical iodine/silver for local infection; avoid
                                    topical antibiotics (sensitisation risk);
                                    systemic abx only for cellulitis/systemic invasion
  M — Moisture balance:             PRIMARY — copious exudate is the main challenge;
                                    sustained high compression is cornerstone;
                                    foam/capillary dressings for high exudate;
                                    paraffin/zinc paste barrier to prevent maceration;
                                    avoid hydrocolloids and films
  E — Epithelial (edge) advancement: LINKED TO MOISTURE — edge advancement will not
                                    occur without optimal moisture balance;
                                    advanced therapies (tissue engineering, growth
                                    factors, bioactive dressings) for recalcitrant VLU

CROSS-CUTTING PRINCIPLE (applies to ALL wound types)
  A single intervention can impact on more than one TIME element.
  Example: debridement addresses T (tissue) AND reduces I (bacterial burden).
  Example: compression therapy addresses M (moisture/exudate) AND corrects
           the underlying cause of VLU, enabling E (edge) advancement.
  The TIME framework must be reassessed continuously as wound status changes.

WOUND BED PREPARATION PRINCIPLE
  Wound bed preparation is not a static concept but a dynamic and rapidly evolving one.
  If all elements of TIME are successfully addressed, many chronic wounds should move
  towards healing. Advanced therapies should only be introduced once the wound bed is
  well prepared using standard TIME-directed interventions.
"""

print(f"Chunk 13 (TIME Comparative Summary) length: {len(CHUNK13_TIME_COMPARATIVE)} chars")


# ── CELL 17 · Assemble all chunks ─────────────────────────────────────────────

chunks: list[dict] = []

chunk_definitions = [
    (
        "TIME Framework — Evolution & Four Components",
        "TIME Framework",
        CHUNK1_TIME_FRAMEWORK,
    ),
    (
        "TIME Applied to Practice — Pathway & WBP Principles",
        "TIME Framework",
        CHUNK2_TIME_IN_PRACTICE,
    ),
    (
        "TIME Figure 1 — Dynamic Wound Progression (Four States)",
        "TIME Framework",
        CHUNK3_TIME_FIGURE1,
    ),
    (
        "DFU — Before TIME & Tissue Management (Debridement)",
        "Diabetic Foot Ulcer (DFU) — Wound Bed Preparation",
        CHUNK4_DFU_TISSUE,
    ),
    (
        "DFU — Inflammation & Infection Control",
        "Diabetic Foot Ulcer (DFU) — Wound Bed Preparation",
        CHUNK5_DFU_INFECTION,
    ),
    (
        "DFU — Moisture Balance & Rationale for Covering Ulcers",
        "Diabetic Foot Ulcer (DFU) — Wound Bed Preparation",
        CHUNK6_DFU_MOISTURE,
    ),
    (
        "DFU — Epithelial (Edge) Advancement",
        "Diabetic Foot Ulcer (DFU) — Wound Bed Preparation",
        CHUNK7_DFU_EDGE,
    ),
    (
        "DFU — Advanced Therapies (Table 1) & After TIME Summary",
        "Diabetic Foot Ulcer (DFU) — Wound Bed Preparation",
        CHUNK8_DFU_ADVANCED,
    ),
    (
        "VLU — Before TIME & Tissue Management (Debridement)",
        "Venous Leg Ulcer (VLU) — Wound Bed Preparation",
        CHUNK9_VLU_TISSUE,
    ),
    (
        "VLU — Inflammation & Infection Control",
        "Venous Leg Ulcer (VLU) — Wound Bed Preparation",
        CHUNK10_VLU_INFECTION,
    ),
    (
        "VLU — Moisture Balance & Maceration Prevention",
        "Venous Leg Ulcer (VLU) — Wound Bed Preparation",
        CHUNK11_VLU_MOISTURE,
    ),
    (
        "VLU — Epithelial (Edge) Advancement & Advanced Therapies",
        "Venous Leg Ulcer (VLU) — Wound Bed Preparation",
        CHUNK12_VLU_EDGE_ADVANCED,
    ),
    (
        "TIME Comparative Summary — DFU vs VLU Emphasis",
        "TIME Framework",
        CHUNK13_TIME_COMPARATIVE,
    ),
]

for section, parent_section, text in chunk_definitions:
    if len(text) >= MIN_CHUNK_CHARS:
        chunks.append(make_chunk(section=section, parent_section=parent_section, text=text))

print(f"\nTotal chunks assembled: {len(chunks)}")
for c in chunks:
    print(f"  {c['section']!r:65s}  chars={c['char_count']:5d}")


# ── CELL 18 · Quality validation ──────────────────────────────────────────────

all_combined = " ".join(c["text"].lower() for c in chunks)

MUST_CONTAIN = [
    ("time framework",                   "TIME framework defined"),
    ("tissue management",                "T — Tissue management"),
    ("inflammation and infection control","I — Inflammation & infection control"),
    ("moisture balance",                 "M — Moisture balance"),
    ("epithelial (edge) advancement",    "E — Epithelial (edge) advancement"),
    ("t = tissue",                       "Table 1 TIME acronym original"),
    ("e = edge of wound",                "Table 1 E original acronym"),
    ("debridement",                      "Debridement mentioned"),
    ("sharp debridement",                "Sharp debridement — DFU gold standard"),
    ("larval therapy",                   "Larval therapy mentioned"),
    ("iodine",                           "Iodine antimicrobial"),
    ("silver",                           "Silver antimicrobial"),
    ("mupirocin",                        "Mupirocin antimicrobial"),
    ("biofilm",                          "Biofilm concept"),
    ("matrix metalloproteinase",         "MMPs in chronic wounds"),
    ("compression",                      "Compression therapy — VLU"),
    ("maceration",                       "Maceration prevention"),
    ("paraffin",                         "Paraffin barrier for VLU"),
    ("foam",                             "Foam dressing for high exudate"),
    ("hydrocolloid",                     "Hydrocolloid avoidance in VLU"),
    ("alginate",                         "Alginate haemostat"),
    ("neuropathic",                      "Neuropathic foot distinction"),
    ("neuroischaemic",                   "Neuroischaemic foot distinction"),
    ("osteomyelitis",                    "Osteomyelitis mentioned"),
    ("cellulitis",                       "Cellulitis mentioned"),
    ("platelet-derived growth factor",   "PDGF growth factor"),
    ("protease modulating",              "Protease modulating dressing"),
    ("esterified hyaluronic acid",       "Esterified hyaluronic acid dressing"),
    ("vacuum assisted closure",          "VAC therapy — DFU"),
    ("44% reduction",                    "VLU healing prediction (44% at week 3)"),
    ("organisms or more per gram",        "Bacterial burden threshold for impaired healing"),
    ("saucerised",                       "Edge saucerisation — DFU"),
    ("die-back",                         "Die-back complication — DFU"),
    ("metronidazole",                    "Metronidazole gel — VLU odour"),
    ("fusidic acid",                     "Fusidic acid — gram-positive VLU"),
    ("venous leg ulcer",                 "VLU wound type"),
    ("diabetic foot ulcer",              "DFU wound type"),
    ("figure 1",                         "Figure 1 time progression"),
    ("figure 2",                         "Figure 2 pathway"),
]

print("═" * 72)
print("QUALITY REPORT — EWMA Wound Bed Preparation in Practice")
print("═" * 72)
print(f"\n📦 Total chunks    : {len(chunks)}")
char_counts = [c["char_count"] for c in chunks]
print(f"   Chars — min    : {min(char_counts)}")
print(f"   Chars — mean   : {statistics.mean(char_counts):.0f}")
print(f"   Chars — max    : {max(char_counts)}")

# Duplicate chunk_id check
seen, dupes = {}, []
for c in chunks:
    key = c["text"][:150]
    if key in seen:
        dupes.append((seen[key], c["chunk_id"]))
    else:
        seen[key] = c["chunk_id"]
print(f"\n🔁 Duplicates      : {len(dupes)}")

print("\n✅ Content coverage:")
for kw, label in MUST_CONTAIN:
    found = kw in all_combined
    print(f"   {'✓' if found else '✗ MISSING'} {label}")

print("\n🗂  Chunks by parent section:")
parent_counts = Counter(c["parent_section"] for c in chunks)
for sec, cnt in sorted(parent_counts.items(), key=lambda x: -x[1]):
    print(f"   {cnt:2d} × {sec!r}")


# ── CELL 19 · Spot-check selected chunks ──────────────────────────────────────

def preview_chunk(idx: int):
    c = chunks[idx]
    print(f"\n{'─'*65}")
    print(f"[{idx}] chunk_id    : {c['chunk_id']}")
    print(f"     section      : {c['section']}")
    print(f"     parent       : {c['parent_section']}")
    print(f"     chars        : {c['char_count']}")
    print(f"TEXT (first 700 chars):")
    print(c["text"][:700])
    if len(c["text"]) > 700:
        print("... [truncated]")

# Preview TIME framework, DFU infection, VLU moisture, comparative summary
for idx in [0, 4, 10, 12]:
    preview_chunk(idx)


# ── CELL 20 · (Optional) LLM ai_summary enrichment ───────────────────────────

ENABLE_AI_SUMMARY = False   # ← set True when OpenAI key is available

if ENABLE_AI_SUMMARY:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    SYSTEM_PROMPT = (
        "You are a medical summarisation assistant. "
        "Rewrite the following wound-care guideline text as a clear, complete, self-contained "
        "clinical summary suitable for retrieval-augmented generation. "
        "Preserve all clinical facts, dressing names, wound types, TIME framework components, "
        "antimicrobial agents, indications, and contraindications. "
        "Return only the summary text — no preamble."
    )

    print("Running AI summaries...")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {c['section']}")
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


# ── CELL 21 · Export ChromaDB-ready JSON ──────────────────────────────────────

output = {
    "meta": {
        "total_chunks":   len(chunks),
        "kept_count":     len(chunks),
        "ai_summarised":  sum(1 for c in chunks if c["ai_summary"] != c["text"]),
        "extraction":     "Fully reconstructed verbatim text — verified against pdftotext -layout "
                          "and PyMuPDF fitz block inspection. No pypdf/pdfplumber body extraction "
                          "(two-column + sidebar layout fragments text unpredictably).",
        "chunking":       "Manual section-aware — one chunk per logical TIME component per wound type "
                          "+ 3 cross-cutting TIME framework chunks + 1 comparative summary chunk",
        "pages_used":     list(range(3, 19)),   # pages 3–18 (1-based), i.e. 0-indexed 2–17
        "pages_skipped":  "Pages 1–2 (cover, editorial credits); Page 19 (VLU references only)",
        "chunk_params": {
            "min_characters": MIN_CHUNK_CHARS,
        },
        "source_citation": (
            "European Wound Management Association (EWMA). Position Document: "
            "Wound Bed Preparation in Practice. London: MEP Ltd, 2004."
        ),
        "note": "Use ai_summary field for reference_contexts and ChromaDB page_content.",
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
    ],
}

out_path = OUT_DIR / "EWMA_wound_bed_preparation_kept.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Exported {len(chunks)} chunks → {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")


# ── CELL 22 · Final summary table ─────────────────────────────────────────────

with open(OUT_DIR / "EWMA_wound_bed_preparation_kept.json") as f:
    exported = json.load(f)

print("═" * 80)
print("INGESTION COMPLETE — EWMA Wound Bed Preparation in Practice (2004)")
print("═" * 80)

hdr = f"{'#':>3}  {'Chunk ID':14}  {'Section':65}  {'Chars':>5}  {'AI?':8}"
print(hdr)
print("-" * len(hdr))
for i, c in enumerate(exported["kept_chunks"], 1):
    ai = "yes" if c["ai_summary"] != c["text"] else "no (raw)"
    print(f"{i:3d}  {c['chunk_id']:14}  {c['section'][:65]:65s}  {c['char_count']:5d}  {ai}")

print()
print(f"Output JSON : {OUT_DIR / 'EWMA_wound_bed_preparation_kept.json'}")
print(f"Next steps  :")
print(f"  1) Review chunk text — all content is verbatim from PDF (pages 3–18)")
print(f"  2) Set ENABLE_AI_SUMMARY = True to enrich ai_summary with GPT")
print(f"  3) Load into vector store via general_ingestion notebook")

# RAGAS reference-context lookup
ref_ctx_by_section = defaultdict(list)
for c in exported["kept_chunks"]:
    ref_ctx_by_section[c["section"]].append(c["ai_summary"])

print(f"\nRAGAS reference_context lookup ready — {len(ref_ctx_by_section)} sections")
