"""
wound_app_02_v4.py  —  VerdaSense Clinical RAG Pipeline (v4_02 · Classifier Fixes)
══════════════════════════════════════════════════════════════════════════════
V4_02 CHANGES vs v4_01 (hybrid BM25 baseline)
──────────────────────────────────────────────
[FIX 1] EXPANDED SUBCLINICAL INFECTION KEYWORD LIST  (classify_wound)
        Old antibiotic_triggers: ["infect", "purulent", "pus", "fever", "sepsis", "cellulitis"]
        New antibiotic_triggers: adds "pain", "painful", "warm", "warmth", "redness",
          "erythema", "swelling", "oedema", "increased exudate", "malodour", "malodorous",
          "delayed healing", "non-healing", "nonhealing", "slowing", "not healing"
        These are standard T.I.M.E. 'I' (Infection/Inflammation) signs that GPs
        describe in free-text notes rather than selecting from a dropdown.
        When these keywords appear in notes AND structured input is "Not infected",
        the classifier flags antibiotic_required=True AND injects a
        CLINICAL ALERT block into the prompt so the LLM reasons about subclinical
        infection risk explicitly.
        Target: cat_d_notes_infection_override (antibiotic FAIL)

[FIX 2] CONSTRAINED DIABETIC ESCALATION  (classify_wound)
        Old: "diabetic" keyword alone → escalate wound_type to 7 (referral + abx)
        New: "diabetic" keyword only escalates referral (not wound type) unless
          NV >= 25% OR structured infection = "Locally infected".
          For clean granulating diabetic wounds (NV < 25%, not infected), the
          wound type stays at its base (1 or 2), referral is still set True
          (conservative practice), and etiology="diabetic_foot" is added to the
          classifier output so the generation prompt receives a targeted etiology
          note without changing the algorithm chunk being retrieved.
        Target: cat_d_notes_diabetic_nonhealing (classifier over-escalation)

[REMOVED] Post-generation verifier LLM call (G3)
        The verifier called a second gpt-4o-mini to check 5 binary questions,
        then optionally a third call for correction. In practice it fired a
        correction in 1/56 cases across v4_00 + v4_01 and that single correction
        still did not resolve the safety failure. Cost ~30% extra LLM calls
        for near-zero benefit. Removed entirely.
        Safety-critical phrases (referral, antibiotic) are guaranteed by the
        deterministic G2 mandatory injection block — no second LLM needed.

Everything else is identical to v4_01:
  - BM25 hybrid retrieval (Sub-queries B + C) — same as v4_01
  - Multi-axis sub-query retrieval (A: dense, B+C: hybrid)
  - Binding algorithm block in prompt (G1)
  - Conditional mandatory referral/antibiotic injection (G2)
  - Narrative query builder
  - SYSTEM_PROMPT grounding rules
  - Response contract (verifier_output field kept but simplified)

Response contract (v4_02 — backwards-compatible with v4_01):
  {
    "result":             str,
    "sources":            list[str],
    "chunk_texts":        list[str],
    "confidence_score":   float,
    "confidence_label":   str,
    "retrieval_notes":    list[str],
    "tissue_breakdown":   dict,
    "reranker_scores":    list,
    "clinical_flags":     dict,
    "narrative_query":    str,
    "classifier_output":  dict,     ← now includes etiology field
    "verifier_output":    dict,     ← always {"ran": false, "reason": "removed in v4_02"}
  }
"""

import os
import json
import re
import torch
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LC_Doc
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# ── Embedding model — must match ingestion ─────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Load vector store ──────────────────────────────────────────────────────────
def load_wound_db(
    persist_directory: str = r"C:\Users\GIGA\OneDrive - Universiti Malaya\Documents\rag-for-beginners\db_wound_care_v3",
):
    print(f"Loading Wound Care KB from {persist_directory}...")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

db = load_wound_db()

# ── BM25 index — built once at startup from entire KB ─────────────────────────
print("Building BM25 index from KB...")
_raw      = db.get()
_all_docs = [
    LC_Doc(page_content=pc, metadata=meta)
    for pc, meta in zip(_raw["documents"], _raw["metadatas"])
]
_bm25_retriever = BM25Retriever.from_documents(_all_docs)
print(f"  BM25 index built over {len(_all_docs)} documents")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION  (unchanged from v4_01)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_tissue_percentages(necrotic_pct: float, slough_pct: float, granulation_pct: float) -> dict:
    total = necrotic_pct + slough_pct + granulation_pct
    if total == 0:
        return {
            "tissue_label":    "insufficient tissue data",
            "clinical_tissue": "unknown",
            "necrotic_pct":    0.0,
            "slough_pct":      0.0,
            "granulation_pct": 0.0,
            "non_viable_pct":  0.0,
        }
    n = necrotic_pct    / total * 100
    s = slough_pct      / total * 100
    g = granulation_pct / total * 100
    nv = n + s

    if n >= 50:
        clinical_tissue = "predominantly necrotic wound"
    elif s >= 50:
        clinical_tissue = "sloughy fibrinous wound bed"
    elif g >= 70:
        clinical_tissue = "healthy granulating wound bed"
    elif n >= 25 and s >= 25:
        clinical_tissue = "mixed necrotic and slough tissue"
    else:
        clinical_tissue = "mixed wound bed tissue"

    return {
        "tissue_label":    clinical_tissue,
        "clinical_tissue": clinical_tissue,
        "necrotic_pct":    round(n, 1),
        "slough_pct":      round(s, 1),
        "granulation_pct": round(g, 1),
        "non_viable_pct":  round(nv, 1),
    }


def normalize_infection(label: str) -> str:
    m = label.lower().strip()
    if "not" in m or "no" in m:
        return "Not infected"
    return "Locally infected"


def normalize_moisture(label: str) -> str:
    m = label.lower().strip()
    if m in ("high", "high exudate"):
        return "High exudate"
    elif m in ("low", "low exudate", "dry"):
        return "Dry"
    else:
        return "Moderate exudate"


def normalize_edge(label: str) -> str:
    m = label.lower().strip()
    if "non" in m or "not" in m or "stall" in m:
        return "Non-advancing wound edge"
    return "Advancing wound edge"


# ══════════════════════════════════════════════════════════════════════════════
# [v4 R1 — UPDATED v4_02] CLINICAL PRE-CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

WOUND_TYPE_QUERY_PHRASES = {
    1: "clean granulating wound dry minimal exudate no infection wound type 1 dressing recommendation",
    2: "clean granulating wound high exudate heavily draining no infection wound type 2 dressing recommendation",
    3: "infected wound dry low exudate antimicrobial dressing wound type 3 antibiotic recommendation",
    4: "infected wound high exudate wet antimicrobial dressing wound type 4 antibiotic recommendation",
    5: "necrotic sloughy wound dry low exudate debridement no infection wound type 5 dressing recommendation",
    6: "necrotic sloughy wound high exudate wet referral hospital wound type 6 dressing recommendation",
    7: "necrotic infected wound dry low exudate referral hospital antibiotic wound type 7 dressing recommendation",
    8: "necrotic infected wound high exudate wet referral hospital antibiotic wound type 8 dressing recommendation",
}

# ── [FIX 1] Expanded subclinical infection triggers ───────────────────────────
# Old list had only explicit infection terms. GPs write subclinical signs in
# free text — these are all standard T.I.M.E. 'I' signs that should trigger
# antimicrobial consideration even when the structured dropdown says "Not infected".
#
# Design notes:
# - "infect" alone is NOT included — it matches inside "No infection currently",
#   causing false positives. Use "infected" or "infection signs" as exact phrases instead.
# - Each trigger is checked as a substring with enough specificity to avoid common
#   false positive contexts (e.g. "warm" is specific enough; "red" alone is too short).
# - "non-healing" and "not healing" correctly fire — delayed healing in a diabetic
#   or chronic wound IS a subclinical infection signal worth flagging.
_ANTIBIOTIC_TRIGGERS = [
    # Explicit infection terms — specific enough to avoid negation false positives
    "purulent", "pus", "fever", "sepsis", "cellulitis",
    "infected wound", "wound infection", "signs of infection",
    # [FIX 1] Subclinical / local inflammation signs (T.I.M.E. 'I' component)
    "increasing pain", "worsening pain", "more painful", "painful over",
    "warmth", "warm to touch", "warm around",
    "redness", "erythema", "perilesional", "peri-wound",
    "swelling", "oedema", "edema",
    "increased exudate", "more exudate", "exudate increasing",
    "malodour", "malodorous", "offensive odour", "offensive smell",
    "non-healing", "nonhealing", "not healing", "failing to heal",
    "deteriorating", "getting worse", "worsening wound",
]

# ── [FIX 2] Referral triggers — unchanged (burns, diabetic handled separately) ─
_REFERRAL_TRIGGERS = [
    "hospital",  "refer",    "specialist", "chronic",
    "burns",     "burn",     "deep",       "full thickness",
    # "diabetic", "non-heal" etc. removed from here — handled by diabetic logic below
]

# Diabetic-specific referral triggers (separate from wound type escalation)
_DIABETIC_TRIGGERS = ["diabetic", "diabetes", "neuropath", "peripheral arterial"]


def classify_wound(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm:  str,
    notes:          str = "",
) -> dict:
    """
    [v4_02] Deterministic clinical pre-classifier.

    FIX 1: Expanded subclinical infection keyword list.
    FIX 2: Constrained diabetic escalation.
      - "diabetic" alone no longer escalates wound_type.
      - Diabetic + clean wound (NV < 25%, not infected) → base wound type,
        referral=True, etiology="diabetic_foot".
      - Diabetic + necrosis (NV >= 25%) OR diabetic + infected → wound type
        escalation proceeds normally.

    Returns:
        {
            "wound_type":            int (1–8),
            "referral_required":     bool,
            "antibiotic_required":   bool,
            "subclinical_infection": bool,   ← NEW: True when fix 1 fires
            "etiology":              str,    ← NEW: "generic" | "diabetic_foot" | "burn" | "skin_tear"
            "escalation_reason":     str | None,
            "algorithm_query":       str,
            "classifier_notes":      str,
        }
    """
    nv       = tissue_profile["non_viable_pct"]
    infected = (infection_norm == "Locally infected")
    high_exu = (moisture_norm == "High exudate")
    notes_lc = notes.lower()

    # ── Detect etiology from notes ─────────────────────────────────────────────
    # Checked before primary classification so etiology influences escalation logic
    is_diabetic  = any(t in notes_lc for t in _DIABETIC_TRIGGERS)
    is_burn      = any(t in notes_lc for t in ["burn", "burns", "scald"])
    is_skin_tear = any(t in notes_lc for t in ["skin tear", "skin-tear", "skintear",
                                                 "fragile skin", "papery skin", "elderly skin",
                                                 "tear", "flap"])

    if is_diabetic:
        etiology = "diabetic_foot"
    elif is_burn:
        etiology = "burn"
    elif is_skin_tear:
        etiology = "skin_tear"
    else:
        etiology = "generic"

    # ── Primary classification — MOH T.I.M.E. algorithm ─────────────────────
    # Correct threshold is NV >= 25% per the MOH wound type matrix.
    # Pre-existing bug in v4_00/v4_01 used mixed thresholds leaving NV 25-49%
    # non-infected in the else clause (wrongly Type 7/8). Fixed here.
    nv_high = nv >= 25
    if not nv_high and not infected:
        wound_type = 2 if high_exu else 1
        referral   = False
        antibiotic = False
    elif not nv_high and infected:
        wound_type = 4 if high_exu else 3
        referral   = False
        antibiotic = True
    elif nv_high and not infected:
        wound_type = 6 if high_exu else 5
        referral   = high_exu
        antibiotic = False
    else:
        wound_type = 8 if high_exu else 7
        referral   = True
        antibiotic = True

    escalation_reason  = None
    subclinical_infection = False

    # ── [FIX 2] Diabetic escalation — constrained ─────────────────────────────
    # Old: "diabetic" alone → escalate wound_type (wrong for clean diabetic wounds)
    # New: diabetic + clean wound → referral only, wound_type unchanged
    #      diabetic + NV >= 25% OR infected → normal escalation (safe to proceed)
    if is_diabetic and not referral:
        referral          = True
        escalation_reason = "Notes indicate diabetic patient — referral escalated (conservative practice)"
        # Do NOT change wound_type for clean granulating diabetic wounds.
        # The etiology="diabetic_foot" field will inject a targeted prompt note.
        # Only escalate wound_type if there is already meaningful necrosis or infection:
        if nv >= 25 and not infected:
            # Borderline necrosis in a diabetic foot — reclassify upward conservatively
            wound_type        = 6 if high_exu else 5
            escalation_reason += " | NV >= 25% in diabetic → Type 5/6"
        # If infected: primary classification already handled Type 7/8 — no change needed

    # ── Non-diabetic referral triggers (burns, deep, specialist etc.) ──────────
    if not referral:
        for trigger in _REFERRAL_TRIGGERS:
            if trigger in notes_lc:
                referral          = True
                escalation_reason = f"Notes contain '{trigger}' — referral escalated"
                break

    # ── [FIX 1] Subclinical infection detection ────────────────────────────────
    # Only fires when structured input says "Not infected" (otherwise antibiotic
    # is already set from primary classification).
    if not antibiotic:
        matched_triggers = [t for t in _ANTIBIOTIC_TRIGGERS if t in notes_lc]
        if matched_triggers:
            antibiotic            = True
            subclinical_infection = True
            trigger_str           = ", ".join(f"'{t}'" for t in matched_triggers[:3])
            esc_note = f"Notes subclinical infection signals ({trigger_str}) — antibiotic escalated"
            escalation_reason     = (escalation_reason + " | " + esc_note
                                     if escalation_reason else esc_note)

    classifier_notes = (
        f"NV={nv:.1f}%, infected={infected}, high_exudate={high_exu}, "
        f"etiology={etiology} "
        f"→ Wound Type {wound_type}, referral={referral}, abx={antibiotic}"
    )
    if subclinical_infection:
        classifier_notes += " | SUBCLINICAL INFECTION RISK (notes keywords)"
    if escalation_reason:
        classifier_notes += f" | ESCALATION: {escalation_reason}"

    return {
        "wound_type":            wound_type,
        "referral_required":     referral,
        "antibiotic_required":   antibiotic,
        "subclinical_infection": subclinical_infection,
        "etiology":              etiology,
        "escalation_reason":     escalation_reason,
        "algorithm_query":       WOUND_TYPE_QUERY_PHRASES[wound_type],
        "classifier_notes":      classifier_notes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE QUERY BUILDER  (unchanged from v4_01)
# ══════════════════════════════════════════════════════════════════════════════

def build_narrative_query(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm:  str,
    edge_norm:      str,
    notes:          str = "",
) -> str:
    g  = tissue_profile["granulation_pct"]
    n  = tissue_profile["necrotic_pct"]
    s  = tissue_profile["slough_pct"]
    nv = tissue_profile["non_viable_pct"]

    if g >= 70:
        tissue_phrase = (
            f"a clean granulating wound bed ({g:.0f}% granulation tissue, "
            f"minimal non-viable tissue)"
        )
    elif n >= 50:
        tissue_phrase = (
            f"a predominantly necrotic wound ({n:.0f}% necrosis, "
            f"{s:.0f}% slough, {nv:.0f}% non-viable total)"
        )
    elif s >= 50:
        tissue_phrase = (
            f"a heavily sloughy wound ({s:.0f}% yellow slough, "
            f"{n:.0f}% necrosis, {nv:.0f}% non-viable total)"
        )
    else:
        tissue_phrase = (
            f"a mixed wound bed ({n:.0f}% necrosis, {s:.0f}% slough, "
            f"{g:.0f}% granulation, {nv:.0f}% non-viable tissue)"
        )

    if infection_norm == "Locally infected":
        infection_phrase = "with signs of local wound infection"
    else:
        infection_phrase = "with no signs of infection"

    moisture_map = {
        "High exudate":    "producing high levels of exudate requiring high-absorbency dressings",
        "Dry":             "presenting with dry to minimal exudate requiring moisture-donating dressings",
        "Moderate exudate":"producing moderate exudate requiring balanced moisture management",
    }
    moisture_phrase = moisture_map.get(moisture_norm, f"with {moisture_norm.lower()}")

    if "Non-advancing" in edge_norm:
        edge_phrase = "and a non-advancing, stalled wound edge suggesting delayed healing"
    else:
        edge_phrase = "and an advancing wound edge indicating active healing"

    query = (
        f"What wound dressing is recommended for {tissue_phrase} "
        f"{infection_phrase}, {moisture_phrase}, {edge_phrase}? "
        f"Include dressing type, contraindications, antibiotic guidance, "
        f"referral criteria, and change frequency per clinical guidelines."
    )

    if notes.strip():
        query += f" Additional clinical context: {notes.strip()[:200]}"

    return query


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL HELPERS  (unchanged from v4_01)
# ══════════════════════════════════════════════════════════════════════════════

def _build_hybrid_retriever(k: int = 10) -> EnsembleRetriever:
    dense_retriever = db.as_retriever(search_kwargs={"k": k})
    _bm25_retriever.k = k
    return EnsembleRetriever(
        retrievers=[dense_retriever, _bm25_retriever],
        weights=[0.6, 0.4],
    )


def _hybrid_search(query: str, k: int) -> list:
    try:
        hybrid  = _build_hybrid_retriever(k=k + 2)
        results = hybrid.invoke(query)
        return results[:k]
    except Exception as e:
        print(f"  [Retrieval] Hybrid search error: {e} — falling back to dense")
        return db.similarity_search(query, k=k)


def _dense_search(query: str, k: int, where: dict = None) -> list:
    try:
        if where:
            docs = db.similarity_search(query, k=k, filter=where)
            if docs:
                return docs
            print(f"  [Retrieval] Metadata filter {where} returned 0 results — falling back to unfiltered")
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"  [Retrieval] Dense search error: {e}")
        return []


def _build_dressing_mechanism_query(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm:  str,
) -> str:
    parts = []
    if moisture_norm == "High exudate":
        parts.append("high absorbency exudate management dressing alginate foam hydrofiber")
    elif moisture_norm == "Dry":
        parts.append("moisture donation rehydration dressing hydrogel film hydrocolloid")
    else:
        parts.append("moderate exudate dressing foam hydrocolloid moisture balance")

    if infection_norm == "Locally infected":
        parts.append("antimicrobial silver iodine infected wound dressing properties indications")

    nv = tissue_profile["non_viable_pct"]
    if nv >= 50:
        parts.append("debridement necrotic slough autolytic enzymatic wound bed preparation")
    elif tissue_profile["granulation_pct"] >= 70:
        parts.append("granulation tissue protection epithelialisation wound healing dressing")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# [v4 R2] MULTI-AXIS SUB-QUERY RETRIEVAL  (unchanged from v4_01)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks_multiaxis(
    narrative_query:  str,
    tissue_profile:   dict,
    infection_norm:   str,
    moisture_norm:    str,
    classifier:       dict,
    notes:            str = "",
    top_n:            int = 6,
) -> tuple[list, list]:
    """
    Multi-axis parallel retrieval with BM25 hybrid.
    Sub-query A: wound-type algorithm chunk (dense-only, k=2)
    Sub-query B: dressing mechanism (hybrid, k=3)
    Sub-query C: patient notes context (hybrid, k=2, only if notes present)
    Deduplication: first occurrence wins (Sub-query A has priority).
    """
    retrieval_notes = []
    seen_ids        = set()
    merged          = []

    # ── Sub-query A: pinned wound-type algorithm chunk (dense only) ────────────
    algo_query = classifier["algorithm_query"]
    wt         = classifier["wound_type"]

    algo_docs = []
    for filter_attempt in [
        {"wound_type": {"$eq": str(wt)}},
        {"wound_type": {"$eq": wt}},
        {"source":     {"$contains": f"Wound Type {wt}"}},
    ]:
        try:
            algo_docs = db.similarity_search(algo_query, k=2, filter=filter_attempt)
            if algo_docs:
                retrieval_notes.append(
                    f"Sub-query A: wound type {wt} chunk pinned via metadata filter {filter_attempt}"
                )
                break
        except Exception:
            continue

    if not algo_docs:
        algo_docs = db.similarity_search(algo_query, k=2)
        retrieval_notes.append(
            f"Sub-query A: wound type {wt} chunk via unfiltered dense (metadata filter unavailable)"
        )

    for doc in algo_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)

    retrieval_notes.append(f"Sub-query A: {len(algo_docs)} algorithm chunks (dense-only)")

    # ── Sub-query B: dressing mechanism (hybrid) ───────────────────────────────
    mech_query = _build_dressing_mechanism_query(tissue_profile, infection_norm, moisture_norm)
    mech_docs  = _hybrid_search(mech_query, k=3)
    added_b = 0
    for doc in mech_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)
            added_b += 1
    retrieval_notes.append(f"Sub-query B: {added_b} dressing mechanism chunks (hybrid dense+BM25)")

    # ── Sub-query C: patient notes (hybrid, only if notes present) ────────────
    if notes.strip():
        notes_query = notes.strip()[:300] + " wound dressing recommendation"
        notes_docs  = _hybrid_search(notes_query, k=2)
        added_c = 0
        for doc in notes_docs:
            doc_id = doc.page_content[:80]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)
                added_c += 1
        retrieval_notes.append(f"Sub-query C: {added_c} notes-context chunks (hybrid dense+BM25)")
    else:
        remaining = top_n - len(merged)
        if remaining > 0:
            fill_docs  = _hybrid_search(narrative_query, k=remaining + 2)
            added_fill = 0
            for doc in fill_docs:
                doc_id = doc.page_content[:80]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)
                    added_fill += 1
                    if len(merged) >= top_n:
                        break
            retrieval_notes.append(f"Sub-query fill: {added_fill} narrative chunks (hybrid)")

    final = merged[:top_n]
    retrieval_notes.append(
        f"v4_02 hybrid multi-axis retrieval: {len(final)} chunks total "
        f"(wound_type={wt}, etiology={classifier.get('etiology','generic')}, "
        f"referral={classifier['referral_required']}, "
        f"abx={classifier['antibiotic_required']})"
    )
    return final, retrieval_notes


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION — SYSTEM PROMPT  (unchanged from v4_01)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Clinical Wound Care Consultant generating evidence-based \
wound dressing recommendations for a general practitioner.

STRICT GROUNDING RULES — you MUST follow these without exception:
1. Every clinical claim, dressing recommendation, and contraindication MUST be \
explicitly supported by one of the numbered Guideline Sources provided in the user \
message. Do NOT use general medical knowledge or training data that is not present \
in the retrieved sources.
2. After each recommendation or rationale point, cite the source number in \
parentheses, e.g. (Source 2) or (Sources 1, 3).
3. If the retrieved sources do not contain information needed to answer a specific \
section, write "Insufficient guideline evidence retrieved for this point" — do NOT \
invent a recommendation.
4. In the Contraindicated Dressings section, begin the list with the exact line: \
"The following dressings are CONTRAINDICATED in this case:" followed by each \
contraindicated item. This line MUST appear verbatim.
5. In the Antibiotic Considerations section, if the clinical picture indicates local \
infection or signs of systemic infection, state explicitly whether antibiotics are \
recommended. Use the exact phrase "Antibiotic therapy is recommended" or \
"Antibiotic therapy is not indicated" as the opening sentence of that section.
6. In the Referral / Escalation section, state explicitly whether referral is required \
using the exact phrase "Referral is recommended" or "Referral is not required at this \
stage" as the opening sentence of that section."""


def _find_algorithm_chunk(chunks: list, wound_type: int) -> tuple:
    """Identify the wound-type algorithm chunk from the retrieved pool."""
    keyword      = f"Wound Type {wound_type}"
    algo_chunk   = None
    other_chunks = []
    for chunk in chunks:
        content = chunk.page_content + chunk.metadata.get("raw_text", "")
        if keyword in content and algo_chunk is None:
            algo_chunk = chunk
        else:
            other_chunks.append(chunk)
    return algo_chunk, other_chunks


def generate_recommendation(
    chunks:          list,
    assessment_text: str,
    narrative_query: str,
    classifier:      dict,
) -> str:
    """
    [v4_02] Grounded generation — G1 (binding block) + G2 (mandatory injections).
    G3 verifier removed. Returns answer string directly (no tuple).
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        wt  = classifier["wound_type"]

        # ── [G1] Separate algorithm chunk from evidence chunks ─────────────────
        algo_chunk, evidence_chunks = _find_algorithm_chunk(chunks, wt)

        evidence_block = ""
        src_offset = 2 if algo_chunk else 1
        for i, chunk in enumerate(evidence_chunks, src_offset):
            source    = chunk.metadata.get("source",    "Unknown")
            authority = chunk.metadata.get("authority", "")
            year      = chunk.metadata.get("year",      "")
            raw_text  = chunk.metadata.get("raw_text", chunk.page_content)
            evidence_block += f"\n--- Guideline Source {i}: {source}"
            if authority:
                evidence_block += f" [{authority}"
                if year:
                    evidence_block += f", {year}"
                evidence_block += "]"
            evidence_block += f" ---\n{raw_text}\n"

        if algo_chunk:
            algo_text = algo_chunk.metadata.get("raw_text", algo_chunk.page_content)
            binding_block = f"""
══════════════════════════════════════════════════════════
BINDING CLINICAL ALGORITHM — Wound Type {wt} (Source 1)
══════════════════════════════════════════════════════════
{algo_text}
══════════════════════════════════════════════════════════
INSTRUCTION: Your PRIMARY DRESSING recommendation MUST be \
selected exclusively from the "Recommended Dressing Materials" list \
in the Binding Clinical Algorithm above (Source 1). \
Do NOT recommend any dressing category that is not in that allowed list.
══════════════════════════════════════════════════════════
"""
        else:
            binding_block = (
                f"\n[Note: Wound Type {wt} algorithm chunk not retrieved. "
                f"Apply strict grounding from available sources only.]\n"
            )
            print(f"  [v4 G1] WARNING: Algorithm chunk for Wound Type {wt} not found")

        # ── [G2] Conditional mandatory injections ─────────────────────────────
        mandatory_injections = ""

        if classifier["referral_required"]:
            reason = classifier.get("escalation_reason", "")
            mandatory_injections += (
                f"\n⚠ MANDATORY CLINICAL INSTRUCTION — REFERRAL:\n"
                f"This wound (Wound Type {wt}) REQUIRES hospital/specialist referral "
                f"per the binding clinical algorithm. "
                f"{'Escalation reason: ' + reason if reason else ''}\n"
                f"The Referral / Escalation section MUST open with the exact phrase: "
                f"\"Referral is recommended\" — this is non-negotiable.\n"
            )

        if classifier["antibiotic_required"]:
            abx_context = ""
            if classifier.get("subclinical_infection"):
                # [FIX 1] Subclinical infection path — provide clinical reasoning
                abx_context = (
                    "\n⚠ CLINICAL ALERT — SUBCLINICAL INFECTION RISK:\n"
                    "The patient's clinical notes contain multiple local infection "
                    "signals (increased pain, warmth, redness, or other T.I.M.E. "
                    "'I' signs) despite the structured input indicating 'Not infected'. "
                    "This discrepancy suggests developing local infection that has not "
                    "yet been formally assessed. You MUST:\n"
                    "  1. Address this discrepancy explicitly in the Antibiotic "
                    "Considerations section.\n"
                    "  2. Recommend an antimicrobial dressing as a clinical precaution.\n"
                    "  3. Advise the GP to reassess infection status at next visit.\n"
                )
            mandatory_injections += abx_context + (
                f"\n⚠ MANDATORY CLINICAL INSTRUCTION — ANTIBIOTIC:\n"
                f"Antibiotic guidance MUST be addressed for this wound.\n"
                f"The Antibiotic Considerations section MUST open with the exact phrase: "
                f"\"Antibiotic therapy is recommended\" — this is non-negotiable.\n"
            )

        # ── [FIX 2] Etiology note for diabetic foot ────────────────────────────
        # Injected as a soft note (not a binding override) when etiology is
        # diabetic_foot. Informs the LLM about special dressing considerations
        # without forcing a specific dressing outside the algorithm allowed list.
        etiology_note = ""
        if classifier.get("etiology") == "diabetic_foot":
            etiology_note = (
                "\n📋 ETIOLOGY NOTE — DIABETIC FOOT:\n"
                "The patient is diabetic. Per AJGP diabetic foot wound guidelines:\n"
                "  - Adhesive bordered foam dressings are CONTRAINDICATED on feet "
                "(risk of pressure and skin damage).\n"
                "  - Hydrocolloid dressings are not recommended for diabetic foot ulcers.\n"
                "  - Prefer non-adhesive dressings anchored with tape or bandages.\n"
                "  - If an antimicrobial dressing is indicated, silver dressings are "
                "the first-line antimicrobial choice for diabetic foot wounds.\n"
                "Incorporate these contraindications into your Contraindicated Dressings section.\n"
            )
        elif classifier.get("etiology") == "burn":
            etiology_note = (
                "\n📋 ETIOLOGY NOTE — BURN WOUND:\n"
                "This is a burn wound. Per burn wound management guidelines:\n"
                "  - For small superficial burns: hydrogel, hydrocolloid, or film "
                "dressings are recommended after initial first aid.\n"
                "  - Foam dressings alone are NOT the first-line choice for burn wounds.\n"
                "  - Paraffin tulle or silicone non-adherent dressings are appropriate "
                "for superficial partial-thickness burns.\n"
                "Incorporate these recommendations into your Primary Dressing section "
                "if supported by the retrieved sources.\n"
            )
        elif classifier.get("etiology") == "skin_tear":
            etiology_note = (
                "\n📋 ETIOLOGY NOTE — SKIN TEAR / FRAGILE SKIN:\n"
                "This is a skin tear or fragile skin wound. Per skin tear guidelines:\n"
                "  - Silicone-covered foam dressing is the recommended primary dressing "
                "(non-adhesive, minimises trauma on removal).\n"
                "  - Adhesive products (bordered foam, adhesive film) are CONTRAINDICATED "
                "on fragile elderly skin — they cause further tears on removal.\n"
                "  - Use remover wipes and remove dressings in the direction of skin flap.\n"
                "Incorporate these into your Primary Dressing and Contraindicated sections.\n"
            )

        # ── Build full human prompt ────────────────────────────────────────────
        human_prompt = f"""CLINICAL QUESTION:
{narrative_query}

{assessment_text}
{mandatory_injections}
{etiology_note}
{binding_block}
ADDITIONAL RETRIEVED CLINICAL GUIDELINES (evidence for all other sections):
{evidence_block}

Provide your recommendation using EXACTLY the following section structure. \
Do not add, rename, or omit any section. Cite source numbers after every claim.

## Primary Dressing
- Name the dressing category and one specific product/brand example cited in the \
guidelines (Source X). The dressing MUST be from the allowed list in Source 1 \
(or the etiology note above if a more specific recommendation applies).
- Explain why this dressing addresses the dominant clinical need based on the \
T.I.M.E. assessment.

## Secondary Dressing
- If a secondary dressing is indicated according to the guidelines, name it and \
explain its role (Source X). If a single dressing suffices, state "No secondary \
dressing is required" and explain why.

## Rationale by T.I.M.E. Factor
- T (Tissue): Link the tissue composition to the dressing mechanism (Source X).
- I (Infection): State the infection status and justify any antimicrobial dressing \
choice (Source X).
- M (Moisture): Explain the exudate level and required dressing absorbency or \
moisture donation (Source X).
- E (Edge): Explain how the wound edge status influences the dressing choice (Source X).

## Contraindicated Dressings
The following dressings are CONTRAINDICATED in this case:
- List each contraindicated dressing type with a brief reason from the guidelines \
(Source X) and/or the etiology note above.
- If no specific contraindications are found in the retrieved sources, write: \
"No specific contraindications identified in retrieved sources."

## Antibiotic Considerations
Begin this section with EXACTLY one of these two sentences (choose based on evidence):
  "Antibiotic therapy is recommended" — if sources support systemic or topical \
antibiotic use for this wound.
  "Antibiotic therapy is not indicated" — if sources indicate no antibiotic is needed.
Then explain the reasoning from the retrieved guidelines (Source X).

## Referral / Escalation
Begin this section with EXACTLY one of these two sentences (choose based on evidence):
  "Referral is recommended" — if sources indicate this wound type requires \
specialist or hospital referral.
  "Referral is not required at this stage" — if sources indicate this can be \
managed in primary care.
Then explain the reasoning from the retrieved guidelines (Source X).

## Dressing Change Frequency
- State the recommended change frequency with guideline citation (Source X).

## Application Tips
- List 2–3 practical application tips from the retrieved guidelines (Source X).

## Clinical Notes
- Summarise any patient-specific considerations from the Additional Clinical Notes \
field above. If no notes were provided, write "No additional clinical notes provided."
"""

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        return response.content

    except Exception as e:
        return f"Clinical analysis error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="wound_index_v4.html",
        context={"request": request},
    )


@app.post("/get_recommendation")
async def get_recommendation(
    necrotic_pct:      float = Form(...),
    slough_pct:        float = Form(...),
    granulation_pct:   float = Form(...),
    infection:         str   = Form(...),
    moisture:          str   = Form(...),
    edge:              str   = Form(...),
    notes:             str   = Form(""),
    tissue_confidence: float = Form(0.0),
):
    try:
        # ── Step 1: Normalise inputs ───────────────────────────────────────────
        tissue_profile = interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct)
        infection_norm = normalize_infection(infection)
        moisture_norm  = normalize_moisture(moisture)
        edge_norm      = normalize_edge(edge)

        print(f"\n[INPUT] N={necrotic_pct}% S={slough_pct}% G={granulation_pct}%")
        print(f"[INPUT] Infection : {infection_norm}")
        print(f"[INPUT] Moisture  : {moisture_norm}")
        print(f"[INPUT] Edge      : {edge_norm}")
        print(f"[INPUT] Notes     : {notes[:80]}{'...' if len(notes) > 80 else ''}")

        # ── Step 2: [v4_02] Clinical pre-classifier (with fixes) ──────────────
        classifier = classify_wound(tissue_profile, infection_norm, moisture_norm, notes)
        print(f"[CLASSIFIER] {classifier['classifier_notes']}")

        # ── Step 3: Build narrative query ──────────────────────────────────────
        narrative_query = build_narrative_query(
            tissue_profile,
            infection_norm,
            moisture_norm,
            edge_norm,
            notes,
        )
        print(f"[RETRIEVAL] Narrative query: {narrative_query[:120]}...")

        # ── Step 4: Multi-axis hybrid retrieval ───────────────────────────────
        top_chunks, retrieval_notes = retrieve_chunks_multiaxis(
            narrative_query = narrative_query,
            tissue_profile  = tissue_profile,
            infection_norm  = infection_norm,
            moisture_norm   = moisture_norm,
            classifier      = classifier,
            notes           = notes,
            top_n           = 6,
        )
        print(f"[RETRIEVAL] {len(top_chunks)} chunks retrieved")
        for i, chunk in enumerate(top_chunks, 1):
            src = chunk.metadata.get("source", "Unknown")
            print(f"  [Chunk {i}] {src} | {chunk.page_content[:80]}...")

        # ── Step 5: Build structured assessment text ───────────────────────────
        assessment_text = f"""T.I.M.E. WOUND ASSESSMENT:

T (Tissue)    : {tissue_profile['clinical_tissue']}
                Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
                Non-viable load: {tissue_profile['non_viable_pct']}%

I (Infection) : {infection_norm}
M (Moisture)  : {moisture_norm}
E (Edge)      : {edge_norm}"""

        if notes.strip():
            assessment_text += f"\n\nADDITIONAL CLINICAL NOTES:\n{notes.strip()}"

        # ── Step 6: [v4_02] Grounded generation (no verifier) ─────────────────
        result = generate_recommendation(
            chunks          = top_chunks,
            assessment_text = assessment_text,
            narrative_query = narrative_query,
            classifier      = classifier,
        )

        # ── Step 7: Build response ─────────────────────────────────────────────
        sources     = list(dict.fromkeys(c.metadata.get("source", "Unknown") for c in top_chunks))
        chunk_texts = [c.page_content for c in top_chunks]

        return JSONResponse({
            "result":            result,
            "sources":           sources,
            "chunk_texts":       chunk_texts,
            "confidence_score":  0.5,
            "confidence_label":  "MEDIUM",
            "retrieval_notes":   retrieval_notes,
            "tissue_breakdown":  {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores":   [],
            "clinical_flags":    {
                "infection_norm":        infection_norm,
                "moisture_norm":         moisture_norm,
                "edge_norm":             edge_norm,
                "subclinical_infection": classifier.get("subclinical_infection", False),
                "etiology":              classifier.get("etiology", "generic"),
            },
            "narrative_query":   narrative_query,
            "classifier_output": classifier,
            "verifier_output":   {
                "ran":    False,
                "reason": "Verifier removed in v4_02 — G2 mandatory injections handle safety-critical phrases deterministically",
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "result":            f"System error: {str(e)}",
                "sources":           [],
                "chunk_texts":       [],
                "confidence_label":  "LOW",
                "confidence_score":  0.0,
                "retrieval_notes":   ["System error during retrieval"],
                "tissue_breakdown":  {},
                "reranker_scores":   [],
                "clinical_flags":    {},
                "narrative_query":   "",
                "classifier_output": {},
                "verifier_output":   {"ran": False, "reason": "error"},
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
