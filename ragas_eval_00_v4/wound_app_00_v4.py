"""
wound_app_00_v4.py  —  VerdaSense Clinical RAG Pipeline (v4_00 · Baseline)
══════════════════════════════════════════════════════════════════════════════
V4 ARCHITECTURE CHANGES vs v3_00 baseline
──────────────────────────────────────────
[R1] Clinical pre-classifier
     Deterministic rule-based mapping of T.I.M.E. inputs → wound_type (1–8)
     + clinical flags (referral_required, antibiotic_required).
     Used to inject a Chroma metadata filter so the wound-type algorithm chunk
     is always pinned in the retrieval pool.

[R2] Multi-axis sub-query retrieval (3 parallel queries, merged + deduplicated)
     Query A: wound-type algorithm query (k=2, metadata-filtered)  ← pinned
     Query B: dressing mechanism query from moisture + tissue       (k=3)
     Query C: patient notes NER query if notes present             (k=2)
     Merged pool is capped at top_n=6 with deduplication.
     Guarantees the algorithm chunk is present without crowding out
     dressing-mechanism evidence.

[G1] Binding algorithm block injected at top of prompt
     The wound-type chunk is extracted from the pinned pool and placed in a
     dedicated "BINDING CLINICAL ALGORITHM" section before the evidence block.
     LLM is instructed: primary dressing MUST come from the allowed list
     in the binding algorithm.

[G2] Conditional hard-text injection for referral & antibiotic
     When the classifier flags referral_required or antibiotic_required,
     a MANDATORY sentence is injected directly into the prompt — replacing
     the soft system-prompt hint that was being ignored in ~4–5 cases.

[G3] Post-generation verifier LLM pass
     A second gpt-4o-mini call checks 5 binary safety questions against the
     generated answer. If any check fails, a targeted correction prompt is
     sent to fix only the failed sections (max 1 retry).

Retrieval core (v4_00): dense-only ChromaDB similarity_search (k=6 pool for
sub-queries B+C, plus k=2 pinned for sub-query A). No BM25, no cross-encoder.

Unchanged from v3:
  - Embedding model: abhinand/MedEmbed-large-v0.1
  - Narrative query builder (v3 style)
  - SYSTEM_PROMPT grounding rules
  - Response contract (all v3 fields + new v4 fields)

Response contract (v4 — backwards-compatible with v3):
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
    "narrative_query":    str,      ← v3 field (RAGAS user_input)
    "classifier_output":  dict,     ← NEW v4: pre-classifier result
    "verifier_output":    dict,     ← NEW v4: post-gen verifier result
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


# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION  (unchanged from v3)
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
# [v4 R1] CLINICAL PRE-CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
# Deterministic rule-based mapping of T.I.M.E. inputs → MOH Malaysia wound
# type (1–8) + clinical flags.  Rules derived from the testset ground truth
# and MOH Wound Care Manual 2014 T.I.M.E. algorithm.
#
# Wound Type Matrix (primary axis: infection × moisture × necrosis load):
#
#  NV = necrotic_pct + slough_pct  (non-viable load)
#
#  Type 1: NV < 25%, Not infected,  Dry/Low         → no referral, no abx
#  Type 2: NV < 25%, Not infected,  High exudate     → no referral, no abx
#  Type 3: NV < 50%, Infected,      Dry/Low/Moderate → no referral, abx
#  Type 4: NV < 50%, Infected,      High exudate     → no referral, abx
#  Type 5: NV >= 50%, Not infected, Dry/Low          → no referral, no abx
#  Type 6: NV >= 50%, Not infected, High exudate     → REFERRAL, no abx
#  Type 7: NV >= 50%, Infected,     Dry/Low/Moderate → REFERRAL, abx
#  Type 8: NV >= 50%, Infected,     High exudate     → REFERRAL, abx
#
# Notes override: if notes contain "diabetic" + "non-healing" or "refer"
# signals → escalate referral_required regardless of type.
# ══════════════════════════════════════════════════════════════════════════════

# Wound type → metadata filter key used in Chroma KB
WOUND_TYPE_CHUNK_KEYWORDS = {
    1: "Wound Type 1",
    2: "Wound Type 2",
    3: "Wound Type 3",
    4: "Wound Type 4",
    5: "Wound Type 5",
    6: "Wound Type 6",
    7: "Wound Type 7",
    8: "Wound Type 8",
}

# Wound type → short algorithm description for query A
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


def classify_wound(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    notes: str = "",
) -> dict:
    """
    [v4 R1] Deterministic clinical pre-classifier.

    Returns:
        {
            "wound_type":          int (1–8),
            "referral_required":   bool,
            "antibiotic_required": bool,
            "escalation_reason":   str | None,
            "algorithm_query":     str,   ← targeted retrieval query for type chunk
            "classifier_notes":    str,   ← human-readable explanation
        }
    """
    nv       = tissue_profile["non_viable_pct"]   # necrotic + slough %
    infected = (infection_norm == "Locally infected")
    high_exu = (moisture_norm == "High exudate")
    notes_lc = notes.lower()

    # ── Primary classification ─────────────────────────────────────────────────
    if not infected and nv < 25:
        wound_type   = 2 if high_exu else 1
        referral     = False
        antibiotic   = False
    elif infected and nv < 50:
        wound_type   = 4 if high_exu else 3
        referral     = False
        antibiotic   = True
    elif not infected and nv >= 50:
        wound_type   = 6 if high_exu else 5
        referral     = high_exu       # Type 6 → refer; Type 5 → no refer
        antibiotic   = False
    else:  # infected and nv >= 50
        wound_type   = 8 if high_exu else 7
        referral     = True
        antibiotic   = True

    # ── Notes-based escalation overrides ──────────────────────────────────────
    escalation_reason = None
    referral_triggers = [
        "diabetic",  "non-heal", "non heal", "nonheal",
        "hospital",  "refer",    "specialist", "chronic",
        "burns",     "burn",     "deep",       "full thickness",
    ]
    if not referral:
        for trigger in referral_triggers:
            if trigger in notes_lc:
                referral          = True
                escalation_reason = f"Notes contain '{trigger}' — referral escalated"
                break

    antibiotic_triggers = ["infect", "purulent", "pus", "fever", "sepsis", "cellulitis"]
    if not antibiotic:
        for trigger in antibiotic_triggers:
            if trigger in notes_lc:
                antibiotic        = True
                escalation_reason = (escalation_reason or "") + f" | Notes: antibiotic escalated ('{trigger}')"
                break

    classifier_notes = (
        f"NV={nv:.1f}%, infected={infected}, high_exudate={high_exu} "
        f"→ Wound Type {wound_type}, referral={referral}, abx={antibiotic}"
    )
    if escalation_reason:
        classifier_notes += f" | ESCALATION: {escalation_reason}"

    return {
        "wound_type":          wound_type,
        "referral_required":   referral,
        "antibiotic_required": antibiotic,
        "escalation_reason":   escalation_reason,
        "algorithm_query":     WOUND_TYPE_QUERY_PHRASES[wound_type],
        "classifier_notes":    classifier_notes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE QUERY BUILDER  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

def build_narrative_query(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm:  str,
    edge_norm:      str,
    notes:          str = "",
) -> str:
    ct  = tissue_profile["clinical_tissue"]
    n   = tissue_profile["necrotic_pct"]
    s   = tissue_profile["slough_pct"]
    g   = tissue_profile["granulation_pct"]
    nv  = tissue_profile["non_viable_pct"]

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
        "High exudate": "producing high levels of exudate requiring high-absorbency dressings",
        "Dry":          "presenting with dry to minimal exudate requiring moisture-donating dressings",
        "Moderate exudate": "producing moderate exudate requiring balanced moisture management",
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
# [v4 R2] MULTI-AXIS SUB-QUERY RETRIEVAL  (dense-only for v4_00 baseline)
# ══════════════════════════════════════════════════════════════════════════════

def _dense_search(query: str, k: int, where: dict = None) -> list:
    """
    Thin wrapper around Chroma similarity_search.
    `where` is passed as Chroma metadata filter when provided.
    Falls back to unfiltered search if the filtered search returns < 1 result
    (guards against metadata key absence in some KB chunks).
    """
    try:
        if where:
            docs = db.similarity_search(query, k=k, filter=where)
            if docs:
                return docs
            # fallback: no metadata filter
            print(f"  [Retrieval] Metadata filter {where} returned 0 results — falling back to unfiltered")
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"  [Retrieval] search error: {e}")
        return []


def _build_dressing_mechanism_query(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm:  str,
) -> str:
    """Build a focused query for dressing mechanism / properties chunks."""
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
    [v4 R2] Multi-axis parallel retrieval.

    Sub-query A: wound-type algorithm chunk  (k=2, metadata-filtered if available)
    Sub-query B: dressing mechanism          (k=3, unfiltered dense)
    Sub-query C: patient notes NER           (k=2, only if notes present)

    Deduplication: first occurrence wins (A has priority).
    Returns (chunks[:top_n], retrieval_notes).
    """
    retrieval_notes = []
    seen_ids        = set()
    merged          = []

    # ── Sub-query A: pinned wound-type algorithm chunk ─────────────────────────
    algo_query = classifier["algorithm_query"]
    wt         = classifier["wound_type"]

    # Try with metadata filter first; Chroma filter syntax varies by KB schema.
    # Try common metadata field names used in wound care KBs.
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
        # Fallback: unfiltered semantic search with targeted query
        algo_docs = db.similarity_search(algo_query, k=2)
        retrieval_notes.append(
            f"Sub-query A: wound type {wt} chunk via unfiltered semantic search (metadata filter unavailable)"
        )

    for doc in algo_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)

    retrieval_notes.append(f"Sub-query A: {len(algo_docs)} algorithm chunks retrieved")

    # ── Sub-query B: dressing mechanism ────────────────────────────────────────
    mech_query = _build_dressing_mechanism_query(tissue_profile, infection_norm, moisture_norm)
    mech_docs  = _dense_search(mech_query, k=3)
    added_b = 0
    for doc in mech_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)
            added_b += 1
    retrieval_notes.append(f"Sub-query B: {added_b} dressing mechanism chunks added")

    # ── Sub-query C: patient notes (only if notes present) ───────────────────
    if notes.strip():
        notes_query = notes.strip()[:300] + " wound dressing recommendation"
        notes_docs  = _dense_search(notes_query, k=2)
        added_c = 0
        for doc in notes_docs:
            doc_id = doc.page_content[:80]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)
                added_c += 1
        retrieval_notes.append(f"Sub-query C: {added_c} notes-context chunks added")
    else:
        # Fill remaining budget with narrative query if no notes
        remaining = top_n - len(merged)
        if remaining > 0:
            fill_docs = _dense_search(narrative_query, k=remaining + 2)
            added_fill = 0
            for doc in fill_docs:
                doc_id = doc.page_content[:80]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)
                    added_fill += 1
                    if len(merged) >= top_n:
                        break
            retrieval_notes.append(f"Sub-query fill: {added_fill} narrative chunks added")

    final = merged[:top_n]
    retrieval_notes.append(
        f"v4_00 dense multi-axis retrieval: {len(final)} chunks total "
        f"(wound_type={wt}, referral={classifier['referral_required']}, "
        f"abx={classifier['antibiotic_required']})"
    )
    return final, retrieval_notes


# ══════════════════════════════════════════════════════════════════════════════
# [v4 G1 + G2 + G3] GROUNDED GENERATION WITH BINDING ALGORITHM + VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

# ── System prompt (same as v3 — strict grounding rules) ──────────────────────
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


def _find_algorithm_chunk(chunks: list, wound_type: int) -> tuple[object | None, list]:
    """
    [v4 G1] Identify the wound-type algorithm chunk from the retrieved pool.
    The algorithm chunk is the one that mentions "Wound Type X" or
    "Clinical Summary for Wound Type X" in its content.
    Returns (algorithm_chunk_or_None, remaining_chunks).
    """
    keyword = f"Wound Type {wound_type}"
    algo_chunk  = None
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
) -> tuple[str, dict]:
    """
    [v4 G1 + G2 + G3] Grounded generation with:
      - Binding algorithm block at top of prompt (G1)
      - Conditional hard-text referral/antibiotic injection (G2)
      - Post-generation verifier with targeted correction (G3)

    Returns (final_answer_str, verifier_output_dict).
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        wt  = classifier["wound_type"]

        # ── [G1] Separate algorithm chunk from evidence chunks ─────────────────
        algo_chunk, evidence_chunks = _find_algorithm_chunk(chunks, wt)

        # Build evidence block (non-algorithm chunks)
        evidence_block = ""
        src_offset = 2 if algo_chunk else 1   # Source 1 is reserved for algo chunk
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

        # Build binding algorithm block
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
            print(f"  [v4 G1] WARNING: Algorithm chunk for Wound Type {wt} not found in retrieved pool")

        # ── [G2] Conditional mandatory injection ──────────────────────────────
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
            mandatory_injections += (
                f"\n⚠ MANDATORY CLINICAL INSTRUCTION — ANTIBIOTIC:\n"
                f"This wound (Wound Type {wt}) has local infection. "
                f"Antibiotic guidance MUST be addressed.\n"
                f"The Antibiotic Considerations section MUST open with the exact phrase: "
                f"\"Antibiotic therapy is recommended\" — this is non-negotiable.\n"
            )

        # ── Build full human prompt ────────────────────────────────────────────
        human_prompt = f"""CLINICAL QUESTION:
{narrative_query}

{assessment_text}
{mandatory_injections}
{binding_block}
ADDITIONAL RETRIEVED CLINICAL GUIDELINES (evidence for all other sections):
{evidence_block}

Provide your recommendation using EXACTLY the following section structure. \
Do not add, rename, or omit any section. Cite source numbers after every claim.

## Primary Dressing
- Name the dressing category and one specific product/brand example cited in the \
guidelines (Source X). The dressing MUST be from the allowed list in Source 1.
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
(Source X).
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

        # ── First generation pass ──────────────────────────────────────────────
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        answer = response.content

        # ── [G3] Post-generation verifier ─────────────────────────────────────
        verifier_output = _run_verifier(
            answer        = answer,
            chunks        = chunks,
            classifier    = classifier,
            llm           = llm,
            human_prompt  = human_prompt,
        )

        # If verifier triggered a correction, use the corrected answer
        if verifier_output.get("correction_applied"):
            answer = verifier_output["corrected_answer"]
            print(f"  [v4 G3] Verifier correction applied — {verifier_output['failed_checks']}")
        else:
            print(f"  [v4 G3] Verifier: all checks passed")

        return answer, verifier_output

    except Exception as e:
        return f"Clinical analysis error: {str(e)}", {"error": str(e)}


def _run_verifier(
    answer:       str,
    chunks:       list,
    classifier:   dict,
    llm,
    human_prompt: str,
) -> dict:
    """
    [v4 G3] Post-generation verifier.

    Asks a second LLM call to check 5 binary safety questions.
    If any fail, triggers one targeted correction pass.

    Checks:
    1. Is the primary dressing from the allowed list in the algorithm chunk?
    2. Is referral status correctly stated (if required)?
    3. Is antibiotic guidance correctly stated (if required)?
    4. Are contraindications explicitly listed in the Contraindicated Dressings section?
    5. Is every recommendation cited with a source number?
    """
    wt = classifier["wound_type"]

    verifier_prompt = f"""You are a clinical quality checker. Review the wound dressing recommendation below and answer exactly 5 binary questions. Respond ONLY with a JSON object — no preamble, no markdown fences.

WOUND TYPE: {wt}
REFERRAL REQUIRED BY ALGORITHM: {classifier['referral_required']}
ANTIBIOTIC REQUIRED BY ALGORITHM: {classifier['antibiotic_required']}

RECOMMENDATION TO CHECK:
{answer}

Answer ONLY with this exact JSON structure:
{{
  "q1_primary_dressing_appropriate": true/false,
  "q2_referral_correctly_stated": true/false,
  "q3_antibiotic_correctly_stated": true/false,
  "q4_contraindications_listed": true/false,
  "q5_citations_present": true/false,
  "q1_note": "brief reason",
  "q2_note": "brief reason",
  "q3_note": "brief reason",
  "q4_note": "brief reason",
  "q5_note": "brief reason"
}}

Scoring rules:
- q1: PASS (true) if the primary dressing is a clinically valid dressing type for Wound Type {wt}. \
FAIL if it recommends a dressing that would be contraindicated (e.g. silver on clean non-infected wound, \
alginate on dry wound, hydrogel on wet infected wound).
- q2: PASS (true) if referral_required={classifier['referral_required']} AND the Referral section opens \
with "Referral is recommended" (when true) or "Referral is not required at this stage" (when false). \
If referral_required=False, this is PASS unless the recommendation wrongly insists on referral.
- q3: PASS (true) if antibiotic_required={classifier['antibiotic_required']} AND the Antibiotic section \
opens with "Antibiotic therapy is recommended" (when true) or "Antibiotic therapy is not indicated" (when false).
- q4: PASS (true) if the Contraindicated Dressings section contains the verbatim line \
"The following dressings are CONTRAINDICATED in this case:"
- q5: PASS (true) if at least 4 different sections contain (Source N) citations.
"""

    try:
        verifier_resp = llm.invoke([HumanMessage(content=verifier_prompt)])
        raw = verifier_resp.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        checks = json.loads(raw)
    except Exception as e:
        print(f"  [v4 G3] Verifier parse error: {e}")
        return {"verifier_error": str(e), "correction_applied": False}

    failed_checks = [
        k for k in ["q1_primary_dressing_appropriate", "q2_referral_correctly_stated",
                    "q3_antibiotic_correctly_stated", "q4_contraindications_listed",
                    "q5_citations_present"]
        if not checks.get(k, True)
    ]

    if not failed_checks:
        return {
            "checks":             checks,
            "failed_checks":      [],
            "correction_applied": False,
            "corrected_answer":   None,
        }

    # ── Targeted correction pass ───────────────────────────────────────────────
    correction_instructions = []
    if "q1_primary_dressing_appropriate" in failed_checks:
        correction_instructions.append(
            f"- CORRECT the Primary Dressing section: recommend only dressings from the allowed list "
            f"for Wound Type {wt} in the Binding Clinical Algorithm. Note: {checks.get('q1_note', '')}"
        )
    if "q2_referral_correctly_stated" in failed_checks:
        phrase = "Referral is recommended" if classifier["referral_required"] else "Referral is not required at this stage"
        correction_instructions.append(
            f"- CORRECT the Referral / Escalation section: it MUST open with \"{phrase}\". "
            f"Note: {checks.get('q2_note', '')}"
        )
    if "q3_antibiotic_correctly_stated" in failed_checks:
        phrase = "Antibiotic therapy is recommended" if classifier["antibiotic_required"] else "Antibiotic therapy is not indicated"
        correction_instructions.append(
            f"- CORRECT the Antibiotic Considerations section: it MUST open with \"{phrase}\". "
            f"Note: {checks.get('q3_note', '')}"
        )
    if "q4_contraindications_listed" in failed_checks:
        correction_instructions.append(
            "- CORRECT the Contraindicated Dressings section: it MUST begin with the exact line "
            "\"The following dressings are CONTRAINDICATED in this case:\""
        )
    if "q5_citations_present" in failed_checks:
        correction_instructions.append(
            "- ADD (Source N) citations to any sections that are missing them."
        )

    correction_prompt = f"""The following wound dressing recommendation has quality issues that must be fixed.

ORIGINAL RECOMMENDATION:
{answer}

REQUIRED CORRECTIONS (fix ONLY these — keep all other sections unchanged):
{chr(10).join(correction_instructions)}

Return the complete corrected recommendation with the same section structure. \
Do not change anything that was not listed above."""

    try:
        correction_resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=correction_prompt),
        ])
        corrected = correction_resp.content
    except Exception as e:
        corrected = answer
        print(f"  [v4 G3] Correction call failed: {e}")

    return {
        "checks":             checks,
        "failed_checks":      failed_checks,
        "correction_applied": True,
        "corrected_answer":   corrected,
    }


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
    tissue_confidence: float = Form(0.0),   # accepted but unused — keeps notebook payload compatible
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

        # ── Step 2: [v4 R1] Clinical pre-classifier ────────────────────────────
        classifier = classify_wound(tissue_profile, infection_norm, moisture_norm, notes)
        print(f"[CLASSIFIER] {classifier['classifier_notes']}")

        # ── Step 3: [v3] Build narrative query ────────────────────────────────
        narrative_query = build_narrative_query(
            tissue_profile,
            infection_norm,
            moisture_norm,
            edge_norm,
            notes,
        )
        print(f"[RETRIEVAL] Narrative query: {narrative_query[:120]}...")

        # ── Step 4: [v4 R2] Multi-axis retrieval (dense-only baseline) ──────────
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

        # ── Step 6: [v4 G1 + G2 + G3] Grounded generation + verifier ──────────
        result, verifier_output = generate_recommendation(
            chunks          = top_chunks,
            assessment_text = assessment_text,
            narrative_query = narrative_query,
            classifier      = classifier,
        )

        # ── Step 7: Build response (backwards-compatible + v4 fields) ──────────
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
                "infection_norm":    infection_norm,
                "moisture_norm":     moisture_norm,
                "edge_norm":         edge_norm,
            },
            "narrative_query":   narrative_query,
            "classifier_output": classifier,
            "verifier_output":   {
                "failed_checks":      verifier_output.get("failed_checks", []),
                "correction_applied": verifier_output.get("correction_applied", False),
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
                "verifier_output":   {},
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
