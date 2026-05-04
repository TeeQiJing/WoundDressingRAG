"""
wound_app_v4.py  —  VerdaSense Clinical RAG Pipeline (v4)
══════════════════════════════════════════════════════════
Changes from v3:
  - File renamed to v4 for consistency with the image pipeline changes
  - No changes to retrieval or generation logic (all v3 improvements preserved)
  - /get_recommendation endpoint now also accepts tissue_confidence from the
    GMM pipeline as an optional informational field (logged but not used in
    the recommendation — tissue percentages already encode the information)

All v3 features retained:
  [FIX 1] ClinicalSignalExtractor  (pre-retrieval, pre-prompt)
  [FIX 2] Multi-axis query generation (Q_T, Q_I, Q_M, Q_E, Q_COMBO)
  [FIX 3] Conflict-aware generation prompt
  [FIX 4] Clinically-weighted confidence scoring
"""

import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LC_Doc
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import torch

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

# ── Embedding model — must match ingestion ────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Cross-encoder reranker ────────────────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Load v2 vector store ──────────────────────────────────────────────────────
def load_wound_db(persist_directory: str = "./db_wound_care_v2"):
    print(f"Loading Wound Care KB from {persist_directory}...")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

db = load_wound_db()


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL CONTEXT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClinicalContext:
    tissue_profile:  dict = field(default_factory=dict)
    infection_norm:  str  = ""
    moisture_norm:   str  = ""
    edge_norm:       str  = ""
    notes:           str  = ""

    is_infected:          bool  = False
    is_dry:               bool  = False
    is_high_exudate:      bool  = False
    needs_combination:    bool  = False
    notes_override:       bool  = False
    notes_infection_flags: list = field(default_factory=list)
    escalation_needed:    bool  = False
    high_nonviable:       bool  = False
    debridement_needed:   bool  = False
    combination_rationale: str  = ""

    primary_strategy:     str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def interpret_tissue_percentages(
    necrotic_pct: float,
    slough_pct: float,
    granulation_pct: float,
) -> dict:
    total = necrotic_pct + slough_pct + granulation_pct
    if total == 0:
        return {
            "tissue_label":    "insufficient tissue data",
            "clinical_tissue": "unknown",
            "dominant_tissue": "unknown",
            "necrotic_pct":    0.0,
            "slough_pct":      0.0,
            "granulation_pct": 0.0,
            "non_viable_pct":  0.0,
            "high_non_viable": False,
            "healing_progress": 0.0,
        }

    n = necrotic_pct    / total * 100
    s = slough_pct      / total * 100
    g = granulation_pct / total * 100

    dominant = max(
        [("necrotic", n), ("slough", s), ("granulation", g)],
        key=lambda x: x[1]
    )[0]
    non_viable_pct  = n + s
    high_non_viable = non_viable_pct > 25

    parts = []
    if g >= 70:
        parts.append(f"predominantly granulating ({g:.0f}%)")
    elif g >= 40:
        parts.append(f"mixed tissue with significant granulation ({g:.0f}%)")
    if n >= 25:
        parts.append(f"high necrotic load ({n:.0f}%)")
    elif n > 0:
        parts.append(f"some necrotic tissue ({n:.0f}%)")
    if s >= 25:
        parts.append(f"significant slough ({s:.0f}%)")
    elif s > 0:
        parts.append(f"some slough ({s:.0f}%)")
    tissue_label = ", ".join(parts) if parts else "mixed wound bed tissue"

    if n >= 25 and s >= 25:
        clinical_tissue = "mixed necrotic and slough tissue with minimal granulation"
    elif n >= 50:
        clinical_tissue = "predominantly necrotic wound, eschar present"
    elif n >= 25:
        clinical_tissue = "significant necrotic tissue with some granulation"
    elif s >= 50:
        clinical_tissue = "sloughy fibrinous wound bed, autolytic debridement needed"
    elif s >= 25 and g >= 40:
        clinical_tissue = "mixed slough and granulation tissue"
    elif g >= 70:
        clinical_tissue = "healthy granulating wound bed, proliferative phase"
    else:
        clinical_tissue = "mixed wound bed tissue"

    return {
        "tissue_label":    tissue_label,
        "clinical_tissue": clinical_tissue,
        "dominant_tissue": dominant,
        "necrotic_pct":    round(n, 1),
        "slough_pct":      round(s, 1),
        "granulation_pct": round(g, 1),
        "non_viable_pct":  round(non_viable_pct, 1),
        "high_non_viable": high_non_viable,
        "healing_progress": round(g / 100, 2),
    }


def normalize_infection(label: str) -> str:
    m = label.lower().strip()
    if "not" in m or "no" in m:
        return "Not infected (no clinical signs)"
    return "Locally infected (erythema, warmth, swelling at wound edge)"


def normalize_moisture(label: str) -> str:
    m = label.lower().strip()
    if m in ("high", "high exudate"):
        return "High exudate (copious drainage, maceration risk)"
    elif m in ("low", "low exudate", "dry"):
        return "Dry (no exudate, desiccated wound bed)"
    else:
        return "Moderate exudate (frequent dressing changes needed)"


def normalize_edge(label: str) -> str:
    m = label.lower().strip()
    if "non" in m or "not" in m or "stall" in m:
        return "Non-advancing (stalled wound edge, no epithelial migration)"
    return "Advancing (wound actively healing, epithelial migration visible)"


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL SIGNAL EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

INFECTION_NOTE_FLAGS = [
    ("foul odor",        "foul/offensive wound odour — strong indicator of infection or colonisation"),
    ("foul odour",       "foul/offensive wound odour — strong indicator of infection or colonisation"),
    ("offensive odor",   "offensive wound odour — strong indicator of infection"),
    ("offensive odour",  "offensive wound odour — strong indicator of infection"),
    ("malodor",          "malodour — bacterial colonisation likely"),
    ("malodour",         "malodour — bacterial colonisation likely"),
    ("increased pain",   "increased wound pain — sign of acute infection or ischaemia"),
    ("pain increase",    "increasing wound pain — sign of acute infection"),
    ("more painful",     "increased wound pain — clinical sign of infection"),
    ("erythema",         "periWound erythema — clinical sign of local infection or cellulitis"),
    ("redness",          "periWound redness — possible local infection"),
    ("warm to touch",    "wound warmth — clinical sign of inflammation/infection"),
    ("swelling",         "periWound swelling — clinical sign of local infection"),
    ("purulent",         "purulent discharge — direct sign of infection"),
    ("pus",              "purulent discharge — direct sign of infection"),
    ("exudate changed",  "change in exudate character — possible infection progression"),
    ("cloudy",           "cloudy exudate — bacterial infection indicator"),
    ("cellulitis",       "cellulitis — systemic infection, urgent management required"),
]

HIGH_RISK_CONDITIONS = [
    "diabet", "immunocompro", "renal failure",
    "peripheral artery", "vascular", "neuropath",
    "steroid", "chemotherapy", "haemodialysis", "hemodialysis",
]

def extract_clinical_signals(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str,
) -> ClinicalContext:
    ctx = ClinicalContext(
        tissue_profile = tissue_profile,
        infection_norm = infection_norm,
        moisture_norm  = moisture_norm,
        edge_norm      = edge_norm,
        notes          = notes,
    )

    notes_lower = notes.lower()

    ctx.is_infected     = "infected" in infection_norm.lower() and "not" not in infection_norm.lower()
    ctx.is_dry          = "dry" in moisture_norm.lower() or "desicat" in moisture_norm.lower()
    ctx.is_high_exudate = "high exudate" in moisture_norm.lower() or "copious" in moisture_norm.lower()

    infection_flags_found = []
    for keyword, description in INFECTION_NOTE_FLAGS:
        if keyword in notes_lower:
            infection_flags_found.append(description)

    seen = set()
    for f in infection_flags_found:
        if f not in seen:
            ctx.notes_infection_flags.append(f)
            seen.add(f)

    is_high_risk = any(cond in notes_lower for cond in HIGH_RISK_CONDITIONS)

    if not ctx.is_infected and ctx.notes_infection_flags:
        ctx.notes_override = True
        ctx.is_infected    = True
        print(f"[CLINICAL] Notes-override: infection flags found: {ctx.notes_infection_flags}")
    elif not ctx.is_infected and is_high_risk and len(notes.strip()) > 20:
        ctx.notes_override = True
        print(f"[CLINICAL] High-risk patient: elevated infection caution")

    nv = tissue_profile.get("non_viable_pct", 0)
    if (
        (ctx.is_infected and nv > 50)
        or (ctx.is_infected and is_high_risk)
        or nv > 75
        or "cellulitis" in notes_lower
    ):
        ctx.escalation_needed = True
        print(f"[CLINICAL] Escalation flagged")

    ctx.high_nonviable    = nv > 25
    ctx.debridement_needed = nv > 25

    combination_reasons = []

    if ctx.is_infected and ctx.is_dry:
        ctx.needs_combination = True
        combination_reasons.append(
            "DRY + INFECTED: wound requires BOTH moisture donation (hydrogel/hydrocolloid) "
            "AND antimicrobial coverage (silver/iodine). These are NOT alternatives — "
            "applying only antimicrobial to a dry wound delays healing."
        )

    if ctx.is_infected and ctx.is_high_exudate and nv > 25:
        ctx.needs_combination = True
        combination_reasons.append(
            "HIGH EXUDATE + INFECTED + SIGNIFICANT NON-VIABLE TISSUE: wound requires "
            "BOTH an absorbent antimicrobial dressing AND active debridement consideration."
        )

    if ctx.notes_override and not ctx.is_dry and not ctx.is_high_exudate:
        ctx.needs_combination = True
        combination_reasons.append(
            "CLINICAL RED FLAGS IN NOTES with moderate moisture: antimicrobial dressing "
            "should be added alongside standard moisture management."
        )

    if combination_reasons:
        ctx.combination_rationale = "\n".join(combination_reasons)

    if ctx.is_dry and ctx.is_infected:
        ctx.primary_strategy = "hydrogel_plus_antimicrobial"
    elif ctx.is_high_exudate and ctx.is_infected:
        ctx.primary_strategy = "absorbent_antimicrobial"
    elif ctx.is_high_exudate:
        ctx.primary_strategy = "absorbent_non_antimicrobial"
    elif ctx.is_dry:
        ctx.primary_strategy = "moisture_donating"
    elif ctx.is_infected:
        ctx.primary_strategy = "antimicrobial_moderate_moisture"
    else:
        ctx.primary_strategy = "protective_healing"

    print(f"[CLINICAL] Strategy: {ctx.primary_strategy}")
    print(f"[CLINICAL] Combination needed: {ctx.needs_combination}")
    print(f"[CLINICAL] Notes override: {ctx.notes_override}")

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# MOISTURE-AWARE ANCHOR QUERY
# ══════════════════════════════════════════════════════════════════════════════

def build_moisture_aware_query(clinical_tissue, infection_norm, moisture_norm, notes=""):
    m    = moisture_norm.lower()
    note = f" {notes.strip()}" if notes.strip() else ""

    if "dry" in m or "desicat" in m:
        return (f"hydrogel dressing dry wound no exudate autolytic debridement "
                f"{clinical_tissue}{note}").strip()
    elif "high exudate" in m or "copious" in m:
        base = f"high exudate wound alginate foam absorbent dressing {clinical_tissue}"
        if "infected" in infection_norm.lower() and "not" not in infection_norm.lower():
            return (base + f" silver antimicrobial infection control{note}").strip()
        return (base + note).strip()
    else:
        if "granulat" in clinical_tissue:
            return (f"granulating wound low moderate exudate silicone foam "
                    f"non-adherent protection healing{note}").strip()
        return (f"moderate exudate wound dressing {clinical_tissue} "
                f"moisture balance{note}").strip()


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-AXIS QUERY EXPANSION (v3)
# ══════════════════════════════════════════════════════════════════════════════

def expand_queries_v3(clinical_ctx: ClinicalContext, llm) -> list[str]:
    tp    = clinical_ctx.tissue_profile
    clin  = tp["clinical_tissue"]
    inf   = clinical_ctx.infection_norm
    mois  = clinical_ctx.moisture_norm
    edge  = clinical_ctx.edge_norm
    notes = clinical_ctx.notes

    queries = []

    # Q_T
    if tp["non_viable_pct"] > 10:
        q_t = (f"wound debridement dressing {clin} necrotic slough tissue removal "
               f"autolytic enzymatic debridement")
    else:
        q_t = f"healthy granulating wound protection {clin} wound bed preparation"
    queries.append(q_t)
    print(f"  Q_T: {q_t}")

    # Q_I
    if clinical_ctx.is_infected:
        q_i = ("antimicrobial dressing wound infection silver iodine cadexomer "
               "critically colonised biofilm management infected wound")
        if clinical_ctx.escalation_needed:
            q_i += " systemic antibiotic referral cellulitis wound infection escalation"
    else:
        q_i = f"non-infected wound healing protection dressing no antimicrobial required"
    queries.append(q_i)
    print(f"  Q_I: {q_i}")

    # Q_M
    if clinical_ctx.is_dry:
        q_m = ("hydrogel moisture donation dry desiccated wound rehydration autolytic "
               "debridement dry wound bed no exudate moisture retentive dressing")
    elif clinical_ctx.is_high_exudate:
        q_m = ("high exudate absorption alginate foam hydrofiber aquacel heavily exuding "
               "wound copious drainage maceration prevention absorbent dressing")
    else:
        q_m = ("moderate exudate wound moisture balance silicone foam non-adherent "
               "dressing moist wound healing environment")
    queries.append(q_m)
    print(f"  Q_M: {q_m}")

    # Q_E
    if "non-advancing" in edge.lower():
        q_e = ("non-advancing wound edge stalled epithelial migration wound edge therapy "
               "periwound skin care advanced wound management edge stimulation")
    else:
        q_e = ("advancing wound edge epithelial migration protection healing dressing "
               "support epithelialisation")
    queries.append(q_e)
    print(f"  Q_E: {q_e}")

    # Q_COMBO
    if clinical_ctx.needs_combination:
        if clinical_ctx.primary_strategy == "hydrogel_plus_antimicrobial":
            q_combo = ("hydrogel combined with silver antimicrobial dressing dry infected wound "
                       "moisture donation plus infection control combination therapy desiccated "
                       "infected wound two dressing strategy")
        elif clinical_ctx.primary_strategy == "absorbent_antimicrobial":
            q_combo = ("silver alginate hydrofiber antimicrobial absorbent dressing infected "
                       "exuding wound debridement slough infection management high exudate "
                       "infected wound combination dressing")
        else:
            q_combo = (f"combination dressing strategy {clin} {mois} {inf} multi-factor "
                       f"wound management concurrent infection moisture management")
        queries.append(q_combo)
        print(f"  Q_COMBO: {q_combo}")

    # LLM-generated variants
    notes_suffix = f"\nAdditional clinical context: {notes.strip()}" if notes.strip() else ""
    prompt = f"""You are a wound care expert building search queries for a clinical guideline database.

Given this wound assessment:
  Tissue     : {clin}
  Infection  : {inf}
  Moisture   : {mois}
  Edge       : {edge}
  Strategy   : {clinical_ctx.primary_strategy}{notes_suffix}

Generate exactly 3 different search queries to find relevant wound dressing guidelines.
IMPORTANT: Each query must cover DIFFERENT clinical terminology. Do NOT let all 3 queries
focus only on infection — cover moisture management, tissue debridement, and combination
strategies in separate queries.

Use varied terminology: product categories, wound types, clinical actions.
Return ONLY the 3 queries, one per line, no numbering, no explanation."""

    try:
        response  = llm.invoke([HumanMessage(content=prompt)])
        llm_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        llm_queries = llm_queries[:3]
        while len(llm_queries) < 3:
            llm_queries.append(
                f"wound dressing recommendation {clin} {mois}"
                if len(llm_queries) == 0 else
                f"dressing recommendation {inf} wound management"
            )
        for i, q in enumerate(llm_queries):
            queries.append(q)
            print(f"  Q_LLM{i+1}: {q}")
    except Exception as e:
        print(f"[QUERY] LLM expansion failed: {e}")
        queries.append(f"wound dressing recommendation {clin} {mois}")
        queries.append(f"dressing selection {inf} {mois} wound care guideline")
        queries.append(f"clinical wound management {clin} dressing protocol")

    structured = (f"wound dressing recommendation: tissue={clin}, infection={inf}, "
                  f"moisture={mois}, edge={edge}")
    if notes.strip():
        structured += f", notes={notes.strip()[:100]}"
    queries.append(structured)
    print(f"  Q_STRUCT: {structured[:80]}...")

    queries.append(build_moisture_aware_query(clin, inf, mois, notes))

    if len(notes.strip()) > 20:
        if clinical_ctx.notes_override:
            q_notes = (f"wound care {notes.strip()} infection signs antimicrobial dressing "
                       f"diabetic wound management high risk patient")
        else:
            q_notes = f"wound care management {notes.strip()} dressing selection guideline"
        queries.append(q_notes)
        print(f"  Q_NOTES: {q_notes[:80]}...")

    return queries


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def build_hybrid_retriever(vectorstore, all_docs: list, k: int = 10):
    dense  = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25   = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])


def reciprocal_rank_fusion(results_per_query: list[list], k: int = 60):
    doc_scores  = {}
    doc_objects = {}
    for query_results in results_per_query:
        for rank, doc in enumerate(query_results, start=1):
            doc_id = hash(doc.page_content)
            doc_scores[doc_id]  = doc_scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_objects[doc_id] = doc
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return (
        [doc_objects[doc_id] for doc_id, _ in sorted_docs],
        doc_scores,
    )


def rerank_with_moisture_boost(query, docs, moisture_norm, is_infected, top_n=6):
    if not docs:
        return [], []

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs).tolist()
    m      = moisture_norm.lower()
    boosted = []

    for score, doc in zip(scores, docs):
        text  = doc.page_content.lower()
        boost = 0.0

        if "dry" in m or "desicat" in m:
            if "hydrogel" in text:
                boost = +0.35
            if any(t in text for t in ["alginate", "hydrofiber", "aquacel", "heavily absorbent"]):
                boost = -0.30
            if is_infected and any(t in text for t in ["silver", "antimicrobial", "iodine", "cadexomer"]):
                boost = max(boost, +0.10)
        elif "high exudate" in m or "copious" in m:
            if any(t in text for t in ["alginate", "foam", "hydrofiber", "highly absorbent"]):
                boost = +0.20
            if is_infected and any(t in text for t in ["silver alginate", "antimicrobial"]):
                boost = +0.30
            if "hydrogel" in text and "dry" not in text:
                boost = -0.30
        elif "moderate" in m:
            if any(t in text for t in ["silicone", "foam", "low to moderate"]):
                boost = +0.10
            if is_infected and any(t in text for t in ["silver", "antimicrobial"]):
                boost = +0.15

        boosted.append((score + boost, doc))

    boosted.sort(key=lambda x: x[0], reverse=True)
    top = boosted[:top_n]

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    raw_scores  = [s for s, _ in top]
    top_docs    = [doc for _, doc in top]
    norm_scores = [round(sigmoid(s), 3) for s in raw_scores]
    return top_docs, norm_scores


def retrieve_relevant_chunks(clinical_ctx: ClinicalContext, llm):
    queries = expand_queries_v3(clinical_ctx, llm)
    print(f"[RETRIEVAL] {len(queries)} queries generated")

    raw      = db.get()
    all_docs = [
        LC_Doc(page_content=pc, metadata=meta)
        for pc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    hybrid = build_hybrid_retriever(db, all_docs, k=10)

    results_per_query = []
    for i, query in enumerate(queries):
        try:
            results = hybrid.invoke(query)
            results_per_query.append(results)
            print(f"  Q{i+1} → {len(results)} chunks")
        except Exception as e:
            print(f"  Q{i+1} → Failed: {e}")
            results_per_query.append([])

    fused, rrf_scores = reciprocal_rank_fusion(results_per_query)
    print(f"[RETRIEVAL] After RRF: {len(fused)} unique chunks")

    structured_idx = min(len(queries) - 3, 8)
    anchor_query   = queries[structured_idx]

    top_chunks, reranker_scores = rerank_with_moisture_boost(
        anchor_query, fused, clinical_ctx.moisture_norm, clinical_ctx.is_infected, top_n=6,
    )
    print(f"[RETRIEVAL] After reranking: {len(top_chunks)} chunks")
    print(f"[RETRIEVAL] Reranker scores: {[f'{s:.2f}' for s in reranker_scores]}")

    return top_chunks, reranker_scores, rrf_scores


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING (v3)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rag_confidence_v3(top_chunks, reranker_scores, rrf_scores, clinical_ctx):
    tp             = clinical_ctx.tissue_profile
    infection_norm = clinical_ctx.infection_norm
    moisture_norm  = clinical_ctx.moisture_norm
    edge_norm      = clinical_ctx.edge_norm
    notes          = clinical_ctx.notes

    confidence = 1.0
    notes_out  = []

    if reranker_scores:
        avg_score = sum(reranker_scores) / len(reranker_scores)
        top_score = max(reranker_scores)

        if avg_score >= 0.70:
            notes_out.append(f"Strong retrieval relevance (avg {avg_score:.0%})")
        elif avg_score >= 0.50:
            confidence -= 0.10
            notes_out.append(f"Moderate retrieval relevance (avg {avg_score:.0%})")
        else:
            confidence -= 0.25
            notes_out.append(f"Low retrieval relevance (avg {avg_score:.0%}) — clinical judgement recommended")
        if top_score >= 0.80:
            confidence = min(1.0, confidence + 0.05)
            notes_out.append(f"Best matching chunk has high relevance ({top_score:.0%})")
    else:
        confidence -= 0.30
        notes_out.append("No reranker scores — retrieval may have failed")

    if rrf_scores and top_chunks:
        top_doc_ids = {hash(c.page_content) for c in top_chunks}
        top_rrf     = [v for k, v in rrf_scores.items() if k in top_doc_ids]
        if top_rrf:
            max_possible = len(top_rrf) * (1.0 / 61.0) * 2
            avg_rrf      = sum(top_rrf) / len(top_rrf)
            consensus    = min(1.0, avg_rrf / max_possible)
            if consensus >= 0.60:
                notes_out.append(f"High query consensus ({consensus:.0%} agreement)")
            elif consensus >= 0.30:
                confidence -= 0.05
                notes_out.append(f"Moderate query consensus ({consensus:.0%})")
            else:
                confidence -= 0.10
                notes_out.append(f"Low query consensus ({consensus:.0%})")

    required_tags = set()
    if tp["non_viable_pct"] > 10:
        required_tags.add("T")
    if clinical_ctx.is_infected:
        required_tags.add("I")
    if any(m in moisture_norm.lower() for m in ["dry", "high exudate", "copious"]):
        required_tags.add("M")
    if "non-advancing" in edge_norm.lower():
        required_tags.add("E")

    covered_tags = set()
    for chunk in top_chunks:
        chunk_tags = chunk.metadata.get("time_tags", "none")
        for tag in ["T", "I", "M", "E"]:
            if tag in chunk_tags.split(","):
                covered_tags.add(tag)

    if required_tags:
        missing = required_tags - covered_tags
        if not missing:
            notes_out.append(f"Retrieved chunks cover all T.I.M.E. factors ({', '.join(sorted(required_tags))})")
        elif len(missing) == 1:
            confidence -= 0.10
            notes_out.append(f"Missing factor '{list(missing)[0]}' in retrieved chunks")
        else:
            confidence -= 0.20
            notes_out.append(f"Missing factors: {', '.join(sorted(missing))}")

    if clinical_ctx.needs_combination:
        chunk_text_combined = " ".join(c.page_content.lower() for c in top_chunks)
        has_moisture_donating = any(t in chunk_text_combined for t in ["hydrogel", "moisture don", "rehydrat"])
        has_antimicrobial     = any(t in chunk_text_combined for t in ["silver", "antimicrobial", "iodine", "cadexomer"])
        has_absorbent         = any(t in chunk_text_combined for t in ["alginate", "foam", "hydrofiber", "absorbent"])

        if clinical_ctx.primary_strategy == "hydrogel_plus_antimicrobial":
            if has_moisture_donating and has_antimicrobial:
                notes_out.append("Combination evidence: moisture-donating and antimicrobial chunks retrieved")
            elif has_moisture_donating:
                confidence -= 0.10
                notes_out.append("Partial combination evidence: moisture chunks found, antimicrobial limited")
            elif has_antimicrobial:
                confidence -= 0.15
                notes_out.append("⚠️ Combination required but moisture-donating chunks not strongly retrieved")
            else:
                confidence -= 0.20
                notes_out.append("⚠️ Combination required but neither key dressing type retrieved")
        elif clinical_ctx.primary_strategy == "absorbent_antimicrobial":
            if has_absorbent and has_antimicrobial:
                notes_out.append("Combination evidence: absorbent and antimicrobial chunks retrieved")
            elif not has_antimicrobial:
                confidence -= 0.15
                notes_out.append("⚠️ Infected high-exudate wound but antimicrobial chunks not retrieved")

    if clinical_ctx.notes_override:
        confidence -= 0.05
        flag_text = (f"Flags: {', '.join(clinical_ctx.notes_infection_flags[:3])}"
                     if clinical_ctx.notes_infection_flags else
                     "High-risk patient notes indicate elevated infection caution")
        notes_out.append(f"Clinical notes contain infection indicators not in structured label. {flag_text}")

    nv = tp.get("non_viable_pct", 0)
    if nv > 75:
        confidence -= 0.10
        notes_out.append(f"Very high non-viable tissue ({nv:.0f}%) — specialist referral warranted")
    elif nv > 25:
        notes_out.append(f"Significant non-viable tissue ({nv:.0f}%) — debridement recommended")

    if clinical_ctx.escalation_needed:
        notes_out.append("⚠️ Escalation flagged: specialist review recommended")

    if len(notes.strip()) > 20:
        notes_out.append("Additional clinical notes incorporated into retrieval queries")

    confidence = max(0.0, min(1.0, round(confidence, 2)))

    if confidence >= 0.80:
        label = "HIGH"
    elif confidence >= 0.55:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "confidence_score": confidence,
        "confidence_label": label,
        "retrieval_notes":  notes_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT-AWARE GENERATION PROMPT (v3)
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation_v3(chunks, assessment_text, confidence_result, clinical_ctx):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        confidence_label = confidence_result["confidence_label"]
        confidence_score = confidence_result["confidence_score"]

        evidence_block = ""
        for i, chunk in enumerate(chunks, 1):
            source    = chunk.metadata.get("source",        "Unknown")
            authority = chunk.metadata.get("authority",     "")
            year      = chunk.metadata.get("year",          "")
            tags      = chunk.metadata.get("time_tags",     "none")
            original  = json.loads(chunk.metadata.get("original_content", "{}"))
            raw_text  = original.get("raw_text", chunk.page_content)

            evidence_block += f"\n--- Guideline Source {i}: {source}"
            if authority:
                evidence_block += f" [{authority}"
                if year:
                    evidence_block += f", {year}"
                evidence_block += "]"
            if tags != "none":
                evidence_block += f" | T.I.M.E. tags: {tags}"
            evidence_block += f" ---\n{raw_text}\n"

        if confidence_label == "LOW":
            confidence_guidance = (
                "⚠️ RETRIEVAL CONFIDENCE IS LOW: Only recommend what evidence clearly supports. "
                "State explicitly where evidence is limited and recommend specialist consultation."
            )
        elif confidence_label == "MEDIUM":
            confidence_guidance = (
                "RETRIEVAL CONFIDENCE IS MEDIUM: Indicate where you are extrapolating "
                "from general principles vs directly supported evidence."
            )
        else:
            confidence_guidance = (
                "RETRIEVAL CONFIDENCE IS HIGH: Guidelines strongly support this wound profile. "
                "Provide a specific, evidence-based recommendation."
            )

        combination_block = ""
        if clinical_ctx.needs_combination:
            combination_block = f"""
⚠️ COMBINATION THERAPY REQUIRED — THIS IS MANDATORY:
{clinical_ctx.combination_rationale}

You MUST recommend a COMBINATION of two complementary dressings, not a single dressing.
Do NOT pick only the antimicrobial dressing and ignore moisture management.
Do NOT pick only the absorbent dressing and ignore antimicrobial needs.
The Primary Dressing and Secondary Dressing sections MUST together address all active factors."""

        notes_override_block = ""
        if clinical_ctx.notes_override:
            flags_text = "\n".join(f"  - {f}" for f in clinical_ctx.notes_infection_flags)
            if not flags_text:
                flags_text = "  - High-risk patient: lower infection threshold applies"
            notes_override_block = f"""
⚠️ CLINICAL NOTES CONTAIN INFECTION INDICATORS:
{flags_text}

INSTRUCTIONS:
1. Treat this wound as LIKELY INFECTED based on clinical signs above
2. Antimicrobial component is MANDATORY, not optional
3. Address this discrepancy explicitly in Clinical Notes section"""

        escalation_block = ""
        if clinical_ctx.escalation_needed:
            escalation_block = """
⚠️ ESCALATION / REFERRAL REQUIRED:
Clinical Notes section MUST include a clear referral recommendation and interim strategy."""

        prompt = f"""You are a Clinical Wound Care Consultant providing an evidence-based dressing recommendation.

{confidence_guidance}
{combination_block}
{notes_override_block}
{escalation_block}

{assessment_text}

RETRIEVED CLINICAL GUIDELINES (ranked by relevance):
{evidence_block}

INSTRUCTIONS — provide your recommendation in the following structure:

## Primary Dressing
- State dressing CATEGORY and one specific brand example from guidelines
- Explain why this addresses the dominant clinical need

## Secondary Dressing
- If combination required: MUST address the other half of the combination
- If not required: explain why single dressing suffices

## Rationale by T.I.M.E. Factor
Address EACH factor independently:
- T (Tissue): Link tissue composition to dressing mechanism
- I (Infection): Address infection status — if infected, explain antimicrobial choice
- M (Moisture): Explain moisture level and absorbency/donation requirement
- E (Edge): Address wound edge status and intervention needed

## Contraindications
- List dressing types NOT to use and why

## Dressing Change Frequency
- Recommended frequency with guideline citation

## Application Tips
- Practical tips from guidelines

## Clinical Notes
- Patient-specific factors from notes
- Referral recommendation if escalation needed
- Layering/sequencing instructions if combination therapy prescribed

Confidence: {confidence_label} ({confidence_score:.0%})
Base EVERY recommendation on the retrieved guideline evidence. Cite source numbers."""

        response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])]) # For OpenAI GPT
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
    necrotic_pct:        float = Form(...),
    slough_pct:          float = Form(...),
    granulation_pct:     float = Form(...),
    infection:           str   = Form(...),
    moisture:            str   = Form(...),
    edge:                str   = Form(...),
    notes:               str   = Form(""),
    tissue_confidence:   float = Form(0.0),  # [NEW v4] GMM confidence from image pipeline (informational)
):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ── Step 1: Normalise inputs ──────────────────────────────────────────
        tissue_profile = interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct)
        infection_norm = normalize_infection(infection)
        moisture_norm  = normalize_moisture(moisture)
        edge_norm      = normalize_edge(edge)

        print(f"\n[INPUT] N={necrotic_pct}% S={slough_pct}% G={granulation_pct}%")
        print(f"[INPUT] Infection       : {infection} → {infection_norm}")
        print(f"[INPUT] Moisture        : {moisture}  → {moisture_norm}")
        print(f"[INPUT] Edge            : {edge}       → {edge_norm}")
        print(f"[INPUT] Notes           : {notes[:80]}{'...' if len(notes) > 80 else ''}")
        print(f"[INPUT] Tissue confidence (GMM): {tissue_confidence:.3f}")

        # ── Step 2: Clinical signal extraction ───────────────────────────────
        clinical_ctx = extract_clinical_signals(
            tissue_profile = tissue_profile,
            infection_norm = infection_norm,
            moisture_norm  = moisture_norm,
            edge_norm      = edge_norm,
            notes          = notes,
        )

        # ── Step 3: RAG retrieval ─────────────────────────────────────────────
        top_chunks, reranker_scores, rrf_scores = retrieve_relevant_chunks(
            clinical_ctx = clinical_ctx,
            llm          = llm,
        )

        # ── Step 4: Confidence scoring ────────────────────────────────────────
        confidence_result = calculate_rag_confidence_v3(
            top_chunks      = top_chunks,
            reranker_scores = reranker_scores,
            rrf_scores      = rrf_scores,
            clinical_ctx    = clinical_ctx,
        )
        print(f"[CONFIDENCE] {confidence_result['confidence_score']:.0%} "
              f"({confidence_result['confidence_label']})")

        # ── Step 5: Build assessment text ─────────────────────────────────────
        assessment_text = f"""T.I.M.E. WOUND ASSESSMENT (from VerdaSense ML pipeline):

        T (Tissue)    : {tissue_profile['tissue_label']}
                        Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
                        Non-viable load: {tissue_profile['non_viable_pct']}% {'(HIGH — debridement consideration)' if tissue_profile['high_non_viable'] else ''}

        I (Infection) : {infection_norm}  [raw label: {infection}]
        M (Moisture)  : {moisture_norm}  [raw label: {moisture}]
        E (Edge)      : {edge_norm}  [raw label: {edge}]

        CLINICAL CONTEXT (pre-reasoning):
        Strategy         : {clinical_ctx.primary_strategy}
        Combination need : {'YES — ' + clinical_ctx.combination_rationale[:120] + '...' if clinical_ctx.needs_combination else 'No'}
        Notes override   : {'YES — infection signs in notes' if clinical_ctx.notes_override else 'No'}
        Escalation       : {'YES' if clinical_ctx.escalation_needed else 'No'}
        """
        if notes.strip():
            assessment_text += f"\nADDITIONAL CLINICAL NOTES:\n  {notes.strip()}\n"
        if clinical_ctx.notes_infection_flags:
            assessment_text += f"\nINFECTION FLAGS IN NOTES:\n"
            for flag in clinical_ctx.notes_infection_flags:
                assessment_text += f"  ⚠️  {flag}\n"

        # ── Step 6: LLM generation ────────────────────────────────────────────
        result = generate_recommendation_v3(
            chunks            = top_chunks,
            assessment_text   = assessment_text,
            confidence_result = confidence_result,
            clinical_ctx      = clinical_ctx,
        )

        # ── Step 7: Build response ────────────────────────────────────────────
        sources = list(dict.fromkeys(
            chunk.metadata.get("source", "Unknown") for chunk in top_chunks
        ))

        chunk_texts = [chunk.page_content for chunk in top_chunks] 

        return JSONResponse({
            "result":           result,
            "sources":          sources,
            "chunk_texts":      chunk_texts,    # ADD THIS
            "confidence_score": confidence_result["confidence_score"],
            "confidence_label": confidence_result["confidence_label"],
            "retrieval_notes":  confidence_result["retrieval_notes"],
            "tissue_breakdown": {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores":    reranker_scores,
            "clinical_flags": {
                "needs_combination":     clinical_ctx.needs_combination,
                "notes_override":        clinical_ctx.notes_override,
                "escalation_needed":     clinical_ctx.escalation_needed,
                "primary_strategy":      clinical_ctx.primary_strategy,
                "infection_flags":       clinical_ctx.notes_infection_flags,
                "combination_rationale": clinical_ctx.combination_rationale,
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "result":           f"System error: {str(e)}",
                "sources":          [],
                "confidence_label": "LOW",
                "confidence_score": 0.0,
                "retrieval_notes":  ["System error during retrieval"],
                "clinical_flags":   {},
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
