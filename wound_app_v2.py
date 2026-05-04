"""
wound_app_v2.py  —  VerdaSense RAG-only Retrieval & Generation Pipeline
════════════════════════════════════════════════════════════════════════
Changes from previous version:
  • Rule engine removed entirely — LLM decides from RAG evidence
  • RAG-only confidence score (reranker quality + RRF consensus + T.I.M.E. coverage)
  • Additional notes injected into query expansion for semantic search
  • Moisture-aware reranking retained (critical safety check)
  • Fallback prompt when retrieval confidence is LOW
  • generate_rag_recommendation() replaces generate_hybrid_recommendation()
"""

import os
import json
import math
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
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
# INPUT NORMALISATION
# Converts simplified ML model output labels into clinical vocabulary
# ══════════════════════════════════════════════════════════════════════════════

def interpret_tissue_percentages(
    necrotic_pct: float,
    slough_pct: float,
    granulation_pct: float,
) -> dict:
    """
    Converts raw K-Means tissue percentages into a structured clinical profile.
    MOH Malaysia threshold: <25% vs >25% non-viable tissue is clinically significant.
    """
    total = necrotic_pct + slough_pct + granulation_pct
    if total == 0:
        return {
            "tissue_label":    "insufficient tissue data",
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

    # Build human-readable tissue label
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

    # Clinical tissue description for search queries
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
        "clinical_tissue": clinical_tissue,   # used directly in search queries
        "dominant_tissue": dominant,
        "necrotic_pct":    round(n, 1),
        "slough_pct":      round(s, 1),
        "granulation_pct": round(g, 1),
        "non_viable_pct":  round(non_viable_pct, 1),
        "high_non_viable": high_non_viable,
        "healing_progress": round(g / 100, 2),
    }


def normalize_infection(label: str) -> str:
    """Map Infected / Not Infected → clinical description."""
    m = label.lower().strip()
    if "not" in m or "no" in m:
        return "Not infected (no clinical signs)"
    return "Locally infected (erythema, warmth, swelling at wound edge)"


def normalize_moisture(label: str) -> str:
    """Map Low / Moderate / High → clinical exudate description."""
    m = label.lower().strip()
    if m in ("high", "high exudate"):
        return "High exudate (copious drainage, maceration risk)"
    elif m in ("low", "low exudate", "dry"):
        return "Dry (no exudate, desiccated wound bed)"
    else:
        return "Moderate exudate (frequent dressing changes needed)"


def normalize_edge(label: str) -> str:
    """Map Advancing / Non Advancing → clinical description."""
    m = label.lower().strip()
    if "non" in m or "not" in m or "stall" in m:
        return "Non-advancing (stalled wound edge, no epithelial migration)"
    return "Advancing (wound actively healing, epithelial migration visible)"


# ══════════════════════════════════════════════════════════════════════════════
# MOISTURE-AWARE ANCHOR QUERY
# Steers retrieval toward the correct absorbency class — the critical safety check
# ══════════════════════════════════════════════════════════════════════════════

def build_moisture_aware_query(
    clinical_tissue: str,
    infection_norm: str,
    moisture_norm: str,
    notes: str = "",
) -> str:
    """
    Builds a targeted query based on moisture level.
    Dry wounds → hydrogel (moisture-donating)
    High exudate → alginate/foam (absorbing)
    Includes patient notes for specificity.
    """
    m    = moisture_norm.lower()
    note = f" {notes.strip()}" if notes.strip() else ""

    if "dry" in m or "desicat" in m:
        return (
            f"hydrogel dressing dry wound no exudate autolytic debridement "
            f"{clinical_tissue}{note}"
        ).strip()
    elif "high exudate" in m or "copious" in m:
        base = f"high exudate wound alginate foam absorbent dressing {clinical_tissue}"
        if "infected" in infection_norm.lower() and "not" not in infection_norm.lower():
            return (base + f" silver antimicrobial infection control{note}").strip()
        return (base + note).strip()
    else:
        if "granulat" in clinical_tissue:
            return (
                f"granulating wound low moderate exudate silicone foam "
                f"non-adherent protection healing{note}"
            ).strip()
        return (
            f"moderate exudate wound dressing {clinical_tissue} "
            f"moisture balance{note}"
        ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-QUERY EXPANSION  (notes-aware)
# Notes are injected into ALL query variants, not just the LLM prompt
# ══════════════════════════════════════════════════════════════════════════════

def expand_queries(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str,
    llm,
) -> list[str]:
    """
    Generates query variants for RRF fusion.

    Query breakdown:
      Q1-Q3 : LLM-generated variants using different clinical terminology
      Q4    : Structured T.I.M.E. query (always reliable anchor)
      Q5    : Moisture-aware safety query (ensures correct absorbency class)
      Q6    : Notes-focused query (if notes provided — patient-specific context)

    Notes are included in Q1-Q3 prompt AND Q4 structured query AND Q6,
    so patient-specific context influences retrieval across multiple angles.
    """
    clinical_tissue = tissue_profile["clinical_tissue"]
    notes_suffix    = f"\nAdditional clinical context: {notes.strip()}" if notes.strip() else ""

    # ── Q1-Q3: LLM-generated variants ────────────────────────────────────────
    prompt = f"""You are a wound care expert building search queries for a clinical guideline database.

Given this wound assessment:
  Tissue     : {clinical_tissue}
  Infection  : {infection_norm}
  Moisture   : {moisture_norm}
  Edge       : {edge_norm}{notes_suffix}

Generate exactly 3 different search queries to find relevant wound dressing guidelines.
Use different clinical terminology in each query — vary between:
  - Dressing product categories (foam, alginate, hydrogel, silver, hydrofiber, silicone)
  - Wound condition descriptions (necrotic, sloughy, granulating, infected, dry, exuding)
  - Clinical actions (debridement, absorption, moisture donation, infection control, autolysis)
  - Include relevant patient context from the additional notes where appropriate

Return ONLY the 3 queries, one per line, no numbering, no explanation."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        llm_queries = llm_queries[:3]
    except Exception as e:
        print(f"[QUERY] LLM expansion failed: {e}, using fallbacks")
        llm_queries = []

    # Pad if LLM returned fewer than 3
    fallbacks = [
        f"wound dressing selection {clinical_tissue} {moisture_norm}",
        f"dressing recommendation {infection_norm} wound",
    ]
    while len(llm_queries) < 3:
        llm_queries.append(fallbacks[len(llm_queries) % len(fallbacks)])

    queries = llm_queries  # Q1-Q3

    # ── Q4: Structured T.I.M.E. query (always included) ──────────────────────
    structured = (
        f"wound dressing recommendation: "
        f"tissue={clinical_tissue}, "
        f"infection={infection_norm}, "
        f"moisture={moisture_norm}, "
        f"edge={edge_norm}"
    )
    if notes.strip():
        structured += f", notes={notes.strip()[:100]}"   # cap length
    queries.append(structured)  # Q4

    # ── Q5: Moisture-aware safety query ──────────────────────────────────────
    queries.append(
        build_moisture_aware_query(clinical_tissue, infection_norm, moisture_norm, notes)
    )  # Q5

    # ── Q6: Notes-focused query (only if notes are substantial) ──────────────
    if len(notes.strip()) > 20:
        queries.append(
            f"wound care management {notes.strip()} dressing selection guideline"
        )   # Q6

    return queries


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def build_hybrid_retriever(vectorstore, all_docs: list, k: int = 10):
    dense  = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25   = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[0.6, 0.4],
    )


def reciprocal_rank_fusion(
    results_per_query: list[list],
    k: int = 60,
) -> tuple[list, dict]:
    """
    Standard RRF fusion. Also returns rrf_scores dict for confidence calculation.
    doc_id → cumulative RRF score (higher = more queries agreed on this chunk).
    """
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


def rerank_with_moisture_boost(
    query: str,
    docs: list,
    moisture_norm: str,
    top_n: int = 5,
) -> tuple[list, list[float]]:
    """
    Cross-encoder reranker with moisture-context boost/penalty.
    Dry wound → boosts hydrogel chunks, penalises absorbent dressings.
    High exudate → boosts alginate/foam, penalises hydrogel.
    Returns (top_docs, sigmoid-normalised scores 0-1).
    """
    if not docs:
        return [], []

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs).tolist()

    m = moisture_norm.lower()
    boosted = []
    for score, doc in zip(scores, docs):
        text  = doc.page_content.lower()
        boost = 0.0

        if "dry" in m or "desicat" in m:
            if "hydrogel" in text:
                boost = +0.30
            if any(t in text for t in ["alginate", "hydrofiber", "aquacel", "heavily absorbent"]):
                boost = -0.30

        elif "high exudate" in m or "copious" in m:
            if any(t in text for t in ["alginate", "foam", "hydrofiber", "highly absorbent"]):
                boost = +0.20
            if "hydrogel" in text and "dry" not in text:
                boost = -0.30

        elif "moderate" in m:
            if any(t in text for t in ["silicone", "foam", "low to moderate"]):
                boost = +0.10

        boosted.append((score + boost, doc))

    boosted.sort(key=lambda x: x[0], reverse=True)
    top = boosted[:top_n]

    raw_scores = [s for s, _ in top]
    top_docs   = [doc for _, doc in top]

    # Sigmoid normalisation → 0-1 range for confidence display
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    norm_scores = [round(sigmoid(s), 3) for s in raw_scores]
    return top_docs, norm_scores


# ══════════════════════════════════════════════════════════════════════════════
# RAG CONFIDENCE SCORING
# Pure RAG confidence — no rule engine involved
# Three signals:
#   1. Reranker quality   — how relevant are the retrieved chunks?
#   2. RRF consensus      — how many queries agreed on the same chunks?
#   3. T.I.M.E. coverage  — do retrieved chunks cover the assessed factors?
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rag_confidence(
    top_chunks: list,
    reranker_scores: list[float],
    rrf_scores: dict,
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str,
) -> dict:
    """
    Returns a confidence dict with score (0-1), label, and explanation notes.
    """
    confidence = 1.0
    notes_out  = []

    # ── SIGNAL 1: Reranker quality ────────────────────────────────────────────
    if reranker_scores:
        avg_score = sum(reranker_scores) / len(reranker_scores)
        top_score = max(reranker_scores)

        if avg_score >= 0.70:
            notes_out.append(
                f"Strong retrieval relevance — top chunks closely match wound profile "
                f"(avg {avg_score:.0%})"
            )
        elif avg_score >= 0.50:
            confidence -= 0.10
            notes_out.append(
                f"Moderate retrieval relevance (avg {avg_score:.0%}) — "
                f"guidelines partially match this wound type"
            )
        else:
            confidence -= 0.25
            notes_out.append(
                f"Low retrieval relevance (avg {avg_score:.0%}) — "
                f"knowledge base may not have specific guidelines for this wound profile; "
                f"clinical judgement recommended"
            )

        # Bonus: top chunk is very strong
        if top_score >= 0.80:
            confidence = min(1.0, confidence + 0.05)
            notes_out.append(f"Best matching chunk has high relevance ({top_score:.0%})")

    else:
        confidence -= 0.30
        notes_out.append("No reranker scores — retrieval may have failed")

    # ── SIGNAL 2: RRF consensus ───────────────────────────────────────────────
    # How many queries independently agreed on the top chunks?
    # High RRF score = chunk appeared high in multiple query rankings
    if rrf_scores and top_chunks:
        top_doc_ids = {hash(c.page_content) for c in top_chunks}
        top_rrf     = [v for k, v in rrf_scores.items() if k in top_doc_ids]

        if top_rrf:
            # Max possible RRF score (rank 1 in all 6 queries): 6 × (1/61) ≈ 0.098
            max_possible = 6 * (1.0 / 61.0)
            avg_rrf      = sum(top_rrf) / len(top_rrf)
            consensus    = min(1.0, avg_rrf / max_possible)

            if consensus >= 0.60:
                notes_out.append(
                    f"High query consensus — multiple search strategies retrieved the same chunks "
                    f"({consensus:.0%} agreement)"
                )
            elif consensus >= 0.30:
                confidence -= 0.05
                notes_out.append(
                    f"Moderate query consensus ({consensus:.0%}) — "
                    f"some variation across search strategies"
                )
            else:
                confidence -= 0.15
                notes_out.append(
                    f"Low query consensus ({consensus:.0%}) — "
                    f"different queries returned different chunks; "
                    f"wound profile may be atypical"
                )

    # ── SIGNAL 3: T.I.M.E. tag coverage ──────────────────────────────────────
    # Do retrieved chunks cover the T.I.M.E. factors present in this wound?
    required_tags = set()
    if tissue_profile["non_viable_pct"] > 10:
        required_tags.add("T")
    if "infected" in infection_norm.lower() and "not" not in infection_norm.lower():
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
            notes_out.append(
                f"Retrieved chunks cover all relevant T.I.M.E. factors "
                f"({', '.join(sorted(required_tags))})"
            )
        elif len(missing) == 1:
            confidence -= 0.10
            notes_out.append(
                f"Retrieved chunks are missing coverage for factor "
                f"'{list(missing)[0]}' — recommendation may be incomplete for this aspect"
            )
        else:
            confidence -= 0.20
            notes_out.append(
                f"Retrieved chunks missing coverage for factors: "
                f"{', '.join(sorted(missing))} — consider supplementing with clinical judgement"
            )

    # ── SIGNAL 4: Non-viable tissue load flag ────────────────────────────────
    nv = tissue_profile.get("non_viable_pct", 0)
    if nv > 75:
        confidence -= 0.10
        notes_out.append(
            f"Very high non-viable tissue load ({nv:.0f}%) — "
            f"surgical debridement and specialist referral may be warranted "
            f"regardless of dressing selection"
        )
    elif nv > 25:
        notes_out.append(
            f"Significant non-viable tissue ({nv:.0f}%) — "
            f"debridement should be considered alongside dressing selection"
        )

    # ── SIGNAL 5: Notes enrich confidence ────────────────────────────────────
    if len(notes.strip()) > 20:
        notes_out.append(
            "Additional clinical notes incorporated into retrieval queries "
            "for patient-specific context"
        )

    # ── Final score and label ─────────────────────────────────────────────────
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
# MAIN RETRIEVAL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_relevant_chunks(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str,
    llm,
) -> tuple[list, list[float], dict]:
    """
    Full retrieval pipeline:
      1. Multi-query expansion (5-6 queries, notes-aware)
      2. Hybrid semantic + BM25 search for each query
      3. RRF fusion across all query results
      4. Moisture-boosted cross-encoder reranking → top 5 chunks
    Returns: (top_chunks, reranker_scores, rrf_scores)
    """
    queries = expand_queries(
        tissue_profile, infection_norm, moisture_norm, edge_norm, notes, llm
    )

    print(f"[RETRIEVAL] {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q}")

    # Load all docs for BM25 (needed at retrieval time)
    raw      = db.get()
    all_docs = [
        LC_Doc(page_content=pc, metadata=meta)
        for pc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    hybrid = build_hybrid_retriever(db, all_docs, k=10)

    results_per_query = []
    for query in queries:
        try:
            results = hybrid.invoke(query)
            results_per_query.append(results)
            print(f"  → {len(results)} chunks")
        except Exception as e:
            print(f"  → Query failed: {e}")
            results_per_query.append([])

    fused, rrf_scores = reciprocal_rank_fusion(results_per_query)
    print(f"[RETRIEVAL] After RRF: {len(fused)} unique chunks")

    # Use Q4 (structured T.I.M.E. query) as anchor for reranking
    anchor_query = queries[3] if len(queries) > 3 else queries[-1]

    top_chunks, reranker_scores = rerank_with_moisture_boost(
        anchor_query,
        fused,
        moisture_norm,
        top_n=5,
    )
    print(f"[RETRIEVAL] After reranking: {len(top_chunks)} chunks passed to LLM")
    print(f"[RETRIEVAL] Reranker scores: {[f'{s:.2f}' for s in reranker_scores]}")

    return top_chunks, reranker_scores, rrf_scores


# ══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION  —  RAG-only, LLM decides from evidence
# ══════════════════════════════════════════════════════════════════════════════

def generate_rag_recommendation(
    chunks: list,
    assessment_text: str,
    confidence_result: dict,
) -> str:
    """
    Full RAG-driven recommendation. No rule engine constraints.
    The LLM reasons from retrieved evidence and the structured assessment.
    When confidence is LOW, prompt includes explicit uncertainty guidance.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        confidence_label = confidence_result["confidence_label"]
        confidence_score = confidence_result["confidence_score"]

        # Build guideline evidence block
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

        # ── Confidence-aware prompt ───────────────────────────────────────────
        if confidence_label == "LOW":
            confidence_guidance = (
                "⚠️  RETRIEVAL CONFIDENCE IS LOW: The knowledge base may not have "
                "specific guidelines for this wound profile. Base your recommendation "
                "ONLY on what the retrieved evidence clearly supports. "
                "Explicitly state where evidence is limited and recommend specialist "
                "consultation for aspects not covered by the guidelines below."
            )
        elif confidence_label == "MEDIUM":
            confidence_guidance = (
                "RETRIEVAL CONFIDENCE IS MEDIUM: Guidelines partially match this wound. "
                "Clearly indicate where you are extrapolating from general principles "
                "vs directly supported evidence."
            )
        else:
            confidence_guidance = (
                "RETRIEVAL CONFIDENCE IS HIGH: Guidelines strongly support this wound profile. "
                "Provide a specific, evidence-based recommendation."
            )

        prompt = f"""You are a Clinical Wound Care Consultant providing an evidence-based dressing recommendation.

{confidence_guidance}

{assessment_text}

RETRIEVED CLINICAL GUIDELINES (ranked by relevance to this wound):
{evidence_block}

INSTRUCTIONS — provide your recommendation structured as follows:

## Primary Dressing
- State the dressing CATEGORY and give one specific brand example found in the guidelines
- Explain why this category is indicated for this wound's tissue, infection, and moisture profile

## Secondary Dressing
- State secondary dressing if needed, or explain why none is required
- Reference the specific wound characteristic that drives this choice

## Rationale by T.I.M.E. Factor
- T (Tissue): Link tissue composition ({assessment_text.split('Necrotic:')[1].split('|')[0].strip() if 'Necrotic:' in assessment_text else 'as described'}) to dressing mechanism
- I (Infection): Address infection status and whether antimicrobial properties are needed
- M (Moisture): Explain how moisture level drives absorbency or moisture-donation requirement
- E (Edge): Address wound edge status and whether intervention is needed

## Contraindications
- List what dressing types must NOT be used for this wound and why (from guidelines)

## Dressing Change Frequency
- State recommended frequency with guideline source citation

## Application Tips
- Practical tips from guidelines relevant to this wound type

## Clinical Notes
- Address any patient-specific factors from the additional notes
- State if referral or specialist review is recommended based on wound characteristics

Confidence in this recommendation: {confidence_label} ({confidence_score:.0%})
Base EVERY recommendation on the retrieved guideline evidence above.
Cite which source supports each key recommendation.
If evidence is insufficient for any section, state this explicitly."""

        response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
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
        name="wound_index_v2.html",
        context={"request": request},
    )


@app.post("/get_recommendation")
async def get_recommendation(
    necrotic_pct:    float = Form(...),
    slough_pct:      float = Form(...),
    granulation_pct: float = Form(...),
    infection:       str   = Form(...),   # "Infected" or "Not Infected"
    moisture:        str   = Form(...),   # "Low" | "Moderate" | "High"
    edge:            str   = Form(...),   # "Advancing" | "Non Advancing"
    notes:           str   = Form(""),    # free-text, injected into retrieval
):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ── STEP 1: Normalise inputs ──────────────────────────────────────────
        tissue_profile = interpret_tissue_percentages(
            necrotic_pct, slough_pct, granulation_pct
        )
        infection_norm = normalize_infection(infection)
        moisture_norm  = normalize_moisture(moisture)
        edge_norm      = normalize_edge(edge)

        print(f"\n[INPUT] N={necrotic_pct}% S={slough_pct}% G={granulation_pct}%")
        print(f"[INPUT] Infection : {infection} → {infection_norm}")
        print(f"[INPUT] Moisture  : {moisture}  → {moisture_norm}")
        print(f"[INPUT] Edge      : {edge}       → {edge_norm}")
        print(f"[INPUT] Notes     : {notes[:80]}{'...' if len(notes) > 80 else ''}")
        print(f"[INPUT] Tissue    : {tissue_profile['clinical_tissue']}")

        # ── STEP 2: RAG retrieval (notes-aware) ───────────────────────────────
        top_chunks, reranker_scores, rrf_scores = retrieve_relevant_chunks(
            tissue_profile = tissue_profile,
            infection_norm = infection_norm,
            moisture_norm  = moisture_norm,
            edge_norm      = edge_norm,
            notes          = notes,
            llm            = llm,
        )

        # ── STEP 3: RAG confidence scoring ────────────────────────────────────
        confidence_result = calculate_rag_confidence(
            top_chunks      = top_chunks,
            reranker_scores = reranker_scores,
            rrf_scores      = rrf_scores,
            tissue_profile  = tissue_profile,
            infection_norm  = infection_norm,
            moisture_norm   = moisture_norm,
            edge_norm       = edge_norm,
            notes           = notes,
        )
        print(f"[CONFIDENCE] {confidence_result['confidence_score']:.0%} "
              f"({confidence_result['confidence_label']})")

        # ── STEP 4: Build structured assessment text for LLM ─────────────────
        assessment_text = f"""T.I.M.E. WOUND ASSESSMENT (from VerdaSense ML pipeline):

  T (Tissue)    : {tissue_profile['tissue_label']}
                  Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
                  Non-viable load: {tissue_profile['non_viable_pct']}% {'(HIGH — debridement consideration)' if tissue_profile['high_non_viable'] else ''}

  I (Infection) : {infection_norm}  [raw label: {infection}]
  M (Moisture)  : {moisture_norm}  [raw label: {moisture}]
  E (Edge)      : {edge_norm}  [raw label: {edge}]
"""
        if notes.strip():
            assessment_text += f"\nADDITIONAL CLINICAL NOTES:\n  {notes.strip()}\n"

        # ── STEP 5: LLM generation ────────────────────────────────────────────
        result = generate_rag_recommendation(
            chunks             = top_chunks,
            assessment_text    = assessment_text,
            confidence_result  = confidence_result,
        )

        # ── STEP 6: Build response ────────────────────────────────────────────
        # Deduplicate sources while preserving order
        sources = list(dict.fromkeys(
            chunk.metadata.get("source", "Unknown") for chunk in top_chunks
        ))

        return JSONResponse({
            "result":           result,
            "sources":          sources,
            "confidence_score": confidence_result["confidence_score"],
            "confidence_label": confidence_result["confidence_label"],
            "retrieval_notes":  confidence_result["retrieval_notes"],
            "tissue_breakdown": {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores": reranker_scores,
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
                "retrieval_notes":  ["System error occurred during retrieval"],
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import os
# import json
# import math
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.messages import HumanMessage
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_core.documents import Document as LC_Doc
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import CrossEncoder
# from dotenv import load_dotenv
# import torch

# from WoundDressingRuleEngineV2 import WoundDressingRuleEngine

# load_dotenv()

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# templates = Jinja2Templates(directory="templates")

# # ── Embedding model — must match ingestion ────────────────────────
# embedding_model = HuggingFaceEmbeddings(
#     model_name="abhinand/MedEmbed-large-v0.1",
#     model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#     encode_kwargs={"normalize_embeddings": True},
# )

# # ── Cross-encoder reranker ────────────────────────────────────────
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # ── Rule engine ───────────────────────────────────────────────────
# rule_engine = WoundDressingRuleEngine()

# # ── Load v2 vector store ──────────────────────────────────────────
# def load_wound_db(persist_directory: str = "./db_wound_care_v2"):
#     print(f"Loading Wound Care KB from {persist_directory}...")
#     return Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_model,
#         collection_metadata={"hnsw:space": "cosine"},
#     )

# db = load_wound_db()

# # ═══════════════════════════════════════════════════════════════════
# # INPUT NORMALISATION
# # The front-end sends simplified labels from the ML models.
# # We convert those into rule-engine vocabulary here so neither the
# # UI nor the rule engine needs to know about each other's format.
# # ═══════════════════════════════════════════════════════════════════

# def interpret_tissue_percentages(
#     necrotic_pct: float,
#     slough_pct: float,
#     granulation_pct: float,
# ) -> dict:
#     """
#     Converts raw K-Means tissue percentages into a structured clinical profile
#     that feeds into the rule engine and retrieval pipeline.
#     MOH Malaysia guideline threshold: <25% vs >25% non-viable tissue.
#     """
#     total = necrotic_pct + slough_pct + granulation_pct
#     if total == 0:
#         return {
#             "tissue_label": "unknown",
#             "dominant_tissue": "unknown",
#             "necrotic_pct": 0.0,
#             "slough_pct": 0.0,
#             "granulation_pct": 0.0,
#             "non_viable_pct": 0.0,
#             "high_non_viable": False,
#             "rule_input": "Granulation (red/pink beefy tissue)",
#             "healing_progress": 0.0,
#         }

#     n = necrotic_pct / total * 100
#     s = slough_pct   / total * 100
#     g = granulation_pct / total * 100

#     dominant = max([("necrotic", n), ("slough", s), ("granulation", g)], key=lambda x: x[1])[0]
#     non_viable_pct = n + s
#     high_non_viable = non_viable_pct > 25

#     # Build readable label
#     parts = []
#     if g >= 70:
#         parts.append(f"predominantly granulating ({g:.0f}%)")
#     elif g >= 40:
#         parts.append(f"mixed tissue, significant granulation ({g:.0f}%)")
#     if n >= 25:
#         parts.append(f"high necrotic load ({n:.0f}%)")
#     elif n > 0:
#         parts.append(f"some necrotic tissue ({n:.0f}%)")
#     if s >= 25:
#         parts.append(f"significant slough ({s:.0f}%)")
#     elif s > 0:
#         parts.append(f"some slough ({s:.0f}%)")
#     tissue_label = ", ".join(parts) if parts else "mixed tissue"

#     # Map to rule engine vocabulary
#     if n >= 25 and s >= 25:
#         rule_input = "Mixed (necrotic and slough)"
#     elif n >= 25:
#         rule_input = "Necrotic (black/brown eschar)"
#     elif s >= 25:
#         rule_input = "Slough (yellow/fibrinous tissue)" if g < 30 else "Mixed (slough and granulation)"
#     elif g >= 70:
#         rule_input = "Granulation (red/pink beefy tissue)"
#     else:
#         rule_input = "Mixed (slough and granulation)"

#     return {
#         "tissue_label":    tissue_label,
#         "dominant_tissue": dominant,
#         "necrotic_pct":    round(n, 1),
#         "slough_pct":      round(s, 1),
#         "granulation_pct": round(g, 1),
#         "non_viable_pct":  round(non_viable_pct, 1),
#         "high_non_viable": high_non_viable,
#         "rule_input":      rule_input,
#         "healing_progress": round(g / 100, 2),
#     }


# def normalize_infection(label: str) -> str:
#     """Map simplified Infected / Not Infected → rule engine vocabulary."""
#     m = label.lower().strip()
#     if "not" in m or "no" in m:
#         return "Not infected (no clinical signs)"
#     return "Locally infected (erythema, warmth, swelling at wound edge)"


# def normalize_moisture(label: str) -> str:
#     """Map Low / Moderate / High → rule engine vocabulary."""
#     m = label.lower().strip()
#     if m in ("high", "high exudate"):
#         return "High exudate (copious drainage, maceration risk)"
#     elif m in ("low", "low exudate", "dry"):
#         return "Dry (no exudate, desiccated wound bed)"
#     else:
#         return "Moderate exudate (frequent dressing changes needed)"


# def normalize_edge(label: str) -> str:
#     """Map Advancing / Non Advancing → rule engine vocabulary."""
#     m = label.lower().strip()
#     if "non" in m or "not" in m or "stall" in m:
#         return "Non-advancing (stalled wound edge, no epithelial migration)"
#     return "Advancing (wound actively healing, epithelial migration visible)"


# # ═══════════════════════════════════════════════════════════════════
# # MOISTURE-AWARE QUERY BUILDER
# # Generates a targeted retrieval query that steers towards the
# # correct dressing type based on moisture — the key safety check.
# # ═══════════════════════════════════════════════════════════════════

# def build_moisture_aware_query(
#     tissue_profile: dict,
#     infection_norm: str,
#     moisture_norm: str,
# ) -> str:
#     m = moisture_norm.lower()
#     t = tissue_profile["rule_input"].lower()

#     if "dry" in m or "desiccated" in m:
#         return (
#             f"hydrogel dressing dry wound no exudate autolytic debridement "
#             f"{tissue_profile['dominant_tissue']}"
#         )
#     elif "high exudate" in m or "copious" in m:
#         base = "high exudate wound alginate foam absorbent dressing"
#         if "infected" in infection_norm.lower() and "not" not in infection_norm.lower():
#             return base + " silver antimicrobial infection"
#         return base
#     else:
#         if "granulat" in t:
#             return "granulating wound low exudate silicone foam non-adherent protection"
#         return f"moderate exudate wound dressing {tissue_profile['dominant_tissue']}"


# # ═══════════════════════════════════════════════════════════════════
# # MULTI-QUERY EXPANSION
# # ═══════════════════════════════════════════════════════════════════

# def expand_queries(time_profile: dict, tissue_profile: dict, llm) -> list[str]:
#     """
#     Generates 3 LLM query variants + 1 structured query + 1 moisture-aware query.
#     Total = 5 queries. More diversity → better RRF fusion coverage.
#     """
#     prompt = f"""You are a wound care expert. Given this T.I.M.E. wound assessment:
# Tissue: {time_profile.get('tissue', 'unknown')}
# Infection: {time_profile.get('infection', 'unknown')}
# Moisture: {time_profile.get('moisture', 'unknown')}
# Edge: {time_profile.get('edge', 'unknown')}

# Generate exactly 3 different search queries to find relevant wound dressing guidelines.
# Use different clinical terminology in each query — vary between:
#   - dressing product names (foam, alginate, hydrogel, silver, hydrofiber)
#   - wound condition descriptions (necrotic, sloughy, granulating, infected, dry, exuding)
#   - clinical actions (debridement, absorption, moisture donation, infection control)
# Return ONLY the 3 queries, one per line, no numbering, no explanation."""

#     try:
#         response = llm.invoke([HumanMessage(content=prompt)])
#         queries  = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
#         queries  = queries[:3]
#     except Exception:
#         queries = []

#     # Structured query — always included
#     queries.append(
#         f"Wound dressing recommendation: tissue={time_profile.get('tissue')}, "
#         f"infection={time_profile.get('infection')}, "
#         f"moisture={time_profile.get('moisture')}, "
#         f"edge={time_profile.get('edge')}"
#     )

#     # Moisture-aware safety query — ensures correct absorbency class is retrieved
#     queries.append(
#         build_moisture_aware_query(
#             tissue_profile,
#             time_profile.get("infection", ""),
#             time_profile.get("moisture", ""),
#         )
#     )

#     # Rule-anchored query — ensures RAG finds evidence FOR the rule decision
#     queries.append(
#         f"{rule_engine.decide(time_profile['tissue_raw'], time_profile['infection'], time_profile['moisture'], time_profile['edge']).primary_category} "
#         f"wound dressing clinical guideline evidence"
#     )

#     return queries


# # ═══════════════════════════════════════════════════════════════════
# # HYBRID RETRIEVAL
# # ═══════════════════════════════════════════════════════════════════

# def build_hybrid_retriever(vectorstore, all_docs: list, k: int = 10):
#     dense   = vectorstore.as_retriever(search_kwargs={"k": k})
#     bm25    = BM25Retriever.from_documents(all_docs)
#     bm25.k  = k

#     # Import here to avoid circular issues at module load
#     return EnsembleRetriever(
#         retrievers=[dense, bm25],
#         weights=[0.6, 0.4],
#     )


# def reciprocal_rank_fusion(results_per_query: list[list], k: int = 60) -> list:
#     doc_scores  = {}
#     doc_objects = {}
#     for query_results in results_per_query:
#         for rank, doc in enumerate(query_results, start=1):
#             doc_id = hash(doc.page_content)
#             doc_scores[doc_id]  = doc_scores.get(doc_id, 0) + 1.0 / (k + rank)
#             doc_objects[doc_id] = doc
#     sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
#     return [doc_objects[doc_id] for doc_id, _ in sorted_docs]


# def rerank_with_moisture_boost(
#     query: str,
#     docs: list,
#     moisture_norm: str,
#     top_n: int = 5,
# ) -> tuple[list, list[float]]:
#     """
#     Cross-encoder reranker with moisture-context boost/penalty.
#     Returns (top_docs, normalised_scores_0_to_1).
#     """
#     if not docs:
#         return [], []

#     pairs  = [(query, doc.page_content) for doc in docs]
#     scores = reranker.predict(pairs).tolist()

#     m = moisture_norm.lower()
#     boosted = []
#     for score, doc in zip(scores, docs):
#         dtype = doc.metadata.get("dressing_type", "general")
#         boost = 0.0
#         if "dry" in m or "desicat" in m:
#             if dtype == "hydrogel":
#                 boost =  0.3
#             elif dtype in ("hydrofiber", "alginate"):
#                 boost = -0.3
#         elif "high exudate" in m or "copious" in m:
#             if dtype in ("alginate", "foam", "hydrofiber"):
#                 boost =  0.2
#             elif dtype == "hydrogel":
#                 boost = -0.3
#         elif "low exudate" in m or "moderate" in m:
#             if dtype in ("silicone", "foam"):
#                 boost =  0.1
#         boosted.append((score + boost, doc))

#     boosted.sort(key=lambda x: x[0], reverse=True)
#     top = boosted[:top_n]
#     raw_scores = [s for s, _ in top]

#     # Normalise scores to 0–1 using sigmoid so they read as confidence
#     def sigmoid(x):
#         return 1.0 / (1.0 + math.exp(-x))

#     norm_scores = [round(sigmoid(s), 3) for s in raw_scores]
#     top_docs    = [doc for _, doc in top]
#     return top_docs, norm_scores


# # ═══════════════════════════════════════════════════════════════════
# # DECISION FUSION
# # Compares rule engine output against RAG-retrieved evidence and
# # produces a confidence score + agreement label.
# # ═══════════════════════════════════════════════════════════════════

# def decision_fusion(
#     rule_decision,
#     rag_chunks: list,
#     reranker_scores: list[float],
#     tissue_profile: dict,
# ) -> dict:
#     notes      = []
#     confidence = 1.0

#     # ── CHECK 1: RAG corroboration ──────────────────────────────────
#     rule_terms = set(
#         w for w in rule_decision.primary_category.lower().split()
#         if len(w) > 4
#     )
#     corroborating = sum(
#         1 for doc in rag_chunks
#         if any(term in doc.page_content.lower() for term in rule_terms)
#     )
#     ratio = corroborating / len(rag_chunks) if rag_chunks else 0.0
#     if ratio >= 0.6:
#         notes.append(f"RAG strongly corroborates rule ({corroborating}/{len(rag_chunks)} chunks agree)")
#     elif ratio >= 0.3:
#         confidence -= 0.10
#         notes.append(f"RAG partially corroborates ({corroborating}/{len(rag_chunks)} chunks)")
#     else:
#         confidence -= 0.25
#         notes.append("RAG shows weak corroboration — recommend manual review")

#     # ── CHECK 2: Reranker score quality ────────────────────────────
#     if reranker_scores:
#         avg_score = sum(reranker_scores) / len(reranker_scores)
#         if avg_score < 0.35:
#             confidence -= 0.15
#             notes.append(f"Low average retrieval relevance ({avg_score:.2f}) — guidelines may not cover this case")
#         elif avg_score > 0.65:
#             notes.append(f"High retrieval relevance ({avg_score:.2f})")

#     # ── CHECK 3: Fallback rule penalty ──────────────────────────────
#     if rule_decision.rule_id == "FALLBACK":
#         confidence -= 0.30
#         notes.append("Rule engine reached fallback — RAG is primary guidance here")

#     # ── CHECK 4: Non-viable tissue load flag ────────────────────────
#     non_viable = tissue_profile.get("non_viable_pct", 0)
#     if non_viable > 75:
#         confidence -= 0.10
#         notes.append(f"High non-viable tissue load ({non_viable:.0f}%) — specialist referral may be warranted")

#     # ── CHECK 5: Healing progress positive signal ───────────────────
#     if tissue_profile.get("healing_progress", 0) > 0.7:
#         notes.append(
#             f"Good healing progress ({tissue_profile['granulation_pct']:.0f}% granulation) — positive indicator"
#         )

#     confidence = max(0.0, min(1.0, confidence))

#     if confidence >= 0.80:
#         label = "HIGH"
#     elif confidence >= 0.55:
#         label = "MEDIUM"
#     else:
#         label = "LOW"

#     return {
#         "confidence_score": round(confidence, 2),
#         "confidence_label": label,
#         "fusion_notes":     notes,
#         "rule_id":          rule_decision.rule_id,
#     }


# # ═══════════════════════════════════════════════════════════════════
# # MAIN RETRIEVAL FUNCTION
# # ═══════════════════════════════════════════════════════════════════

# def retrieve_relevant_chunks(time_profile: dict, tissue_profile: dict, llm) -> tuple[list, list[float]]:
#     """
#     Full retrieval pipeline:
#     1. Multi-query expansion (5 queries)
#     2. Hybrid semantic + BM25 search
#     3. RRF fusion across all query results
#     4. Moisture-boosted cross-encoder reranking → top 5 chunks
#     Returns (top_chunks, normalised_reranker_scores)
#     """
#     queries = expand_queries(time_profile, tissue_profile, llm)
#     print(f"[RETRIEVAL] {len(queries)} queries expanded:")
#     for q in queries:
#         print(f"  — {q}")

#     # Load all docs for BM25
#     raw       = db.get()
#     all_docs  = [
#         LC_Doc(page_content=pc, metadata=meta)
#         for pc, meta in zip(raw["documents"], raw["metadatas"])
#     ]
#     hybrid = build_hybrid_retriever(db, all_docs, k=10)

#     results_per_query = []
#     for query in queries:
#         try:
#             results = hybrid.invoke(query)
#             results_per_query.append(results)
#             print(f"  Query → {len(results)} chunks")
#         except Exception as e:
#             print(f"  Query failed: {e}")
#             results_per_query.append([])

#     fused = reciprocal_rank_fusion(results_per_query)
#     print(f"[RETRIEVAL] After RRF fusion: {len(fused)} unique chunks")

#     anchor_query = queries[-2]  # structured query is the reliable anchor
#     top_chunks, scores = rerank_with_moisture_boost(
#         anchor_query,
#         fused,
#         time_profile.get("moisture", ""),
#         top_n=5,
#     )
#     print(f"[RETRIEVAL] After reranking: {len(top_chunks)} chunks → LLM")
#     return top_chunks, scores


# # ═══════════════════════════════════════════════════════════════════
# # LLM GENERATION — rule-anchored, explanation-only mode
# # ═══════════════════════════════════════════════════════════════════

# def generate_hybrid_recommendation(
#     chunks: list,
#     assessment_text: str,
#     rule_decision,
#     fusion_result: dict,
# ) -> str:
#     try:
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#         prompt = f"""You are a Clinical Wound Care Consultant.

# The RULE ENGINE has already determined the correct dressing category.
# Your role is ONLY to explain and enrich this decision using the guideline evidence.
# You must NOT contradict or override the rule engine decision.

# {assessment_text}

# RETRIEVED GUIDELINE EVIDENCE (ranked by relevance):
# """
#         for i, chunk in enumerate(chunks, 1):
#             source       = chunk.metadata.get("source", "Unknown")
#             authority    = chunk.metadata.get("authority", "")
#             year         = chunk.metadata.get("year", "")
#             original     = json.loads(chunk.metadata.get("original_content", "{}"))
#             raw_text     = original.get("raw_text", chunk.page_content)

#             prompt += f"\n--- Guideline Source {i}: {source}"
#             if authority:
#                 prompt += f" [{authority}, {year}]"
#             prompt += f" ---\n{raw_text}\n"

#         prompt += f"""
# YOUR TASK:
# 1. Confirm the rule engine's primary dressing: {rule_decision.primary_category}
#    Give the specific product name and brand example from the guidelines.
# 2. Confirm or refine the secondary dressing if needed.
# 3. Explain HOW each T.I.M.E. factor drives this specific dressing choice.
# 4. List contraindications from BOTH the rule engine AND the guidelines.
# 5. State dressing change frequency with guideline citation.
# 6. Add any application tips relevant to this wound.
# 7. Cite which source document supports each recommendation.

# FORMAT: Use ## headers and bullet points. Be specific and clinical.
# DO NOT suggest alternatives that contradict the contraindications listed above.
# Confidence in this recommendation: {fusion_result['confidence_label']} ({fusion_result['confidence_score']:.0%})

# RECOMMENDATION:"""

#         response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
#         return response.content

#     except Exception as e:
#         return f"Clinical analysis error: {str(e)}"


# # ═══════════════════════════════════════════════════════════════════
# # ROUTES
# # ═══════════════════════════════════════════════════════════════════

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse(
#         request=request,
#         name="wound_index_v2.html",
#         context={"request": request},
#     )


# @app.post("/get_recommendation")
# async def get_recommendation(
#     necrotic_pct:    float = Form(...),
#     slough_pct:      float = Form(...),
#     granulation_pct: float = Form(...),
#     infection:       str   = Form(...),   # "Infected" or "Not Infected"
#     moisture:        str   = Form(...),   # "Low" | "Moderate" | "High"
#     edge:            str   = Form(...),   # "Advancing" | "Non Advancing"
#     notes:           str   = Form(""),
# ):
#     try:
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#         # ── STEP 1: Normalise inputs ──────────────────────────────────
#         tissue_profile   = interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct)
#         infection_norm   = normalize_infection(infection)
#         moisture_norm    = normalize_moisture(moisture)
#         edge_norm        = normalize_edge(edge)

#         print(f"\n[INPUT] Tissue: N={necrotic_pct}% S={slough_pct}% G={granulation_pct}%")
#         print(f"[INPUT] Infection: {infection} → {infection_norm}")
#         print(f"[INPUT] Moisture:  {moisture}  → {moisture_norm}")
#         print(f"[INPUT] Edge:      {edge}       → {edge_norm}")
#         print(f"[INPUT] Rule input tissue: {tissue_profile['rule_input']}")

#         # ── STEP 2: Rule engine decision ──────────────────────────────
#         rule_decision = rule_engine.decide(
#             tissue_profile["rule_input"],
#             infection_norm,
#             moisture_norm,
#             edge_norm,
#         )
#         print(f"[RULE] {rule_decision.rule_id} → {rule_decision.primary_category}")

#         # Build time_profile dict for retrieval (includes raw labels for queries)
#         time_profile = {
#             "tissue":      tissue_profile["tissue_label"],
#             "tissue_raw":  tissue_profile["rule_input"],
#             "infection":   infection_norm,
#             "moisture":    moisture_norm,
#             "edge":        edge_norm,
#             "rule_decision": rule_decision.primary_category,
#         }

#         # ── STEP 3: RAG retrieval ─────────────────────────────────────
#         top_chunks, reranker_scores = retrieve_relevant_chunks(time_profile, tissue_profile, llm)

#         # ── STEP 4: Decision fusion + confidence ──────────────────────
#         fusion = decision_fusion(rule_decision, top_chunks, reranker_scores, tissue_profile)
#         print(f"[FUSION] Confidence: {fusion['confidence_score']} ({fusion['confidence_label']})")

#         # ── STEP 5: Build assessment text for LLM ─────────────────────
#         assessment_text = f"""T.I.M.E. WOUND ASSESSMENT (from ML pipeline):
#   T (Tissue)    : {tissue_profile['tissue_label']}
#                   Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
#   I (Infection) : {infection_norm}  [raw: {infection}]
#   M (Moisture)  : {moisture_norm}  [raw: {moisture}]
#   E (Edge)      : {edge_norm}  [raw: {edge}]
# {"  Notes: " + notes.strip() if notes.strip() else ""}

# RULE ENGINE DECISION (Rule ID: {rule_decision.rule_id}):
#   Primary   : {rule_decision.primary_category} — e.g. {rule_decision.primary_example}
#   Secondary : {rule_decision.secondary_category or "Not required"}{" — " + rule_decision.secondary_example if rule_decision.secondary_example else ""}
#   Frequency : {rule_decision.change_frequency}

# CONTRAINDICATIONS (rule engine):
# {chr(10).join("  ❌ " + c for c in rule_decision.contraindications)}

# CLINICAL RATIONALE (rule engine):
#   {rule_decision.clinical_rationale}

# DECISION FUSION:
#   Confidence : {fusion['confidence_score']:.0%} ({fusion['confidence_label']})
# {chr(10).join("  • " + n for n in fusion['fusion_notes'])}"""

#         # ── STEP 6: LLM generation ────────────────────────────────────
#         result = generate_hybrid_recommendation(top_chunks, assessment_text, rule_decision, fusion)

#         # ── STEP 7: Build response ────────────────────────────────────
#         sources = list(dict.fromkeys(
#             chunk.metadata.get("source", "Unknown") for chunk in top_chunks
#         ))

#         return JSONResponse({
#             "result":             result,
#             "sources":            sources,
#             "rule_id":            fusion["rule_id"],
#             "rule_primary":       rule_decision.primary_category,
#             "confidence_score":   fusion["confidence_score"],
#             "confidence_label":   fusion["confidence_label"],
#             "fusion_notes":       fusion["fusion_notes"],
#             "tissue_breakdown": {
#                 "necrotic_pct":    tissue_profile["necrotic_pct"],
#                 "slough_pct":      tissue_profile["slough_pct"],
#                 "granulation_pct": tissue_profile["granulation_pct"],
#             },
#             "reranker_scores": reranker_scores,
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JSONResponse(
#             {"result": f"System error: {str(e)}", "sources": [], "confidence_label": "LOW"},
#             status_code=500,
#         )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)