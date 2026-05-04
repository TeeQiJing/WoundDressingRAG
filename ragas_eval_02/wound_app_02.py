"""
wound_app_02.py  —  VerdaSense Clinical RAG Pipeline (v02 · + Cross-Encoder Reranker)
═══════════════════════════════════════════════════════════════════════════════════════
Changes from v01 (one change only):
  [CHANGE] After hybrid EnsembleRetriever fetches k=10 candidates, pass them
           through a CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2) and
           re-rank by relevance score. Keep top-6 after reranking.
           No moisture/infection boosting — that is v03's addition.

Everything else is identical to v01:
  - Single flat query built from T.I.M.E. labels  (no expansion)
  - Hybrid dense (0.6) + BM25 (0.4), each k=10, RRF fusion → top-6 pool
  - No clinical signal extraction
  - No moisture / infection boosting
  - No confidence logic  (fixed "MEDIUM" / 0.5)
  - Identical generate_recommendation() function including the structured prompt
  - Identical assessment_text and evidence_block construction

Ablation hypothesis:
  The reranker discards noisy BM25 keyword matches before generation, which
  should recover Faithfulness and Answer Relevancy (which dropped in v01 due
  to diluted context) while keeping the Recall gain from hybrid retrieval.

Plain reranker chosen over moisture-boost reranker deliberately:
  Boosting is an orthogonal signal (clinical domain heuristic).
  Isolating the pure cross-encoder effect here keeps the ablation clean.
  Boost variant can be tested as a sub-branch of v03 if desired.

Response contract (unchanged — matches wound_ragas_ablation.ipynb):
  {
    "result":           str,
    "sources":          list[str],
    "chunk_texts":      list[str],
    "confidence_score": float,
    "confidence_label": str,
    "retrieval_notes":  list[str],
    "tissue_breakdown": dict,
    "reranker_scores":  list,
    "clinical_flags":   dict,
  }
"""

import os
import json
import torch
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LC_Doc
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# ── Cross-encoder reranker (NEW in v02) ───────────────────────────────────────
from sentence_transformers import CrossEncoder

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

# ── Load vector store ─────────────────────────────────────────────────────────
def load_wound_db(
    persist_directory: str = r"C:\Users\GIGA\OneDrive - Universiti Malaya\Documents\rag-for-beginners\db_wound_care_v2",
):
    print(f"Loading Wound Care KB from {persist_directory}...")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

db = load_wound_db()

# ── Cross-encoder — loaded once at startup ────────────────────────────────────
print("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder ready.")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION  (identical to v00/v01)
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
    n  = necrotic_pct    / total * 100
    s  = slough_pct      / total * 100
    g  = granulation_pct / total * 100
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
# SINGLE QUERY BUILDER  (identical to v00/v01 — no expansion)
# ══════════════════════════════════════════════════════════════════════════════

def build_single_query(
    clinical_tissue: str,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str = "",
) -> str:
    """One flat query concatenating T.I.M.E. labels — no expansion."""
    parts = [clinical_tissue, infection_norm, moisture_norm, edge_norm, "wound dressing recommendation"]
    if notes.strip():
        parts.append(notes.strip()[:120])
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL  (identical to v01)
# ══════════════════════════════════════════════════════════════════════════════

def build_hybrid_retriever(vectorstore, all_docs: list, k: int = 10):
    """
    EnsembleRetriever combining:
      - Dense: ChromaDB cosine similarity, weight 0.6
      - Sparse: BM25 over all KB documents, weight 0.4
    Both fetch k candidates; LangChain RRF fusion produces the merged list.
    """
    dense_retriever  = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever   = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


# ══════════════════════════════════════════════════════════════════════════════
# [v02 CHANGE] CROSS-ENCODER RERANKING
# ══════════════════════════════════════════════════════════════════════════════

def rerank_with_cross_encoder(query: str, docs: list, top_k: int = 6) -> tuple:
    """
    Re-rank hybrid candidates using CrossEncoder relevance scores.
    Returns (reranked_docs[:top_k], scores_list).

    Pure cross-encoder scoring — no moisture/infection boosting.
    Boost will be layered on separately in v03 to keep the ablation clean.
    """
    if not docs:
        return docs, []

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs).tolist()

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    reranked   = [doc   for _, doc  in scored_docs[:top_k]]
    top_scores = [round(float(s), 4) for s, _ in scored_docs[:top_k]]

    print(f"  [Reranker] {len(docs)} candidates → top {len(reranked)} after cross-encoder")
    for i, (s, doc) in enumerate(scored_docs[:top_k]):
        src = doc.metadata.get("source", "?")
        print(f"    [{i+1}] score={s:.3f}  src={src}")

    return reranked, top_scores


def retrieve_chunks(query: str, top_n: int = 6) -> tuple:
    """
    Step 1: Hybrid retrieval (dense + BM25, k=10 each → RRF fusion)
    Step 2: Cross-encoder reranking → top_n
    Returns (reranked_docs, reranker_scores).
    """
    raw      = db.get()
    all_docs = [
        LC_Doc(page_content=pc, metadata=meta)
        for pc, meta in zip(raw["documents"], raw["metadatas"])
    ]

    hybrid   = build_hybrid_retriever(db, all_docs, k=10)
    raw_docs = hybrid.invoke(query)
    print(f"  [Hybrid]   Retrieved {len(raw_docs)} candidates (dense 0.6 + BM25 0.4)")

    reranked, scores = rerank_with_cross_encoder(query, raw_docs, top_k=top_n)
    return reranked, scores


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION  (identical to v00/v01 — prompt, evidence_block, and all structure
#              are exactly the same; do NOT change this between ablation steps)
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(chunks: list, assessment_text: str) -> str:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        evidence_block = ""
        for i, chunk in enumerate(chunks, 1):
            source    = chunk.metadata.get("source",    "Unknown")
            authority = chunk.metadata.get("authority", "")
            year      = chunk.metadata.get("year",      "")
            try:
                original = json.loads(chunk.metadata.get("original_content", "{}"))
                raw_text = original.get("raw_text", chunk.page_content)
            except Exception:
                raw_text = chunk.page_content

            evidence_block += f"\n--- Guideline Source {i}: {source}"
            if authority:
                evidence_block += f" [{authority}"
                if year:
                    evidence_block += f", {year}"
                evidence_block += "]"
            evidence_block += f" ---\n{raw_text}\n"

        prompt = f"""You are a Clinical Wound Care Consultant providing an evidence-based dressing recommendation.

{assessment_text}

RETRIEVED CLINICAL GUIDELINES (ranked by relevance):
{evidence_block}

Provide your recommendation using the following structure:

## Primary Dressing
- Dressing category and one specific brand example from the guidelines
- Why this addresses the dominant clinical need

## Secondary Dressing
- If a second dressing is needed, state it; otherwise explain why a single dressing suffices

## Rationale by T.I.M.E. Factor
- T (Tissue): link tissue composition to dressing mechanism
- I (Infection): address infection status and antimicrobial choice if needed
- M (Moisture): explain moisture level and absorbency/donation requirement
- E (Edge): address wound edge status

## Contraindications
- Dressing types NOT to use and why

## Dressing Change Frequency
- Recommended frequency with guideline citation

## Application Tips
- Practical tips from the guidelines

## Clinical Notes
- Any patient-specific considerations from the notes field

Base every recommendation on the retrieved guideline evidence. Cite source numbers."""

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
        # ── Step 1: Normalise inputs ──────────────────────────────────────────
        tissue_profile = interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct)
        infection_norm = normalize_infection(infection)
        moisture_norm  = normalize_moisture(moisture)
        edge_norm      = normalize_edge(edge)

        print(f"\n[INPUT] N={necrotic_pct}% S={slough_pct}% G={granulation_pct}%")
        print(f"[INPUT] Infection : {infection_norm}")
        print(f"[INPUT] Moisture  : {moisture_norm}")
        print(f"[INPUT] Edge      : {edge_norm}")
        print(f"[INPUT] Notes     : {notes[:80]}{'...' if len(notes) > 80 else ''}")

        # ── Step 2: Build single query (identical to v00/v01) ─────────────────
        query = build_single_query(
            tissue_profile["clinical_tissue"],
            infection_norm,
            moisture_norm,
            edge_norm,
            notes,
        )
        print(f"[RETRIEVAL] Query : {query[:120]}...")

        # ── Step 3: Hybrid retrieval + cross-encoder reranking (v02 change) ───
        top_chunks, reranker_scores = retrieve_chunks(query, top_n=6)
        print(f"[RETRIEVAL] {len(top_chunks)} chunks after hybrid + cross-encoder reranking")

        # ── Step 4: Build assessment text (identical to v00/v01) ──────────────
        assessment_text = f"""T.I.M.E. WOUND ASSESSMENT:

  T (Tissue)    : {tissue_profile['clinical_tissue']}
                  Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
                  Non-viable load: {tissue_profile['non_viable_pct']}%

  I (Infection) : {infection_norm}
  M (Moisture)  : {moisture_norm}
  E (Edge)      : {edge_norm}
"""
        if notes.strip():
            assessment_text += f"\nADDITIONAL CLINICAL NOTES:\n  {notes.strip()}\n"

        # ── Step 5: LLM generation (identical to v00/v01) ─────────────────────
        result = generate_recommendation(top_chunks, assessment_text)

        # ── Step 6: Build response (matches notebook contract) ────────────────
        sources     = list(dict.fromkeys(c.metadata.get("source", "Unknown") for c in top_chunks))
        chunk_texts = [c.page_content for c in top_chunks]

        return JSONResponse({
            "result":           result,
            "sources":          sources,
            "chunk_texts":      chunk_texts,
            "confidence_score": 0.5,
            "confidence_label": "MEDIUM",
            "retrieval_notes":  [
                "v02: hybrid (dense 0.6 + BM25 0.4, k=10 each) → cross-encoder reranked → top-6",
                "cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2",
                f"top-6 reranker scores: {reranker_scores}",
            ],
            "tissue_breakdown": {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores":  reranker_scores,
            "clinical_flags":   {},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "result":           f"System error: {str(e)}",
                "sources":          [],
                "chunk_texts":      [],
                "confidence_label": "LOW",
                "confidence_score": 0.0,
                "retrieval_notes":  ["System error during retrieval"],
                "tissue_breakdown": {},
                "reranker_scores":  [],
                "clinical_flags":   {},
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)