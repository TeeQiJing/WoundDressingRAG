"""
wound_app_00.py  —  VerdaSense Clinical RAG Pipeline (v00 · Baseline)
══════════════════════════════════════════════════════════════════════
Ablation baseline — stripped to the absolute minimum for RAGAS evaluation.

Architecture:
  - Dense semantic search only (ChromaDB similarity_search, k=6)
  - No BM25 / hybrid retrieval
  - No clinical signal extraction
  - No multi-axis query expansion  (single query built from T.I.M.E. inputs)
  - No cross-encoder reranker
  - No moisture / infection boosting
  - No confidence logic (fixed label "MEDIUM")
  - Standard generation prompt (no combination / escalation blocks)

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
from langchain_core.messages import HumanMessage

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


# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION  (kept identical to v4 for fair comparison)
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
# SINGLE QUERY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_single_query(clinical_tissue: str, infection_norm: str, moisture_norm: str, edge_norm: str, notes: str = "") -> str:
    """One flat query concatenating T.I.M.E. labels — no expansion."""
    parts = [clinical_tissue, infection_norm, moisture_norm, edge_norm, "wound dressing recommendation"]
    if notes.strip():
        parts.append(notes.strip()[:120])
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# DENSE RETRIEVAL — k=6, no reranking
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, k: int = 6):
    docs = db.similarity_search(query, k=k)
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION — standard prompt, no special blocks
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(chunks, assessment_text: str) -> str:
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
    tissue_confidence: float = Form(0.0),   # accepted but not used (keeps notebook payload compatible)
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

        # ── Step 2: Build single query ────────────────────────────────────────
        query = build_single_query(
            tissue_profile["clinical_tissue"],
            infection_norm,
            moisture_norm,
            edge_norm,
            notes,
        )
        print(f"[RETRIEVAL] Query: {query[:120]}...")

        # ── Step 3: Dense retrieval (k=6) ─────────────────────────────────────
        top_chunks = retrieve_chunks(query, k=6)
        print(f"[RETRIEVAL] {len(top_chunks)} chunks retrieved")

        # ── Step 4: Build assessment text ─────────────────────────────────────
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

        # ── Step 5: LLM generation ────────────────────────────────────────────
        result = generate_recommendation(top_chunks, assessment_text)

        # ── Step 6: Build response (matches notebook contract) ────────────────
        sources     = list(dict.fromkeys(c.metadata.get("source", "Unknown") for c in top_chunks))
        chunk_texts = [c.page_content for c in top_chunks]

        return JSONResponse({
            "result":           result,
            "sources":          sources,
            "chunk_texts":      chunk_texts,
            "confidence_score": 0.5,         # fixed — no scoring logic in baseline
            "confidence_label": "MEDIUM",    # fixed label
            "retrieval_notes":  ["Baseline: dense retrieval only (k=6), no reranking"],
            "tissue_breakdown": {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores":  [],          # no reranker in baseline
            "clinical_flags":   {},          # no clinical signal extraction in baseline
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
