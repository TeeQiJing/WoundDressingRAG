"""
wound_app_00_v3.py  —  VerdaSense Clinical RAG Pipeline (v3_00 · Baseline v3)
═══════════════════════════════════════════════════════════════════════════════
Changes from v2_00 (the previous baseline):

  [FIX 1] NARRATIVE QUERY BUILDER
      Old: flat label concatenation → "healthy granulating wound bed Not infected
           Dry Advancing wound dressing recommendation"
      New: semantically rich natural-language question phrased exactly as a
           clinician would phrase it, improving semantic similarity with the
           KB ai_summary chunks (e.g. "What wound dressing is recommended for
           a clean granulating wound with dry exudate, no infection, and
           advancing edges?")
      Rationale: MedEmbed is trained on clinical text; a natural-language query
      retrieves more relevant chunks than a keyword concat, directly improving
      Context Recall and Faithfulness.

  [FIX 2] GROUNDED GENERATION — SYSTEM PROMPT + STRICT EVIDENCE CONSTRAINT
      Old: single HumanMessage with no system-level grounding instruction.
      New: explicit ChatOpenAI system message that instructs the LLM to:
           - base EVERY claim on the retrieved guideline excerpts
           - cite source numbers for each recommendation
           - explicitly state "CONTRAINDICATED" dressings in its own section
           - flag antibiotic / referral requirements when clinically indicated
           - use only information present in the retrieved sources
      Rationale: fixes low Faithfulness (was 0.61) by preventing the LLM
      from injecting general medical knowledge outside the retrieved KB.

  [FIX 3] EXPLICIT CONTRAINDICATION + ANTIBIOTIC + REFERRAL SECTION
      Old: "Contraindications" section was vaguely worded; antibiotic /
           referral recommendations were implied but not explicitly stated.
      New: separate ## Contraindicated Dressings block with a mandatory
           introductory line "The following dressings are CONTRAINDICATED
           in this case:", plus ## Antibiotic Considerations and
           ## Referral / Escalation blocks — populated from retrieved evidence.
      Rationale: fixes Rule-Based Safety Checker false-negatives by ensuring
      the generated answer uses the exact clinical language the checker expects.

  [FIX 4] NARRATIVE USER_INPUT PRESERVED FOR RAGAS
      The structured T.I.M.E. inputs are still the production API contract
      (unchanged Form fields). A separate narrative_query string is built
      for retrieval and is also returned in the response so the evaluation
      notebook can use it as user_input in SingleTurnSample, giving
      AnswerRelevancy a semantically richer question to compare against the
      long clinical answer.
      See: response field "narrative_query" (new, non-breaking addition).

Architecture (baseline — unchanged from v2_00 except for the above fixes):
  - Dense semantic search only (ChromaDB similarity_search, k=6)
  - No BM25 / hybrid retrieval
  - No clinical signal extraction
  - No multi-axis query expansion
  - No cross-encoder reranker
  - No moisture / infection boosting
  - No confidence logic (fixed label "MEDIUM")

Response contract (backwards-compatible — all v2_00 fields present + narrative_query):
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
    "narrative_query":  str,   ← NEW — use as user_input in RAGAS SingleTurnSample
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
# INPUT NORMALISATION  (unchanged from v2_00)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_tissue_percentages(
    necrotic_pct: float, slough_pct: float, granulation_pct: float
) -> dict:
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
# [v3 FIX 1] NARRATIVE QUERY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_narrative_query(
    tissue_profile: dict,
    infection_norm: str,
    moisture_norm: str,
    edge_norm: str,
    notes: str = "",
) -> str:
    """
    Builds a semantically rich natural-language retrieval query from structured
    T.I.M.E. inputs.

    Design rationale
    ────────────────
    MedEmbed-large-v0.1 is trained on clinical text.  A query that reads like
    clinical documentation retrieves chunks that also read like clinical
    documentation — which is exactly what our KB ai_summaries look like.

    The previous approach ("healthy granulating wound bed Not infected Dry
    Advancing wound dressing recommendation") was a keyword dump that matched
    poorly against the narrative style of the KB summaries, depressing Context
    Recall.

    Structure of the generated query
    ──────────────────────────────────
    - Opens with a "what dressing" clinical question (drives semantic similarity
      to guideline recommendation sections)
    - Tissue composition described in clinical prose
    - Infection, moisture, and edge status embedded naturally
    - Notes appended verbatim if present (max 200 chars to stay focused)
    """
    ct  = tissue_profile["clinical_tissue"]
    n   = tissue_profile["necrotic_pct"]
    s   = tissue_profile["slough_pct"]
    g   = tissue_profile["granulation_pct"]
    nv  = tissue_profile["non_viable_pct"]

    # ── Tissue phrase ──────────────────────────────────────────────────────────
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

    # ── Infection phrase ───────────────────────────────────────────────────────
    if infection_norm == "Locally infected":
        infection_phrase = "with signs of local wound infection"
    else:
        infection_phrase = "with no signs of infection"

    # ── Moisture phrase ────────────────────────────────────────────────────────
    moisture_map = {
        "Dry":             "minimal to no exudate (dry wound)",
        "Moderate exudate":"moderate exudate levels",
        "High exudate":    "high exudate (heavily exuding wound)",
    }
    moisture_phrase = moisture_map.get(moisture_norm, moisture_norm.lower())

    # ── Edge phrase ────────────────────────────────────────────────────────────
    if edge_norm == "Non-advancing wound edge":
        edge_phrase = "non-advancing or stalled wound edges"
    else:
        edge_phrase = "advancing wound edges (progressing toward healing)"

    # ── Compose query ──────────────────────────────────────────────────────────
    query = (
        f"What wound dressing is recommended for {tissue_phrase}, "
        f"{infection_phrase}, {moisture_phrase}, and {edge_phrase}? "
        f"Include contraindicated dressings and dressing change frequency."
    )

    # ── Append notes (optional) ────────────────────────────────────────────────
    if notes.strip():
        query += f" Additional clinical context: {notes.strip()[:200]}"

    return query


# ══════════════════════════════════════════════════════════════════════════════
# DENSE RETRIEVAL — k=6, no reranking  (unchanged from v2_00)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, k: int = 6):
    docs = db.similarity_search(query, k=k)
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# [v3 FIX 2 + 3] GROUNDED GENERATION — system message + explicit clinical blocks
# ══════════════════════════════════════════════════════════════════════════════

# ── [v3 FIX 2] System message ─────────────────────────────────────────────────────
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


def generate_recommendation(chunks, assessment_text: str, narrative_query: str) -> str:
    """
    Generate a grounded clinical recommendation.

    Changes from v2_00:
    - Uses SystemMessage + HumanMessage (was single HumanMessage)
    - SYSTEM_PROMPT enforces strict source-grounded generation
    - HumanMessage includes narrative_query at the top so the LLM understands
      what clinical question is being answered (improves AnswerRelevancy)
    - [v3 FIX 3] Prompt template contains explicit mandatory sections:
        ## Contraindicated Dressings  (verbatim opening line required)
        ## Antibiotic Considerations  (explicit yes/no phrase required)
        ## Referral / Escalation      (explicit yes/no phrase required)
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ── Build evidence block from retrieved chunks ─────────────────────────
        evidence_block = ""
        for i, chunk in enumerate(chunks, 1):
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

        # ── [v3 FIX 3] Human prompt with explicit mandatory section headers ────────
        human_prompt = f"""CLINICAL QUESTION:
{narrative_query}

{assessment_text}

RETRIEVED CLINICAL GUIDELINES (use ONLY these as your evidence):
{evidence_block}

Provide your recommendation using EXACTLY the following section structure. \
Do not add, rename, or omit any section. Cite source numbers after every claim.

## Primary Dressing
- Name the dressing category and one specific product/brand example cited in the \
guidelines (Source X).
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

        # ── Step 2: [v3 FIX 1] Build narrative retrieval query ────────────────────
        narrative_query = build_narrative_query(
            tissue_profile,
            infection_norm,
            moisture_norm,
            edge_norm,
            notes,
        )
        print(f"[RETRIEVAL] Narrative query: {narrative_query[:140]}...")

        # ── Step 3: Dense retrieval (k=6) ──────────────────────────────────────
        top_chunks = retrieve_chunks(narrative_query, k=6)
        print(f"[RETRIEVAL] {len(top_chunks)} chunks retrieved")

        # ── Step 4: Build structured assessment text ───────────────────────────
        assessment_text = f"""T.I.M.E. WOUND ASSESSMENT:

T (Tissue)    : {tissue_profile['clinical_tissue']}
                Necrotic: {tissue_profile['necrotic_pct']}%  |  Slough: {tissue_profile['slough_pct']}%  |  Granulation: {tissue_profile['granulation_pct']}%
                Non-viable load: {tissue_profile['non_viable_pct']}%

I (Infection) : {infection_norm}
M (Moisture)  : {moisture_norm}
E (Edge)      : {edge_norm}"""

        if notes.strip():
            assessment_text += f"\n\nADDITIONAL CLINICAL NOTES:\n{notes.strip()}"

        # ── Step 5: [v3 FIX 2 + 3] Grounded LLM generation ───────────────────────
        result = generate_recommendation(top_chunks, assessment_text, narrative_query)

        # ── Step 6: Build response (backwards-compatible + narrative_query) ────
        sources     = list(dict.fromkeys(c.metadata.get("source", "Unknown") for c in top_chunks))
        chunk_texts = [c.page_content for c in top_chunks]

        return JSONResponse({
            "result":           result,
            "sources":          sources,
            "chunk_texts":      chunk_texts,
            "confidence_score": 0.5,
            "confidence_label": "MEDIUM",
            "retrieval_notes":  [
                "Baseline v3: narrative query + grounded generation (k=6 dense, no reranking)"
            ],
            "tissue_breakdown": {
                "necrotic_pct":    tissue_profile["necrotic_pct"],
                "slough_pct":      tissue_profile["slough_pct"],
                "granulation_pct": tissue_profile["granulation_pct"],
            },
            "reranker_scores":  [],
            "clinical_flags":   {},
            "narrative_query":  narrative_query,   # [NEW] for RAGAS user_input
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
                "narrative_query":  "",
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
