# import os
# import json
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.messages import HumanMessage
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv

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

# # ── LOAD WOUND CARE DB ────────────────────────────────────────────────────────

# def load_wound_db(persist_directory="./db_wound_care"):
#     print(f"📂 Loading Wound Care Knowledge Base from {persist_directory}...")
#     embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_model,
#         collection_metadata={"hnsw:space": "cosine"}
#     )
#     return vectorstore

# # Global DB instance
# db = load_wound_db()
# retriever = db.as_retriever(search_kwargs={"k": 5}) # Increased k for better context

# # ── CLINICAL REASONING ENGINE ────────────────────────────────────────────────

# def generate_clinical_recommendation(chunks, assessment):
#     try:
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#         # Specialized Clinical Prompt
#         prompt_text = f"""You are a Clinical Wound Care Consultant. 
# Analyze the patient's TIME assessment and provide a dressing recommendation based ONLY on the provided guideline documents.

# PATIENT ASSESSMENT (T.I.M.E):
# {assessment}

# REFERENCE GUIDELINES:
# """
#         for i, chunk in enumerate(chunks):
#             # Extract source filename from metadata if available
#             source = chunk.metadata.get("source", "Unknown Document")
#             original_data = json.loads(chunk.metadata.get("original_content", "{}"))
            
#             prompt_text += f"\n--- Source: {source} ---\n"
#             prompt_text += f"CONTENT: {original_data.get('raw_text', '')}\n"
            
#             # Include Table data for dressing specs
#             for table in original_data.get("tables_html", []):
#                 prompt_text += f"DATA TABLE: {table}\n"

#         prompt_text += """
# INSTRUCTIONS:
# 1. Identify the most appropriate primary and secondary dressing.
# 2. Explain the rationale using the 'Tissue', 'Infection', 'Moisture', or 'Edge' factors provided.
# 3. If the guidelines mention specific contraindications, list them.
# 4. Format the output with clear headers and bullet points.

# RECOMMENDATION:"""

#         message = HumanMessage(content=[{"type": "text", "text": prompt_text}])
#         response = llm.invoke([message])
#         return response.content

#     except Exception as e:
#         return f"Clinical analysis error: {str(e)}"

# # ── ROUTES ───────────────────────────────────────────────────────────────────

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     # Ensure your template file is named wound_index.html in the templates folder
#     return templates.TemplateResponse(
#         request=request, 
#         name="wound_index.html", 
#         context={"request": request}
#     )

# @app.post("/get_recommendation")
# async def get_recommendation(assessment: str = Form(...)):
#     try:
#         # 1. Retrieve clinical chunks based on the assessment text
#         chunks = retriever.invoke(assessment)

#         # 2. Generate the specific recommendation
#         result = generate_clinical_recommendation(chunks, assessment)

#         return JSONResponse({"result": result})

#     except Exception as e:
#         return JSONResponse({"result": f"System error: {str(e)}"}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import torch

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# ── EMBEDDING MODEL (same as ingestion — must match!) ─────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ── RERANKER (cross-encoder scores query-chunk relevance) ─────────
# Free model, runs locally, specifically trained for relevance ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── LOAD VECTOR STORE ─────────────────────────────────────────────
def load_wound_db(persist_directory="./db_wound_care"):
    print(f"Loading Wound Care Knowledge Base from {persist_directory}...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

db = load_wound_db()

# ── UPGRADE 1: MULTI-QUERY EXPANSION ─────────────────────────────
def expand_queries(time_profile: dict, llm) -> list[str]:
    """
    Takes a T.I.M.E. profile dict and generates 3 diverse query variants.
    This helps retrieve chunks that use different terminology for the same concept.
    e.g. 'slough' vs 'fibrinous tissue' vs 'yellow necrotic material'
    """
    prompt = f"""You are a wound care expert. Given this T.I.M.E. wound assessment:
    Tissue: {time_profile.get('tissue', 'unknown')}
    Infection: {time_profile.get('infection', 'unknown')}
    Moisture: {time_profile.get('moisture', 'unknown')}
    Edge: {time_profile.get('edge', 'unknown')}

    Generate exactly 3 different search queries to find relevant wound dressing guidelines.
    Use different clinical terminology in each query.
    Return ONLY the 3 queries, one per line, no numbering, no explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    queries = queries[:3]  # safety cap

    # Always include the original structured query too
    original = (f"Wound dressing recommendation for: tissue={time_profile.get('tissue')}, "
                f"infection={time_profile.get('infection')}, "
                f"moisture={time_profile.get('moisture')}, "
                f"edge={time_profile.get('edge')}")
    queries.append(original)
    return queries


# ── UPGRADE 2: HYBRID SEARCH (semantic + BM25) ───────────────────
def build_hybrid_retriever(vectorstore, all_docs, k=10):
    """
    Combines dense vector search (semantic) with BM25 (keyword).
    Dense search finds conceptually similar chunks.
    BM25 ensures exact clinical terms like 'Aquacel Ag' or 'hydrocolloid' are never missed.
    k=10 per retriever so we have plenty before reranking trims to top 5.
    """
    # Dense retriever — semantic similarity
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # Sparse retriever — keyword (BM25)
    # Needs all documents from your vector store
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # EnsembleRetriever merges both
    # weights=[0.6, 0.4] means semantic search is slightly more trusted
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    return hybrid_retriever


# ── UPGRADE 3: RECIPROCAL RANK FUSION ────────────────────────────
def reciprocal_rank_fusion(results_per_query: list[list], k: int = 60) -> list:
    """
    Merges ranked results from multiple queries into one unified ranking.
    RRF formula: score = sum(1 / (k + rank)) for each time a doc appears.

    Why k=60? It's the standard RRF constant that dampens the effect of
    very high ranks without ignoring low-ranked results entirely.

    A document that appears at rank 1 in two queries scores higher than
    one that appears at rank 1 in only one query — rewarding consistency.
    """
    doc_scores = {}   # doc_id → cumulative RRF score
    doc_objects = {}  # doc_id → actual Document object

    for query_results in results_per_query:
        for rank, doc in enumerate(query_results, start=1):
            # Use page content hash as unique ID (avoid storing duplicates)
            doc_id = hash(doc.page_content)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_objects[doc_id] = doc

    # Sort by cumulative RRF score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_objects[doc_id] for doc_id, _ in sorted_docs]


# ── UPGRADE 4: RERANKER ───────────────────────────────────────────
def rerank_chunks(query: str, docs: list, top_n: int = 5) -> list:
    """
    Cross-encoder reranker scores each (query, chunk) pair directly.
    Unlike bi-encoders (which embed query and doc separately),
    cross-encoders read both together — much more accurate for relevance.

    We pass top_n=5 so only the 5 most relevant chunks go to the LLM,
    keeping the prompt focused and reducing hallucination risk.
    """
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # Zip scores with docs and sort descending
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]


# ── MAIN RETRIEVAL FUNCTION ───────────────────────────────────────
def retrieve_relevant_chunks(time_profile: dict, llm) -> list:
    """
    Full upgraded retrieval:
    1. Multi-query expansion → 4 query variants
    2. Hybrid search for each query (semantic + BM25)
    3. RRF fusion across all query results
    4. Reranker trims to top 5 most relevant chunks
    """
    # Step 1: Expand queries
    queries = expand_queries(time_profile, llm)
    print(f"Expanded to {len(queries)} queries:")
    for q in queries:
        print(f"  - {q}")

    # Step 2: Hybrid search for all documents (needed for BM25)
    all_docs_raw = db.get()
    from langchain_core.documents import Document as LC_Doc
    all_docs = [
        LC_Doc(page_content=pc, metadata=meta)
        for pc, meta in zip(all_docs_raw["documents"], all_docs_raw["metadatas"])
    ]
    hybrid = build_hybrid_retriever(db, all_docs, k=10)

    # Step 3: Retrieve for each query variant
    results_per_query = []
    for query in queries:
        results = hybrid.invoke(query)
        results_per_query.append(results)
        print(f"  Query retrieved {len(results)} chunks")

    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion(results_per_query)
    print(f"After RRF fusion: {len(fused)} unique chunks")

    # Step 5: Rerank to top 5
    # Use the original structured query as the reranking anchor
    anchor_query = queries[-1]
    top_chunks = rerank_chunks(anchor_query, fused, top_n=5)
    print(f"After reranking: {len(top_chunks)} chunks passed to LLM")

    return top_chunks


# ── CLINICAL RECOMMENDATION (unchanged prompt, better context) ────
def generate_clinical_recommendation(chunks: list, assessment: str) -> str:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
        prompt_text = f"""You are a Clinical Wound Care Consultant.
        Analyze the patient's T.I.M.E. assessment below and provide a specific
        dressing recommendation based ONLY on the provided guideline documents.
        
        {assessment}
        
        REFERENCE GUIDELINES (ranked by relevance):
        """
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown Document")
            original_data = json.loads(chunk.metadata.get("original_content", "{}"))
            prompt_text += f"\n--- Source {i}: {source} ---\n"
            prompt_text += f"CONTENT: {original_data.get('raw_text', chunk.page_content)}\n"
            for table in original_data.get("tables_html", []):
                prompt_text += f"DATA TABLE: {table}\n"
 
        prompt_text += """
        INSTRUCTIONS:
        1. Identify the PRIMARY dressing (contact layer / main dressing).
        2. Identify the SECONDARY dressing (absorbent layer / cover) if needed.
        3. For each, state the product category and give one brand example if mentioned in the guidelines.
        4. Explain the rationale linking EACH T.I.M.E. factor to the dressing choice.
        5. List any contraindications mentioned in the guidelines.
        6. State the recommended dressing change frequency.
        7. Cite which source document supports each recommendation.
        8. Format using clear headers (##) and bullet points.
        
        RECOMMENDATION:"""
 
        response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt_text}])])
        return response.content
 
    except Exception as e:
        return f"Clinical analysis error: {str(e)}"

# ── ROUTES ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="wound_index.html", context={"request": request}
    )

@app.post("/get_recommendation")
async def get_recommendation(
    tissue:    str = Form(...),
    infection: str = Form(...),
    moisture:  str = Form(...),
    edge:      str = Form(...),
    notes:     str = Form(""),       # optional — empty string default
):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
        # ── 1. Build structured TIME profile ────────────────────────
        # Passed directly into retrieve_relevant_chunks() for multi-query expansion
        time_profile = {
            "tissue":    tissue,
            "infection": infection,
            "moisture":  moisture,
            "edge":      edge,
        }
 
        # ── 2. Build human-readable assessment string ────────────────
        # Passed into generate_clinical_recommendation() as the prompt context.
        # Clear structured format helps the LLM reason about each component.
        assessment_lines = [
            "T.I.M.E. WOUND ASSESSMENT:",
            f"  T (Tissue)    : {tissue}",
            f"  I (Infection) : {infection}",
            f"  M (Moisture)  : {moisture}",
            f"  E (Edge)      : {edge}",
        ]
        if notes.strip():
            assessment_lines.append(f"\nADDITIONAL CLINICAL NOTES:\n  {notes.strip()}")
 
        assessment_text = "\n".join(assessment_lines)
 
        # ── 3. Retrieve relevant guideline chunks ────────────────────
        top_chunks = retrieve_relevant_chunks(time_profile, llm)
 
        # ── 4. Extract source filenames for UI chips ─────────────────
        sources = [
            chunk.metadata.get("source", "Unknown")
            for chunk in top_chunks
        ]
 
        # ── 5. Generate clinical recommendation ──────────────────────
        result = generate_clinical_recommendation(top_chunks, assessment_text)
 
        return JSONResponse({
            "result":  result,
            "sources": sources,          # list of source PDF filenames
        })
 
    except Exception as e:
        return JSONResponse(
            {"result": f"System error: {str(e)}", "sources": []},
            status_code=500
        )
 
 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)