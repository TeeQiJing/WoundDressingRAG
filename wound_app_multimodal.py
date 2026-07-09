"""
wound_app_multimodal.py — VerdaSense Clinical RAG Pipeline (MULTIMODAL · FYP2)
===================================================================================
Path : rag-for-beginners/wound_app_multimodal.py

Sits beside wound_app_unimodal.py. Everything the unimodal app does is preserved
(rule-based wound-type classifier, R1-C multi-axis dense retrieval, BGE embeddings,
the senior CV pipeline via the HF segmenter Space, manual I/M/E + notes). FYP2 adds
the multimodal layer and a patient-facing output contract.

WHAT'S NEW vs UNIMODAL
──────────────────────
  1. KB v4 → v5 (BGE, R4-B winner)   db_wound_care_v5_bge / wound_care_v5_bge  (160 chunks, 9 sources)
  2. VLM CAPTION at the GENERATION stage (R5: caption helps generation, not retrieval).
     A vision model looks at the raw wound photo and emits a structured clinical caption
     that is passed to the generation LLM alongside retrieved chunks + the T.I.M.E. payload.
  3. ETIOLOGY classification by the VLM (DFU / VLU / pressure / arterial / burn / skin-tear / …)
     — no new CV model trained; surfaced in the UI.
  4. WOUND DEPTH by the VLM (superficial / cavity), combined with an optional patient self-report.
  5. PATIENT-FRIENDLY output schema (v5 testset format) with [S#] citations — one generation,
     two render modes (Dev = evidence + citations + caption internals; Prod = product gallery).
  6. STATIC PRODUCT GALLERY (DyaMed) — placeholder images, grounded in the wound type.
  7. multimodal_enabled flag → live A/B of unimodal vs multimodal in the same UI.

Retrieval, the rule-based classifier, confidence scoring, token/cost accounting, and the
CV pipeline are all UNCHANGED from the unimodal app.

RESPONSE CONTRACT — POST /get_recommendation  (superset of unimodal)
────────────────────────────────────────────
  ...all unimodal fields..., plus:
    multimodal_enabled     : bool
    vlm                    : { model, label, caption, time_crossvalidation,
                               etiology:{label,confidence,rationale},
                               depth:{label,rationale}, periwound,
                               anatomical_location, urgency_flags[], dressing_implications,
                               input_tokens, output_tokens, cost_usd, latency_ms, error }
    wound_depth_final      : "superficial" | "cavity"
    dfu_flag               : bool
    demographics           : { diabetes, depth_self_report }
    product_gallery        : list[{ name, brand, dressing_class, moh_category, availability, note, image }]
    view_default           : "dev" | "prod"
"""

import os, re, json, uuid, time, base64, sys, torch
from dotenv import load_dotenv

# Force UTF-8 console output so the Unicode in our log lines (─, →, ⚠, emoji) never
# crashes the request handler on a cp1252 Windows console. Does not affect JSON responses.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL  (R4-B: BGE Large En V1.5 — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ══════════════════════════════════════════════════════════════════════════════
# CHROMA DB  (v4 → v5 BGE)
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = os.environ.get(
    "WOUND_DB_PATH_V5",
    r"C:\Users\GIGA\OneDrive - Universiti Malaya\Documents\rag-for-beginners\db_wound_care_v5_bge",
)
DB_COLLECTION = "wound_care_v5_bge"


def load_wound_db(persist_directory: str = DB_PATH):
    print(f"[DB] Loading Wound Care KB v5 from {persist_directory}...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        collection_name=DB_COLLECTION,
    )
    print(f"[DB] Loaded {db._collection.count()} chunks (v5)")
    return db


db = load_wound_db()

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="VerdaSense Multimodal RAG (FYP2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRIES
# ══════════════════════════════════════════════════════════════════════════════

# Generation LLMs (unchanged from unimodal)
MODEL_REGISTRY = {
    "gpt-4o-mini": {
        "label": "GPT-4o mini (OpenAI)",
        "input_price": 0.150, "output_price": 0.600, "no_think": False,
    },
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash (Google)",
        "input_price": 0.300, "output_price": 2.500, "no_think": False,
    },
    "qwen/qwen3.5-35b-a3b": {
        "label": "Qwen3.5-35B-A3B (OpenRouter)",
        "input_price": 0.140, "output_price": 1.000, "no_think": True,
    },
}
VALID_MODELS = set(MODEL_REGISTRY.keys())

# Vision LLMs for the caption stage (must be vision-capable — Qwen text model excluded)
VLM_REGISTRY = {
    "gpt-4o-mini": {
        "label": "GPT-4o mini Vision (OpenAI)",
        "input_price": 0.150, "output_price": 0.600,
    },
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash Vision (Google)",
        "input_price": 0.300, "output_price": 2.500,
    },
}
VALID_VLMS = set(VLM_REGISTRY.keys())


# ── LLM factory (unchanged) ─────────────────────────────────────────────────────

def make_llm(model_key: str):
    if model_key == "gpt-4o-mini":
        return ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif model_key == "gemini-2.5-flash":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
    elif model_key == "qwen/qwen3.5-35b-a3b":
        return ChatOpenAI(
            model="qwen/qwen3.5-35b-a3b", temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            extra_body={
                "reasoning": {"effort": "none"},
                "include_reasoning": False,
            },
        )
    else:
        raise ValueError(f"Unknown model key: {model_key!r}")


def _extract_tokens(response) -> tuple[int, int]:
    try:
        meta = response.usage_metadata or {}
        inp = int(meta.get("input_tokens",  meta.get("prompt_tokens",     0)) or 0)
        out = int(meta.get("output_tokens", meta.get("completion_tokens", 0)) or 0)
        return inp, out
    except Exception:
        return 0, max(1, len(getattr(response, "content", "") or "") // 4)


def _compute_cost(registry: dict, model_key: str, input_tokens: int, output_tokens: int) -> float:
    cfg = registry.get(model_key, {"input_price": 0, "output_price": 0})
    return (
        (input_tokens  / 1_000_000) * cfg["input_price"] +
        (output_tokens / 1_000_000) * cfg["output_price"]
    )


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ══════════════════════════════════════════════════════════════════════════════
# ETIOLOGY MAPPINGS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

ETIOLOGY_TO_WOUND_TYPE = {
    "burn": "burn", "skin_tear": "skin_tear",
    "diabetic_foot": "dfu", "generic": "general",
}
ETIOLOGY_TO_WOUND_CATEGORY = {
    "burn": "wound_specific", "skin_tear": "assessment",
    "diabetic_foot": "wound_specific", "generic": "algorithm",
}

# ══════════════════════════════════════════════════════════════════════════════
# INPUT NORMALISATION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct):
    total = necrotic_pct + slough_pct + granulation_pct
    if total == 0:
        return {
            "tissue_label": "insufficient tissue data", "clinical_tissue": "unknown",
            "necrotic_pct": 0.0, "slough_pct": 0.0,
            "granulation_pct": 0.0, "non_viable_pct": 0.0,
        }
    n = necrotic_pct / total * 100
    s = slough_pct / total * 100
    g = granulation_pct / total * 100
    nv = n + s
    if   n >= 50:              clinical_tissue = "predominantly necrotic wound"
    elif s >= 50:              clinical_tissue = "sloughy fibrinous wound bed"
    elif g >= 70:              clinical_tissue = "healthy granulating wound bed"
    elif n >= 25 and s >= 25:  clinical_tissue = "mixed necrotic and slough tissue"
    else:                      clinical_tissue = "mixed wound bed tissue"
    return {
        "tissue_label": clinical_tissue, "clinical_tissue": clinical_tissue,
        "necrotic_pct": round(n, 1), "slough_pct": round(s, 1),
        "granulation_pct": round(g, 1), "non_viable_pct": round(nv, 1),
    }


def normalize_infection(label: str) -> str:
    m = label.lower().strip()
    return "Not infected" if ("not" in m or "no" in m) else "Locally infected"


def normalize_moisture(label: str) -> str:
    m = label.lower().strip()
    if m in ("high", "high exudate"):      return "High exudate"
    if m in ("low", "low exudate", "dry"): return "Dry"
    return "Moderate exudate"


def normalize_edge(label: str) -> str:
    m = label.lower().strip()
    return ("Non-advancing wound edge"
            if ("non" in m or "not" in m or "stall" in m)
            else "Advancing wound edge")


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL PRE-CLASSIFIER  (rule-based — unchanged)
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

_ANTIBIOTIC_TRIGGERS = [
    "purulent", "pus", "fever", "sepsis", "cellulitis",
    "infected wound", "wound infection", "signs of infection",
    "increasing pain", "worsening pain", "more painful", "painful over",
    "warmth", "warm to touch", "warm around",
    "redness", "erythema", "perilesional", "peri-wound",
    "swelling", "oedema", "edema",
    "increased exudate", "more exudate", "exudate increasing",
    "malodour", "malodorous", "offensive odour", "offensive smell",
    "non-healing", "nonhealing", "not healing", "failing to heal",
    "deteriorating", "getting worse", "worsening wound",
]
_DIABETIC_TRIGGERS = ["diabetic", "diabetes", "neuropath", "peripheral arterial"]
_REFERRAL_TRIGGERS = ["hospital", "refer", "specialist", "chronic",
                      "burns", "burn", "deep", "full thickness"]


def classify_wound(tissue_profile, infection_norm, moisture_norm, notes=""):
    nv = tissue_profile["non_viable_pct"]
    infected = (infection_norm == "Locally infected")
    high_exu = (moisture_norm == "High exudate")
    notes_lc = notes.lower()

    is_diabetic  = any(t in notes_lc for t in _DIABETIC_TRIGGERS)
    is_burn      = any(t in notes_lc for t in ["burn", "burns", "scald"])
    is_skin_tear = any(t in notes_lc for t in [
        "skin tear", "skin-tear", "skintear", "fragile skin", "papery skin",
        "elderly skin", "tear", "flap"])

    if   is_diabetic:  etiology = "diabetic_foot"
    elif is_burn:      etiology = "burn"
    elif is_skin_tear: etiology = "skin_tear"
    else:              etiology = "generic"

    nv_high = nv >= 25
    if   not nv_high and not infected: wound_type, referral, antibiotic = (2 if high_exu else 1), False, False
    elif not nv_high and infected:     wound_type, referral, antibiotic = (4 if high_exu else 3), False, True
    elif nv_high     and not infected: wound_type, referral, antibiotic = (6 if high_exu else 5), high_exu, False
    else:                              wound_type, referral, antibiotic = (8 if high_exu else 7), True, True

    escalation_reason     = None
    subclinical_infection = False

    if is_diabetic and not referral:
        referral = True
        escalation_reason = "Notes indicate diabetic patient — referral escalated"
        if nv >= 25 and not infected:
            wound_type = 6 if high_exu else 5
            escalation_reason += " | NV >= 25% in diabetic → Type 5/6"

    if not referral:
        for trigger in _REFERRAL_TRIGGERS:
            if trigger in notes_lc:
                referral = True
                escalation_reason = f"Notes contain '{trigger}' — referral escalated"
                break

    if not antibiotic:
        matched = [t for t in _ANTIBIOTIC_TRIGGERS if t in notes_lc]
        if matched:
            antibiotic = True
            subclinical_infection = True
            ts = ", ".join(f"'{t}'" for t in matched[:3])
            note_ = f"Notes subclinical infection signals ({ts}) — antibiotic escalated"
            escalation_reason = (escalation_reason + " | " + note_) if escalation_reason else note_

    classifier_notes = (
        f"NV={nv:.1f}%, infected={infected}, high_exudate={high_exu}, etiology={etiology} "
        f"→ Wound Type {wound_type}, referral={referral}, abx={antibiotic}"
    )
    if subclinical_infection: classifier_notes += " | SUBCLINICAL INFECTION RISK"
    if escalation_reason:     classifier_notes += f" | ESCALATION: {escalation_reason}"

    return {
        "wound_type": wound_type,
        "referral_required": referral,
        "antibiotic_required": antibiotic,
        "subclinical_infection": subclinical_infection,
        "etiology": etiology,
        "escalation_reason": escalation_reason,
        "algorithm_query": WOUND_TYPE_QUERY_PHRASES[wound_type],
        "classifier_notes": classifier_notes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE QUERY BUILDER  (R1-C — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def build_narrative_query(tissue_profile, infection_norm, moisture_norm, edge_norm, notes=""):
    g, n, s, nv = (tissue_profile[k] for k in
                   ("granulation_pct", "necrotic_pct", "slough_pct", "non_viable_pct"))

    if   g >= 70: tissue_phrase = f"a clean granulating wound bed ({g:.0f}% granulation tissue)"
    elif n >= 50: tissue_phrase = f"a predominantly necrotic wound ({n:.0f}% necrosis, {s:.0f}% slough, {nv:.0f}% non-viable)"
    elif s >= 50: tissue_phrase = f"a heavily sloughy wound ({s:.0f}% yellow slough, {n:.0f}% necrosis, {nv:.0f}% non-viable)"
    else:         tissue_phrase = f"a mixed wound bed ({n:.0f}% necrosis, {s:.0f}% slough, {g:.0f}% granulation, {nv:.0f}% non-viable)"

    infection_phrase = ("with signs of local wound infection"
                        if infection_norm == "Locally infected"
                        else "with no signs of infection")

    moisture_map = {
        "High exudate":     "producing high levels of exudate requiring high-absorbency dressings",
        "Dry":              "presenting with dry to minimal exudate requiring moisture-donating dressings",
        "Moderate exudate": "producing moderate exudate requiring balanced moisture management",
    }
    moisture_phrase = moisture_map.get(moisture_norm, f"with {moisture_norm.lower()}")

    edge_phrase = ("and a non-advancing, stalled wound edge suggesting delayed healing"
                   if "Non-advancing" in edge_norm
                   else "and an advancing wound edge indicating active healing")

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
# CONFIDENCE + DENSE RETRIEVAL  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(narrative_query: str, k: int = 3) -> tuple[float, str]:
    try:
        results = db.similarity_search_with_score(narrative_query, k=k)
        if not results:
            return 0.0, "LOW"
        distances = [score for _, score in results]
        avg_dist = sum(distances) / len(distances)
        confidence = max(0.0, round(1.0 - (avg_dist / 2.0), 4))
        label = "HIGH" if confidence >= 0.75 else ("MEDIUM" if confidence >= 0.50 else "LOW")
        return confidence, label
    except Exception as e:
        print(f"[Confidence] error: {e}")
        return 0.0, "LOW"


def _dense_search(query: str, k: int, where: dict = None) -> list:
    try:
        if where:
            docs = db.similarity_search(query, k=k, filter=where)
            if docs:
                return docs
            print(f"  [Retrieval] filter {where} returned 0 — unfiltered fallback")
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"  [Retrieval] dense search error: {e}")
        return []


def _build_dressing_mechanism_query(tissue_profile, infection_norm, moisture_norm) -> str:
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
    narrative_query, tissue_profile, infection_norm,
    moisture_norm, classifier, notes="", top_n=6,
):
    retrieval_notes = []
    seen_ids = set()
    merged = []

    algo_query = classifier["algorithm_query"]
    wt = classifier["wound_type"]
    etiology = classifier.get("etiology", "generic")

    # ── Sub-query A: pinned wound-type algorithm chunk ───────────────────────────
    algo_docs, filter_strategy = [], "none"
    docs = _dense_search(algo_query, k=2, where={"wound_type": {"$eq": str(wt)}})
    if docs:
        algo_docs, filter_strategy = docs, f"wound_type={wt} (dense pinned)"
    if not algo_docs and etiology != "generic":
        wt_from_etiology = ETIOLOGY_TO_WOUND_TYPE.get(etiology, "general")
        docs = _dense_search(algo_query, k=2, where={"wound_type": {"$eq": wt_from_etiology}})
        if docs:
            algo_docs, filter_strategy = docs, f"wound_type={wt_from_etiology} (etiology={etiology})"
    if not algo_docs:
        wc_from_etiology = ETIOLOGY_TO_WOUND_CATEGORY.get(etiology, "algorithm")
        docs = _dense_search(algo_query, k=2, where={"wound_category": {"$eq": wc_from_etiology}})
        if docs:
            algo_docs, filter_strategy = docs, f"wound_category={wc_from_etiology}"
    if not algo_docs:
        algo_docs, filter_strategy = _dense_search(algo_query, k=2), "unfiltered dense (all filters failed)"

    retrieval_notes.append(
        f"Sub-query A: wound type {wt} | strategy={filter_strategy} | {len(algo_docs)} chunks")
    for doc in algo_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id); merged.append(doc)

    # ── Sub-query B: dressing mechanism ──────────────────────────────────────────
    mech_query = _build_dressing_mechanism_query(tissue_profile, infection_norm, moisture_norm)
    mech_docs = _dense_search(mech_query, k=3)
    added_b = 0
    for doc in mech_docs:
        doc_id = doc.page_content[:80]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id); merged.append(doc); added_b += 1
    retrieval_notes.append(f"Sub-query B: {added_b} dressing mechanism chunks (dense)")

    # ── Sub-query C: patient notes or narrative fill ─────────────────────────────
    if notes.strip():
        notes_query = notes.strip()[:300] + " wound dressing recommendation"
        notes_docs = _dense_search(notes_query, k=2)
        added_c = 0
        for doc in notes_docs:
            doc_id = doc.page_content[:80]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id); merged.append(doc); added_c += 1
        retrieval_notes.append(f"Sub-query C: {added_c} notes-context chunks (dense)")
    else:
        remaining = top_n - len(merged)
        if remaining > 0:
            fill_docs = _dense_search(narrative_query, k=remaining + 2)
            added_fill = 0
            for doc in fill_docs:
                doc_id = doc.page_content[:80]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id); merged.append(doc); added_fill += 1
                    if len(merged) >= top_n: break
            retrieval_notes.append(f"Sub-query fill: {added_fill} narrative chunks (dense)")

    final = merged[:top_n]
    retrieval_notes.append(
        f"Dense multi-axis total: {len(final)} chunks "
        f"(wound_type={wt}, etiology={etiology}, "
        f"referral={classifier['referral_required']}, abx={classifier['antibiotic_required']})"
    )
    return final, retrieval_notes


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE ORDERING + EVIDENCE LIST  (S1 = binding algorithm chunk, then the rest)
# ══════════════════════════════════════════════════════════════════════════════

def _find_algorithm_chunk(chunks, wound_type):
    keyword, algo_chunk, other_chunks = f"Wound Type {wound_type}", None, []
    for chunk in chunks:
        content = chunk.page_content + chunk.metadata.get("raw_text", "")
        if keyword in content and algo_chunk is None:
            algo_chunk = chunk
        else:
            other_chunks.append(chunk)
    return algo_chunk, other_chunks


def order_sources(chunks, wound_type):
    """Return chunks ordered so that the binding algorithm chunk is S1, then the rest.
    The same ordering is used for the prompt's [S#] labels and the UI evidence list."""
    algo_chunk, other_chunks = _find_algorithm_chunk(chunks, wound_type)
    ordered = ([algo_chunk] if algo_chunk else []) + other_chunks
    return ordered, (algo_chunk is not None)


def build_evidence_list(ordered_chunks: list) -> list:
    evidence = []
    for i, chunk in enumerate(ordered_chunks, start=1):
        meta = chunk.metadata
        raw_text = meta.get("raw_text", chunk.page_content)
        evidence.append({
            "index": i,
            "source": meta.get("source", "Unknown"),
            "authority": meta.get("authority", ""),
            "year": str(meta.get("year", "")) if meta.get("year") else "",
            "raw_text": raw_text,
            "wound_type": meta.get("wound_type", None),
            "wound_category": meta.get("wound_category", None),
        })
    return evidence


# ══════════════════════════════════════════════════════════════════════════════
# VLM CAPTIONER  [FYP2 CORE]  — caption + etiology + depth from the raw photo
# ══════════════════════════════════════════════════════════════════════════════

# BLIND VLM (G4-P winner). The VLM is deliberately NOT shown the upstream CV labels: G4-P proved that
# showing them makes the model echo/anchor to them (0% discrepancy detection), defeating cross-validation.
# It assesses the image independently; the discrepancy vs the CV labels is computed downstream (Step 7b).
VLM_SYSTEM_PROMPT = """You are an INDEPENDENT clinical wound-assessment assistant examining a wound \
photograph. You are NOT given the upstream CV pipeline's labels — assess the wound purely from what you \
SEE, in the language of wound-care documentation. Be objective: do NOT assume the wound is healthy. If \
you see signs of infection (perilesional erythema, purulence/pus, spreading redness, heavy slough, \
odour cues) report "Infected". Estimate tissue proportions, infection, moisture and depth from the image \
alone. Never invent detail you cannot see; if the image is unclear, say so.

Return ONLY a single JSON object (no markdown, no commentary) with EXACTLY these keys:
{
  "caption": "<150-300 word structured visual clinical assessment>",
  "infection": "<Infected | Not infected | Undetermined — your INDEPENDENT visual read>",
  "moisture": "<Dry | Moderate | High | Undetermined>",
  "tissue": {"necrotic_pct": <int>, "slough_pct": <int>, "granulation_pct": <int>},
  "etiology": {"label": "<diabetic_foot_ulcer | venous_leg_ulcer | arterial_ulcer | pressure_injury | burn | skin_tear | surgical | traumatic | generic | undetermined>", "confidence": "<low|moderate|high>", "rationale": "<short visual reasoning>"},
  "depth": {"label": "<superficial | cavity | undetermined>", "rationale": "<short visual reasoning>"},
  "periwound": "<maceration / erythema / fragility / healthy / unclear>",
  "anatomical_location": "<best guess of body location, or 'unclear'>",
  "urgency_flags": ["<short visual red-flags, e.g. 'spreading erythema'; empty list if none>"],
  "dressing_implications": "<dressing PROPERTIES the wound visually appears to need: absorption, antimicrobial, moisture donation, cavity filling, periwound protection>",
  "note": "<one line: your key visual reasoning>"
}"""


def _vlm_human_prompt(time_payload_text: str, demographics_text: str) -> str:
    # BLIND: the CV T.I.M.E. labels (time_payload_text) are deliberately IGNORED here — see G4-P.
    return f"""Assess this wound photograph INDEPENDENTLY and objectively.

Patient context: {demographics_text}

Estimate the tissue proportions, infection status, moisture level and depth PURELY from the image, \
infer the most likely etiology, and state the dressing properties the wound appears to need. \
Return the JSON object described in the system message."""


def _parse_vlm_json(text: str) -> dict:
    """Robustly pull the JSON object out of a VLM response."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # grab the outermost {...}
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except Exception:
        return {}


def _empty_vlm_result(error: str = "") -> dict:
    return {
        "caption": "", "time_crossvalidation": "",
        "infection": "Undetermined", "moisture": "Undetermined", "tissue": {},
        "etiology": {"label": "undetermined", "confidence": "low", "rationale": ""},
        "depth": {"label": "undetermined", "rationale": ""},
        "periwound": "", "anatomical_location": "", "urgency_flags": [],
        "dressing_implications": "",
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        "latency_ms": 0.0, "error": error,
    }


def generate_vlm_caption(image_b64: str, mime: str, time_payload_text: str,
                         demographics_text: str, vlm_model_key: str) -> dict:
    """Vision pass: structured clinical caption + etiology + depth from the raw photo."""
    t0 = time.perf_counter()
    try:
        vlm = make_llm(vlm_model_key)  # gpt-4o-mini / gemini-2.5-flash are vision-capable
        data_uri = f"data:{mime};base64,{image_b64}"
        messages = [
            SystemMessage(content=VLM_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": _vlm_human_prompt(time_payload_text, demographics_text)},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]),
        ]
        response = vlm.invoke(messages)
        raw = _strip_thinking(response.content if isinstance(response.content, str)
                              else str(response.content))
        parsed = _parse_vlm_json(raw)
        inp, out = _extract_tokens(response)
        latency_ms = (time.perf_counter() - t0) * 1000

        out_dict = _empty_vlm_result()
        out_dict.update({
            "caption": parsed.get("caption", "") or raw[:1200],
            "time_crossvalidation": parsed.get("note", ""),   # blind: filled downstream w/ computed discrepancy
            "infection": parsed.get("infection", "Undetermined") or "Undetermined",
            "moisture": parsed.get("moisture", "Undetermined") or "Undetermined",
            "tissue": parsed.get("tissue", {}) or {},
            "etiology": parsed.get("etiology", out_dict["etiology"]) or out_dict["etiology"],
            "depth": parsed.get("depth", out_dict["depth"]) or out_dict["depth"],
            "periwound": parsed.get("periwound", ""),
            "anatomical_location": parsed.get("anatomical_location", ""),
            "urgency_flags": parsed.get("urgency_flags", []) or [],
            "dressing_implications": parsed.get("dressing_implications", ""),
            "input_tokens": inp, "output_tokens": out,
            "cost_usd": _compute_cost(VLM_REGISTRY, vlm_model_key, inp, out),
            "latency_ms": round(latency_ms, 1),
            "error": "" if parsed else "VLM returned non-JSON; raw caption used",
        })
        # normalise nested etiology/depth shape
        if not isinstance(out_dict["etiology"], dict):
            out_dict["etiology"] = {"label": str(out_dict["etiology"]), "confidence": "low", "rationale": ""}
        if not isinstance(out_dict["depth"], dict):
            out_dict["depth"] = {"label": str(out_dict["depth"]), "rationale": ""}
        return out_dict
    except Exception as e:
        import traceback; traceback.print_exc()
        res = _empty_vlm_result(error=str(e))
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res


def resolve_wound_depth(vlm_depth_label: str, self_report: str) -> str:
    sr = (self_report or "").lower()
    if sr in ("cavity", "deep", "deeper"):
        return "cavity"
    if (vlm_depth_label or "").lower() == "cavity":
        return "cavity"
    return "superficial"


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION — PATIENT-FRIENDLY SCHEMA (G1-F / v5 testset format) with [S#] cites
# ══════════════════════════════════════════════════════════════════════════════

PATIENT_SYSTEM_PROMPT = """You are VerdaSense, a clinical wound-care assistant writing a SHORT, \
PLAIN-LANGUAGE dressing guide for a patient managing their wound at home. Write warmly, simply and \
professionally, and define any clinical term in-line (e.g. "slough (soft dead tissue)").

STRICT GROUNDING & SAFETY RULES — follow without exception:
1. Ground every clinical claim in the numbered Sources [S1]..[Sn] in the user message. After each claim, \
cite the source(s) like [S1] or [S3][S5]. Do NOT use outside knowledge that is not in the Sources.
2. TYPE BEFORE PRODUCT. The dressing TYPE/category is fixed by the wound-care algorithm in Source [S1]. \
Your "Primary" dressing MUST be a category from that algorithm list. State the generic dressing TYPE first.
3. PRODUCTS ARE QUOTED, NEVER INVENTED. Brand/product names must be quoted verbatim from the Sources. \
If no product appears in the Sources, give the dressing type only and omit the example. Whenever you name a \
brand, append its generic dressing class in parentheses, e.g. "Flaminal Forte (alginogel)" or \
"RenoCare Thin (thin hydrocolloid)". The class is stated in the product's monograph in the Sources.
3b. CONTRAINDICATION CONSISTENCY GUARD (clinical safety). The binding algorithm in Source [S1] lists the \
dressing classes that are EXCLUDED/contraindicated for this wound type (e.g. charcoal/activated carbon, silver). \
Before finalising, cross-check every product and dressing you recommend against that exclusion list. If a \
product's generic class is excluded for this wound type, you MUST NOT recommend it — even if a local DyaMed \
protocol lists it — and MUST NOT place it in "Example Products" or "Step-by-Step Care". The binding algorithm's \
contraindications OVERRIDE local product protocols. Never recommend any dressing whose class also appears in \
your "Dressings to Avoid" section — that is a contradiction and is forbidden.
3c. MATCH ABSORBENCY TO EXUDATE. Where the algorithm/sources offer a choice, match the primary dressing to \
the exudate level: for HIGH exudate prefer a high-absorbency primary (alginate, hydrofibre/gelling fibre, or \
alginogel such as Flaminal Forte) over plain foam (foam suits moderate exudate); for DRY/LOW exudate prefer a \
moisture-donating primary (hydrogel, film, hydrocolloid). Only choose from classes the binding algorithm [S1] allows.
3d. THE VLM VISUAL ASSESSMENT IS ADVISORY AND NEVER DE-ESCALATES. If a visual (image) assessment is provided, \
it may only ADD caution — if the photo shows infection, necrosis, or a red flag the T.I.M.E. labels missed, raise \
it. It must NEVER DOWNGRADE care: when the T.I.M.E. labels or the patient's notes indicate infection, a \
subclinical-infection signal, or a need for antibiotics/referral, a "clean-looking" image MUST NOT soften the \
antimicrobial, antibiotic, or referral advice, or shift the primary dressing off an antimicrobial. Escalation is \
driven by the labels and the notes; the image can reinforce it but can never remove it.
4. Keep each section to at most 2 short sentences. Total under ~280 words. Mobile-friendly.
5. Use EXACTLY these section headers, in this order, and do not add or omit any:

## Your Wound
## Dressing You Need
## Example Products
## Dressings to Avoid
## How Often to Change
## Antibiotics?
## Do You Need to See a Doctor?
## Step-by-Step Care
## ⚠️ Warning — Get Help Now

In "Dressing You Need" give a **Primary:** and a **Secondary:** line.
In "Antibiotics?" never tell the patient to take antibiotics themselves — say whether they are likely \
needed and to see a clinician for a swab (culture & sensitivity).
In "Do You Need to See a Doctor?" state clearly whether urgent referral is needed.
The "⚠️ Warning — Get Help Now" section must stay direct and unsoftened even though the rest is gentle."""


def _build_source_block(ordered_chunks) -> str:
    """All retrieved chunks labelled [S1]..[Sn] (S1 = binding algorithm chunk)."""
    block = ""
    for i, chunk in enumerate(ordered_chunks, start=1):
        meta = chunk.metadata
        source = meta.get("source", "Unknown")
        authority = meta.get("authority", "")
        year = meta.get("year", "")
        raw_text = meta.get("raw_text", chunk.page_content)
        header = f"\n[S{i}] {source}"
        if authority:
            header += f" [{authority}"
            if year: header += f", {year}"
            header += "]"
        if i == 1:
            header += "  ← BINDING WOUND-TYPE ALGORITHM (primary dressing must come from here)"
        block += header + f"\n{raw_text}\n"
    return block


def _build_etiology_note(classifier) -> str:
    etiology = classifier.get("etiology", "generic")
    if etiology == "diabetic_foot":
        return ("\nETIOLOGY NOTE — DIABETIC FOOT: adhesive bordered foam and hydrocolloid are not "
                "recommended on diabetic feet; prefer non-adhesive dressings; silver is first-line "
                "antimicrobial; offloading and glycaemic control are essential alongside the dressing.\n")
    if etiology == "burn":
        return ("\nETIOLOGY NOTE — BURN: small superficial burns use hydrogel, hydrocolloid, film, or "
                "silicone/paraffin tulle; foam alone is not first-line.\n")
    if etiology == "skin_tear":
        return ("\nETIOLOGY NOTE — SKIN TEAR / FRAGILE SKIN: silicone-covered foam is preferred; avoid "
                "adhesives on fragile skin; remove in the direction of the skin flap.\n")
    return ""


def build_patient_messages(
    ordered_chunks, assessment_text, narrative_query, classifier,
    vlm_block, depth_block, model_key,
) -> list:
    """Build the [SystemMessage, HumanMessage] for the patient-friendly generation.
    Shared by both the blocking endpoint and the streaming endpoint."""
    wt = classifier["wound_type"]
    source_block = _build_source_block(ordered_chunks)

    mandatory = ""
    if classifier["referral_required"]:
        reason = classifier.get("escalation_reason", "") or ""
        mandatory += (
            f"\nMANDATORY — REFERRAL: This wound (Type {wt}) requires hospital/specialist referral. "
            f"{('Reason: ' + reason) if reason else ''} The 'Do You Need to See a Doctor?' section "
            f"must clearly say YES, urgent referral is needed.\n"
        )
    if classifier["antibiotic_required"]:
        if classifier.get("subclinical_infection"):
            mandatory += (
                "\nCLINICAL ALERT — SUBCLINICAL INFECTION: the notes contain infection signals despite a "
                "'Not infected' CV label. Acknowledge this, recommend an antimicrobial dressing, and advise review.\n"
            )
        mandatory += ("\nMANDATORY — ANTIBIOTIC: the 'Antibiotics?' section must say antibiotics are likely "
                      "needed and to see a clinician for a swab (do NOT tell the patient to self-medicate).\n")

    etiology_note = _build_etiology_note(classifier)

    human_prompt = f"""PATIENT QUESTION (internal — answer in plain language):
{narrative_query}

{assessment_text}
{vlm_block}
{depth_block}
{mandatory}
{etiology_note}
NUMBERED CLINICAL SOURCES (cite as [S#]):
{source_block}

Write the patient guide now using EXACTLY the 9 required section headers, in order, citing [S#] after each claim.
If wound_depth is 'cavity', recommend cavity-filling forms (rope/ribbon alginate, hydrofibre, cavity foam) rather than flat sheets, and say so.
If the visual assessment flags a discrepancy with the CV labels (e.g. possible infection), surface it gently in 'Do You Need to See a Doctor?' and the warning section."""

    sys_prompt = PATIENT_SYSTEM_PROMPT
    if MODEL_REGISTRY.get(model_key, {}).get("no_think", False):
        sys_prompt = "/no_think\n\n" + sys_prompt

    return [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]


def generate_patient_recommendation(
    ordered_chunks, assessment_text, narrative_query, classifier,
    vlm_block, depth_block, llm, model_key,
) -> tuple[str, int, int]:
    try:
        messages = build_patient_messages(
            ordered_chunks, assessment_text, narrative_query, classifier,
            vlm_block, depth_block, model_key)
        response = llm.invoke(messages)
        result = _strip_thinking(response.content)
        inp, out = _extract_tokens(response)
        return result, inp, out
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Clinical analysis error: {str(e)}", 0, 0


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT GALLERY  (static DyaMed catalogue — placeholder images for now)
# ══════════════════════════════════════════════════════════════════════════════

PRODUCT_CATALOG = {
    "dermacyn_solution": {
        "name": "Dermacyn WoundCare Solution", "brand": "Sonoma/DyaMed",
        "dressing_class": "Super-oxidised HOCl antimicrobial cleanser/soak",
        "moh_category": "Antimicrobial wound cleanser", "availability": "clinic"},
    "dermacyn_hydrogel": {
        "name": "Dermacyn WoundCare Hydrogel", "brand": "Sonoma/DyaMed",
        "dressing_class": "HOCl antimicrobial hydrogel",
        "moh_category": "Hydrogel", "availability": "clinic"},
    "flaminal_hydro": {
        "name": "Flaminal Hydro", "brand": "Flen Health/DyaMed",
        "dressing_class": "Enzyme alginogel (3.5% alginate)",
        "moh_category": "Alginate/Alginogel (light-moderate exudate)", "availability": "clinic"},
    "flaminal_forte": {
        "name": "Flaminal Forte", "brand": "Flen Health/DyaMed",
        "dressing_class": "Enzyme alginogel (5.5% alginate)",
        "moh_category": "Alginate/Alginogel (moderate-heavy exudate)", "availability": "clinic"},
    "zorflex": {
        "name": "Zorflex", "brand": "DyaMed",
        "dressing_class": "100% activated carbon cloth (antimicrobial contact layer)",
        "moh_category": "Charcoal/activated carbon", "availability": "clinic"},
    "zorflex_la": {
        "name": "Zorflex LA", "brand": "DyaMed",
        "dressing_class": "Activated carbon cloth, low-adherent (dry wounds)",
        "moh_category": "Charcoal/activated carbon", "availability": "clinic"},
    "drawtex": {
        "name": "Drawtex", "brand": "DyaMed",
        "dressing_class": "Hydroconductive dressing (LevaFiber)",
        "moh_category": "High-exudate absorbent (hydroconductive)", "availability": "clinic"},
    "renocare_thin": {
        "name": "RenoCare Thin", "brand": "DyaMed",
        "dressing_class": "Thin hydrocolloid",
        "moh_category": "Hydrocolloid", "availability": "OTC"},
    "renocare_b": {
        "name": "RenoCare B", "brand": "DyaMed",
        "dressing_class": "Foam-backed hydrocolloid",
        "moh_category": "Hydrocolloid", "availability": "OTC"},
    "renofoam": {
        "name": "RenoFoam", "brand": "DyaMed",
        "dressing_class": "Polyurethane foam",
        "moh_category": "Foam", "availability": "OTC"},
}

# Wound-type → grounded DyaMed product set (from the DyaMed WT protocols)
WT_PRODUCTS = {
    1: ["renocare_thin", "renofoam"],
    2: ["renofoam", "flaminal_hydro", "drawtex"],
    3: ["dermacyn_hydrogel", "zorflex", "renocare_b"],
    4: ["flaminal_forte", "drawtex", "zorflex"],
    5: ["dermacyn_hydrogel", "dermacyn_solution", "renocare_b"],
    6: ["flaminal_forte", "drawtex", "dermacyn_solution"],
    7: ["dermacyn_hydrogel", "zorflex_la", "dermacyn_solution"],
    8: ["flaminal_forte", "drawtex", "dermacyn_solution", "zorflex"],
}


def build_product_gallery(wound_type: int, moisture_norm: str = "") -> list:
    keys = list(WT_PRODUCTS.get(wound_type, ["renofoam", "flaminal_hydro"]))
    # Exudate-tier match: Flaminal Hydro (light-moderate) ↔ Forte (moderate-heavy)
    high, dry = (moisture_norm == "High exudate"), (moisture_norm == "Dry")
    adjusted = []
    for k in keys:
        if high and k == "flaminal_hydro": k = "flaminal_forte"
        elif dry and k == "flaminal_forte": k = "flaminal_hydro"
        if k not in adjusted:
            adjusted.append(k)
    keys = adjusted
    gallery = []
    for k in keys:
        p = PRODUCT_CATALOG.get(k)
        if not p:
            continue
        gallery.append({
            "name": p["name"], "brand": p["brand"],
            "dressing_class": p["dressing_class"], "moh_category": p["moh_category"],
            "availability": p["availability"],
            "note": "Example product — any equivalent of this dressing class is suitable.",
            "image": None,  # placeholder rendered client-side until a real gallery is built
        })
    return gallery


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="wound_index_multimodal.html",
        context={"request": request},
    )


async def _prepare_context(
    necrotic_pct, slough_pct, granulation_pct, infection, moisture, edge,
    notes, llm_model, vlm_model, multimodal_enabled, diabetes, depth_self_report, image,
) -> dict:
    """Everything up to (but not including) generation: normalise → classify → retrieve →
    VLM caption → depth/DFU → prompt blocks. Returns the JSON-serialisable `meta` plus the
    internals the generation step needs. Shared by the blocking and streaming endpoints."""
    t_start = time.perf_counter()
    mm_on = str(multimodal_enabled).lower() in ("true", "1", "yes", "on")
    session_id = str(uuid.uuid4())
    if llm_model not in VALID_MODELS:
        print(f"[Warn] unknown LLM '{llm_model}' — fallback gpt-4o-mini"); llm_model = "gpt-4o-mini"
    if vlm_model not in VALID_VLMS:
        print(f"[Warn] unknown VLM '{vlm_model}' — fallback gpt-4o-mini"); vlm_model = "gpt-4o-mini"

    print(f"\n{'─'*60}")
    print(f"[Session] {session_id[:8]} | gen={llm_model} | vlm={vlm_model} | multimodal={mm_on}")

    # ── Step 1: Normalise inputs ────────────────────────────────────────────────
    tissue_profile = interpret_tissue_percentages(necrotic_pct, slough_pct, granulation_pct)
    infection_norm = normalize_infection(infection)
    moisture_norm  = normalize_moisture(moisture)
    edge_norm      = normalize_edge(edge)

    notes_for_rules = notes
    if str(diabetes).lower() == "yes" and "diabet" not in notes.lower():
        notes_for_rules = (notes + " | Patient reports diabetes.").strip(" |")

    print(f"[Input]  N={tissue_profile['necrotic_pct']}% S={tissue_profile['slough_pct']}% "
          f"G={tissue_profile['granulation_pct']}% | I={infection_norm} M={moisture_norm} E={edge_norm}")

    # ── Step 2: Rule-based classifier ───────────────────────────────────────────
    classifier = classify_wound(tissue_profile, infection_norm, moisture_norm, notes_for_rules)
    print(f"[Classifier] {classifier['classifier_notes']}")

    # ── Step 3: Narrative query ─────────────────────────────────────────────────
    narrative_query = build_narrative_query(
        tissue_profile, infection_norm, moisture_norm, edge_norm, notes)

    # ── Step 4: Confidence ──────────────────────────────────────────────────────
    confidence_score, confidence_label = _compute_confidence(narrative_query, k=3)

    # ── Step 5: Dense multi-axis retrieval (UNCHANGED) ──────────────────────────
    t_retr = time.perf_counter()
    top_chunks, retrieval_notes = retrieve_chunks_multiaxis(
        narrative_query=narrative_query, tissue_profile=tissue_profile,
        infection_norm=infection_norm, moisture_norm=moisture_norm,
        classifier=classifier, notes=notes, top_n=6,
    )
    retrieval_latency_ms = (time.perf_counter() - t_retr) * 1000
    ordered_chunks, has_algo = order_sources(top_chunks, classifier["wound_type"])
    evidence = build_evidence_list(ordered_chunks)

    # ── Step 6: Assessment text (CV labels) ─────────────────────────────────────
    assessment_text = (
        f"T.I.M.E. WOUND ASSESSMENT (from CV pipeline):\n"
        f"T (Tissue)    : {tissue_profile['clinical_tissue']} — "
        f"Necrotic {tissue_profile['necrotic_pct']}%, Slough {tissue_profile['slough_pct']}%, "
        f"Granulation {tissue_profile['granulation_pct']}% (non-viable {tissue_profile['non_viable_pct']}%)\n"
        f"I (Infection) : {infection_norm}\n"
        f"M (Moisture)  : {moisture_norm}\n"
        f"E (Edge)      : {edge_norm}\n"
        f"Rule-based wound type: {classifier['wound_type']} "
        f"(referral={classifier['referral_required']}, antibiotic={classifier['antibiotic_required']})"
    )
    if notes.strip():
        assessment_text += f"\nPatient notes: {notes.strip()}"

    # ── Step 7: VLM caption (multimodal only, needs image) ──────────────────────
    vlm_result = _empty_vlm_result(error="multimodal disabled" if not mm_on else "no image provided")
    image_provided = False
    if mm_on and image is not None:
        try:
            img_bytes = await image.read()
            if img_bytes:
                image_provided = True
                img_b64 = base64.b64encode(img_bytes).decode("ascii")
                mime = image.content_type or "image/jpeg"
                demo_bits = []
                if str(diabetes).lower() == "yes":   demo_bits.append("patient reports diabetes")
                elif str(diabetes).lower() == "no":  demo_bits.append("no diabetes reported")
                if depth_self_report not in ("unknown", ""):
                    demo_bits.append(f"patient says wound looks '{depth_self_report}'")
                demo_text = "; ".join(demo_bits) if demo_bits else "none provided"
                print(f"[VLM]    captioning image ({len(img_bytes)} bytes) with {vlm_model}…")
                vlm_result = generate_vlm_caption(
                    image_b64=img_b64, mime=mime, time_payload_text=assessment_text,
                    demographics_text=demo_text, vlm_model_key=vlm_model,
                )
                print(f"[VLM]    etiology={vlm_result['etiology'].get('label')} "
                      f"depth={vlm_result['depth'].get('label')} "
                      f"({vlm_result['latency_ms']:.0f} ms, ${vlm_result['cost_usd']:.6f})")
        except Exception as e:
            print(f"[VLM]    error reading/caption image: {e}")
            vlm_result = _empty_vlm_result(error=str(e))

    # ── Step 7b: Compute the CV discrepancy (blind VLM vs CV label) — the real cross-validation ──
    if image_provided and not vlm_result.get("error"):
        cv_infected  = (infection_norm == "Locally infected")
        vlm_inf      = vlm_result.get("infection", "Undetermined")
        vlm_known    = vlm_inf in ("Infected", "Not infected")
        vlm_infected = (vlm_inf == "Infected")
        if vlm_known and (vlm_infected != cv_infected):
            if vlm_infected and not cv_infected:      # the dangerous direction: CV MISSED infection
                vlm_result["time_crossvalidation"] = (
                    "⚠ DISCREPANCY: the photo shows possible infection, but the CV pipeline labelled it "
                    "'not infected'. Recommend clinical review before self-care.")
                if "possible missed infection (visual)" not in vlm_result.get("urgency_flags", []):
                    vlm_result.setdefault("urgency_flags", []).append("possible missed infection (visual)")
            else:                                     # VLM sees clean but CV said infected
                vlm_result["time_crossvalidation"] = (
                    "Note: the photo does not obviously show infection, though the CV pipeline labelled it "
                    "'infected'. The CV infection label is retained for safety; monitor closely.")
        else:
            vlm_result["time_crossvalidation"] = (
                f"Independent visual read ({vlm_inf}) is consistent with the CV infection label.")
        print(f"[VLM]    blind read: infection={vlm_inf} | CV={infection_norm} | "
              f"{'DISCREPANCY' if (vlm_known and vlm_infected!=cv_infected) else 'agrees'}")

    # ── Step 8: Resolve depth + DFU flag ────────────────────────────────────────
    wound_depth_final = resolve_wound_depth(
        vlm_result["depth"].get("label", "undetermined") if image_provided else "undetermined",
        depth_self_report,
    )
    loc = (vlm_result.get("anatomical_location", "") + " " + notes).lower()
    is_foot = any(w in loc for w in ["foot", "ankle", "plantar", "heel", "toe"])
    et_label = (vlm_result["etiology"].get("label", "") or "").lower()
    dfu_flag = (
        (str(diabetes).lower() == "yes" and is_foot)
        or "diabetic_foot" in et_label
        or classifier.get("etiology") == "diabetic_foot"
    )

    # ── Step 9: Build VLM + depth prompt blocks (generation context) ────────────
    if image_provided and not vlm_result.get("error"):
        uf = vlm_result.get("urgency_flags") or []
        vlm_block = (
            "\nVLM VISUAL ASSESSMENT (direct observation of the wound photo — use to personalise "
            "'Your Wound', cross-check the CV labels, and inform urgency; do NOT cite as [S#]):\n"
            f"- Caption: {vlm_result.get('caption','')}\n"
            f"- CV cross-validation: {vlm_result.get('time_crossvalidation','')}\n"
            f"- Likely etiology: {vlm_result['etiology'].get('label')} "
            f"({vlm_result['etiology'].get('confidence')})\n"
            f"- Periwound: {vlm_result.get('periwound','')}\n"
            f"- Visual urgency flags: {', '.join(uf) if uf else 'none noted'}\n"
            f"- Dressing implications (visual): {vlm_result.get('dressing_implications','')}\n"
        )
    else:
        vlm_block = ("\n(No usable wound image — recommendation is grounded in the CV T.I.M.E. labels "
                     "and guidelines only.)\n")

    depth_block = f"\nWOUND DEPTH (resolved): {wound_depth_final}"
    if dfu_flag:
        depth_block += "\nDFU CONTEXT: diabetic foot context active — add offloading + glycaemic + referral caveat."
    depth_block += "\n"

    sources = list(dict.fromkeys(c.metadata.get("source", "Unknown") for c in ordered_chunks))
    chunk_texts = [c.page_content for c in ordered_chunks]
    product_gallery = build_product_gallery(classifier["wound_type"], moisture_norm)

    meta = {
        "evidence":         evidence,
        "sources":          sources,
        "chunk_texts":      chunk_texts,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "retrieval_notes":  retrieval_notes,
        "tissue_breakdown": {
            "necrotic_pct":    tissue_profile["necrotic_pct"],
            "slough_pct":      tissue_profile["slough_pct"],
            "granulation_pct": tissue_profile["granulation_pct"],
        },
        "clinical_flags": {
            "infection_norm":        infection_norm,
            "moisture_norm":         moisture_norm,
            "edge_norm":             edge_norm,
            "subclinical_infection": classifier.get("subclinical_infection", False),
            "etiology":              classifier.get("etiology", "generic"),
            "wound_type":            classifier.get("wound_type"),
        },
        "narrative_query":      narrative_query,
        "classifier_output":    classifier,
        "session_id":           session_id,
        "llm_model":            llm_model,
        "llm_label":            MODEL_REGISTRY[llm_model]["label"],
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "multimodal_enabled":   mm_on,
        "image_provided":       image_provided,
        "vlm": {
            "model": vlm_model, "label": VLM_REGISTRY[vlm_model]["label"],
            "caption":               vlm_result.get("caption", ""),
            "time_crossvalidation":  vlm_result.get("time_crossvalidation", ""),
            "etiology":              vlm_result.get("etiology", {}),
            "depth":                 vlm_result.get("depth", {}),
            "periwound":             vlm_result.get("periwound", ""),
            "anatomical_location":   vlm_result.get("anatomical_location", ""),
            "urgency_flags":         vlm_result.get("urgency_flags", []),
            "dressing_implications": vlm_result.get("dressing_implications", ""),
            "input_tokens":  vlm_result.get("input_tokens", 0),
            "output_tokens": vlm_result.get("output_tokens", 0),
            "cost_usd":      round(vlm_result.get("cost_usd", 0.0), 8),
            "latency_ms":    vlm_result.get("latency_ms", 0.0),
            "error":         vlm_result.get("error", ""),
        },
        "wound_depth_final": wound_depth_final,
        "dfu_flag":          dfu_flag,
        "demographics":      {"diabetes": diabetes, "depth_self_report": depth_self_report},
        "product_gallery":   product_gallery,
        "view_default":      "prod",
    }

    return {
        "meta": meta,
        "ordered_chunks":  ordered_chunks,
        "assessment_text": assessment_text,
        "narrative_query": narrative_query,
        "classifier":      classifier,
        "vlm_block":       vlm_block,
        "depth_block":     depth_block,
        "vlm_cost":        vlm_result.get("cost_usd", 0.0) if image_provided else 0.0,
        "llm_model":       llm_model,
        "t_start":         t_start,
    }


def _sse(obj: dict) -> str:
    """Serialise one event as a Server-Sent-Events frame (single-line JSON, safe to split on \\n\\n)."""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/get_recommendation")
async def get_recommendation(
    necrotic_pct:       float = Form(...),
    slough_pct:         float = Form(...),
    granulation_pct:    float = Form(...),
    infection:          str   = Form(...),
    moisture:           str   = Form(...),
    edge:               str   = Form(...),
    notes:              str   = Form(""),
    tissue_confidence:  float = Form(0.0),
    llm_model:          str   = Form("gpt-4o-mini"),
    vlm_model:          str   = Form("gpt-4o-mini"),
    multimodal_enabled: str   = Form("true"),
    diabetes:           str   = Form("unknown"),
    depth_self_report:  str   = Form("unknown"),
    image:              UploadFile = File(None),
):
    """Blocking (non-streaming) endpoint — full JSON in one response. Kept as a fallback."""
    t0 = time.perf_counter()
    try:
        ctx = await _prepare_context(
            necrotic_pct, slough_pct, granulation_pct, infection, moisture, edge,
            notes, llm_model, vlm_model, multimodal_enabled, diabetes, depth_self_report, image)
        llm_model = ctx["llm_model"]
        llm = make_llm(llm_model)
        t_gen = time.perf_counter()
        result, input_tokens, output_tokens = generate_patient_recommendation(
            ordered_chunks=ctx["ordered_chunks"], assessment_text=ctx["assessment_text"],
            narrative_query=ctx["narrative_query"], classifier=ctx["classifier"],
            vlm_block=ctx["vlm_block"], depth_block=ctx["depth_block"],
            llm=llm, model_key=llm_model,
        )
        generation_latency_ms = (time.perf_counter() - t_gen) * 1000
        total_latency_ms      = (time.perf_counter() - ctx["t_start"]) * 1000
        gen_cost   = _compute_cost(MODEL_REGISTRY, llm_model, input_tokens, output_tokens)
        total_cost = gen_cost + ctx["vlm_cost"]
        return JSONResponse({
            **ctx["meta"],
            "result":               result,
            "input_tokens":         input_tokens,
            "output_tokens":        output_tokens,
            "cost_usd":             round(total_cost, 8),
            "gen_cost_usd":         round(gen_cost, 8),
            "generation_latency_ms": round(generation_latency_ms, 1),
            "total_latency_ms":      round(total_latency_ms, 1),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return JSONResponse({
            "result": f"System error: {str(e)}", "evidence": [], "sources": [], "chunk_texts": [],
            "confidence_score": 0.0, "confidence_label": "LOW",
            "retrieval_notes": ["System error during processing"],
            "tissue_breakdown": {}, "clinical_flags": {}, "narrative_query": "",
            "classifier_output": {}, "session_id": "", "llm_model": llm_model,
            "llm_label": MODEL_REGISTRY.get(llm_model, {}).get("label", llm_model),
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "retrieval_latency_ms": 0.0, "generation_latency_ms": 0.0, "total_latency_ms": total_ms,
            "multimodal_enabled": str(multimodal_enabled).lower() in ("true","1","yes","on"),
            "image_provided": False, "vlm": _empty_vlm_result(error=str(e)),
            "wound_depth_final": "superficial", "dfu_flag": False,
            "demographics": {}, "product_gallery": [], "view_default": "prod",
        }, status_code=500)


@app.post("/get_recommendation_stream")
async def get_recommendation_stream(
    necrotic_pct:       float = Form(...),
    slough_pct:         float = Form(...),
    granulation_pct:    float = Form(...),
    infection:          str   = Form(...),
    moisture:           str   = Form(...),
    edge:               str   = Form(...),
    notes:              str   = Form(""),
    tissue_confidence:  float = Form(0.0),
    llm_model:          str   = Form("gpt-4o-mini"),
    vlm_model:          str   = Form("gpt-4o-mini"),
    multimodal_enabled: str   = Form("true"),
    diabetes:           str   = Form("unknown"),
    depth_self_report:  str   = Form("unknown"),
    image:              UploadFile = File(None),
):
    """Streaming endpoint (SSE). Emits one `meta` event (panels render immediately),
    then `delta` events token-by-token, then a final `done` event with analytics."""
    try:
        ctx = await _prepare_context(
            necrotic_pct, slough_pct, granulation_pct, infection, moisture, edge,
            notes, llm_model, vlm_model, multimodal_enabled, diabetes, depth_self_report, image)
    except Exception as e:
        import traceback; traceback.print_exc()
        async def err_gen():
            yield _sse({"type": "error", "message": str(e)})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    model_key = ctx["llm_model"]
    messages = build_patient_messages(
        ctx["ordered_chunks"], ctx["assessment_text"], ctx["narrative_query"],
        ctx["classifier"], ctx["vlm_block"], ctx["depth_block"], model_key)
    # OpenAI-compatible models can return token usage during streaming; Gemini reports it natively.
    stream_kwargs = {}
    if model_key in ("gpt-4o-mini", "qwen/qwen3.5-35b-a3b"):
        stream_kwargs["stream_usage"] = True

    async def event_gen():
        # 1) metadata first — UI renders VLM panel, evidence, product gallery instantly
        yield _sse({"type": "meta", **ctx["meta"]})

        full, usage = "", None
        t_gen = time.perf_counter()
        try:
            llm = make_llm(model_key)
            async for chunk in llm.astream(messages, **stream_kwargs):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if piece:
                    full += piece
                    yield _sse({"type": "delta", "text": piece})
                um = getattr(chunk, "usage_metadata", None)
                if um:
                    usage = um
        except Exception as e:
            import traceback; traceback.print_exc()
            yield _sse({"type": "error", "message": str(e)})
            return

        result = _strip_thinking(full)
        if usage:
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
        else:
            inp, out = 0, max(1, len(full) // 4)
        gen_cost   = _compute_cost(MODEL_REGISTRY, model_key, inp, out)
        total_cost = gen_cost + ctx["vlm_cost"]
        gen_lat   = (time.perf_counter() - t_gen) * 1000
        total_lat = (time.perf_counter() - ctx["t_start"]) * 1000
        print(f"[Stream] done | in={inp} out={out} | gen=${gen_cost:.6f} total=${total_cost:.6f} "
              f"| {gen_lat:.0f} ms gen / {total_lat:.0f} ms total")
        yield _sse({
            "type": "done",
            "result":                result,
            "input_tokens":          inp,
            "output_tokens":         out,
            "cost_usd":              round(total_cost, 8),
            "gen_cost_usd":          round(gen_cost, 8),
            "generation_latency_ms": round(gen_lat, 1),
            "total_latency_ms":      round(total_lat, 1),
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
