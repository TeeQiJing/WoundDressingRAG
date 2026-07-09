# %% [markdown]
# # VerdaSense Ablation Study — Stage FYP2 / Experiment G4-A: VLM Caption vs No-Caption (Multi-Run)
#
# **Research question:** *Does adding a VLM clinical caption to the generation stage improve the
# dressing recommendation versus the unimodal (no-caption) baseline?*
#
# This is the headline FYP2 multimodal experiment. The two arms are **literally the production
# app's Multimodal On/Off toggle** (`wound_app_multimodal.py`):
#
# | Arm  | Label | What the generator sees |
# |------|-------|--------------------------|
# | **G4-A0** | Unimodal baseline (no caption) | retrieved KB chunks + T.I.M.E. payload (multimodal **OFF**) |
# | **G4-A1** | Multimodal (with VLM caption) | retrieved KB chunks + T.I.M.E. payload + **VLM caption** (multimodal **ON**) |
#
# **Fixed across both arms (so the only difference is the caption):**
#   - KB / embedding  : `db_wound_care_v5_bge` · `BAAI/bge-large-en-v1.5` (R4-B winner)
#   - Query strategy  : R1-C multi-axis sub-queries (A+B+C)
#   - Retrieval       : Dense-only, k=6  (R5 confirmed: caption is generation-stage only, NOT retrieval)
#   - Generation LLM  : `gpt-4o-mini`  (temperature 0)
#   - Prompt          : G1-F patient-friendly schema (the production `PATIENT_SYSTEM_PROMPT`)
#   - VLM (A1 only)   : `gpt-4o-mini` Vision  (G4-B will ablate this vs Gemini-Vision)
#
# **RAGAS judge:** `gpt-4o-mini` + `text-embedding-3-small` (NEVER changed — same as FYP1)
# **Runs per arm:** 3 (`N_RUNS=3`) → mean ± SD reported.
#
# **Methodology note (improvement over FYP1):** instead of re-implementing the pipeline inline,
# this notebook **imports the live production functions** from `wound_app_multimodal.py`, so the
# ablation measures the *real* system with zero re-implementation drift. The arms call the same code
# path the app uses for Off / On.
#
# **Honest framing (state this in the write-up):** the caption is expected to help most where the
# CV labels are *wrong or incomplete* — the adversarial **Cat G** (label says clean, image infected)
# and the **cavity** case — and may add little on canonical Cat A where the labels already suffice.
# A per-category A0-vs-A1 breakdown (Cell 16) is therefore the key result, not just the global mean.
# Because FA scores grounding in the **KB chunks only**, any visual claim the caption introduces that
# is *not* in the guidelines can *lower* A1's FA — that is a real, measured risk, reported honestly.
#
# **Output files (results/):**
#   - `G4A_{arm}_results_all.json`  — all per-case records across 3 runs
#   - `G4A_{arm}_ragas.json`        — multi-run aggregated FA/AR
#   - `G4A_summary.json`            — A0 vs A1 mean±SD for every metric + per-category delta
#   - `G4A_vlm_captions.json`       — frozen VLM caption snapshots (reproducibility)
#   - `G4A_per_case.csv`            — flat table for inspection

# %% [markdown]
# ## Cell 0 — Environment, Paths & Import the Production Pipeline

# %%
import os, sys, gc, re, json, time, base64, statistics, warnings, datetime
from pathlib import Path
from collections import Counter, defaultdict

import torch
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
def _find_root(start):
    start = Path(start).resolve()
    for cand in [start, *start.parents]:
        if (cand / "wound_app_multimodal.py").exists():
            return cand
    return start
PROJECT_ROOT = _find_root(Path(__file__).parent if "__file__" in dir() else Path.cwd())
NOTEBOOK_DIR = PROJECT_ROOT / "RAGAS_EVAL" / "G4A_Multimodal_Caption"
RESULTS_DIR  = NOTEBOOK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TESTSET_PATH = PROJECT_ROOT / "ragas_testset" / "wound_testset_v5.json"

# Make the production app importable + run from project root (relative paths inside it resolve)
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"Project root : {PROJECT_ROOT}")
print(f"Testset      : {TESTSET_PATH.name}")
print(f"Results dir  : {RESULTS_DIR}")
print(f"CUDA         : {torch.cuda.is_available()}")
print("\nImporting wound_app_multimodal (loads BGE + v5 KB — ~15–30 s)…")

# THE SYSTEM UNDER TEST — import the real production pipeline
import wound_app_multimodal as mm

# RAGAS
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

print(f"\n✅ Pipeline imported. KB chunks in v5 store: {mm.db._collection.count()}")

# %% [markdown]
# ## Cell 1 — Configuration
#
# Two arms (A0 = caption OFF, A1 = caption ON). Generation LLM + VLM are FIXED here
# (G4-B ablates the VLM; this experiment isolates *caption presence*).

# %%
EXP_ID    = "G4A"
N_RUNS    = 3
ARMS      = ["A0", "A1"]

GEN_LLM   = "gpt-4o-mini"      # generation model — fixed for G4-A
VLM_MODEL = "gpt-4o-mini"      # vision model for A1 — fixed for G4-A (G4-B ablates this)

# RAGAS judge — NEVER changed (identical to FYP1)
RAGAS_LLM_MODEL   = "gpt-4o-mini"
RAGAS_EMBED_MODEL = "text-embedding-3-small"

# VLM captions are temperature-0 (near-deterministic) and costly → generate once per case,
# reuse across the 3 runs. The 3-run variance therefore reflects generation stochasticity.
CACHE_CAPTIONS = True

ARM_CONFIG = {
    "A0": {"label": "Unimodal baseline (no caption)",
           "multimodal": False,
           "description": "Generation from retrieved KB chunks + T.I.M.E. payload only. "
                          "Equivalent to the app with Multimodal OFF."},
    "A1": {"label": "Multimodal (VLM caption)",
           "multimodal": True,
           "description": "Same retrieval + payload PLUS the VLM clinical caption injected into "
                          "generation. Equivalent to the app with Multimodal ON."},
}

print("Configuration")
print(f"  EXP_ID         : {EXP_ID}")
print(f"  N_RUNS         : {N_RUNS}")
print(f"  Arms           : A0 (no caption) vs A1 (VLM caption)")
print(f"  Generation LLM : {GEN_LLM}  (fixed, temp 0)")
print(f"  VLM (A1)       : {VLM_MODEL}  (fixed)")
print(f"  Retrieval      : R1-C multi-axis · dense · k=6 · BGE v5  (fixed, caption-independent)")
print(f"  Prompt         : G1-F patient schema (production PATIENT_SYSTEM_PROMPT)")
print(f"  RAGAS judge    : {RAGAS_LLM_MODEL} + {RAGAS_EMBED_MODEL}  (fixed)")
print(f"  Cache captions : {CACHE_CAPTIONS}")

assert GEN_LLM in mm.VALID_MODELS, f"{GEN_LLM} not in MODEL_REGISTRY"
assert VLM_MODEL in mm.VALID_VLMS, f"{VLM_MODEL} not in VLM_REGISTRY"

# %% [markdown]
# ## Cell 2 — Load v5 Testset (multimodal subset: cases WITH an image)
#
# G4-A needs an image (the A1 arm reads it). We run on every case that has a resolvable
# `image_ref`. Cases without an image are skipped (logged).

# %%
with open(TESTSET_PATH, encoding="utf-8") as f:
    full_testset = json.load(f)

testset, skipped = [], []
for tc in full_testset:
    ref = tc.get("image_ref")
    if ref and (PROJECT_ROOT / ref).exists():
        testset.append(tc)
    else:
        skipped.append(tc.get("case_id"))

print(f"Loaded {len(full_testset)} cases · {len(testset)} have a usable image · skipped {len(skipped)}")
if skipped:
    print(f"  Skipped (no/broken image): {skipped}")
print(f"  Category distribution: {dict(Counter(tc['category'] for tc in testset))}")

# Validate the fields the correctness checks need
need = ["time_payload", "reference", "reference_contexts", "allowed_dressings",
        "contraindicated_dressings", "antibiotic_required", "referral_required", "image_ref"]
miss = [k for k in need if k not in testset[0]]
print("✅ all required fields present" if not miss else f"⚠ missing fields: {miss}")

# %% [markdown]
# ## Cell 3 — RAGAS Judge (fixed: gpt-4o-mini + text-embedding-3-small)

# %%
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0))
ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBED_MODEL))
print(f"RAGAS judge : {RAGAS_LLM_MODEL} (LLM) + {RAGAS_EMBED_MODEL} (embed) — fixed across A0/A1")

# %% [markdown]
# ## Cell 4 — Per-case Pipeline Helpers (wrap the production functions)
#
# These mirror exactly what `wound_app_multimodal._prepare_context()` does, calling the SAME
# imported functions. `build_case_inputs()` does the shared work (normalise → classify → retrieve →
# order sources → assessment text) once per case; the A0/A1 arms then differ only by the VLM caption.

# %%
def _img_to_b64(image_ref: str):
    """Load image_ref → (base64, mime)."""
    p = PROJECT_ROOT / image_ref
    data = p.read_bytes()
    ext  = p.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/png")
    return base64.b64encode(data).decode("ascii"), mime


def build_case_inputs(tc: dict) -> dict:
    """Shared, caption-independent pipeline (identical to _prepare_context up to retrieval)."""
    tp   = tc["time_payload"]
    prof = mm.interpret_tissue_percentages(tp["necrotic_pct"], tp["slough_pct"], tp["granulation_pct"])
    inf  = mm.normalize_infection(tp["infection"])
    moi  = mm.normalize_moisture(tp["moisture"])
    edg  = mm.normalize_edge(tp["edge"])
    notes = tp.get("notes", "") or ""

    diabetic = bool(tc.get("demographics", {}).get("diabetic", False))
    diabetes = "yes" if diabetic else "no"
    depth_self = "cavity" if tc.get("wound_depth") == "cavity" else "surface"

    # Fold diabetes into notes for the rule layer (same as the app)
    notes_for_rules = notes
    if diabetes == "yes" and "diabet" not in notes.lower():
        notes_for_rules = (notes + " | Patient reports diabetes.").strip(" |")

    cls = mm.classify_wound(prof, inf, moi, notes_for_rules)
    nq  = mm.build_narrative_query(prof, inf, moi, edg, notes)

    chunks, ret_notes = mm.retrieve_chunks_multiaxis(
        narrative_query=nq, tissue_profile=prof, infection_norm=inf,
        moisture_norm=moi, classifier=cls, notes=notes, top_n=6)
    ordered, _ = mm.order_sources(chunks, cls["wound_type"])

    assessment_text = (
        f"T.I.M.E. WOUND ASSESSMENT (from CV pipeline):\n"
        f"T (Tissue)    : {prof['clinical_tissue']} — "
        f"Necrotic {prof['necrotic_pct']}%, Slough {prof['slough_pct']}%, "
        f"Granulation {prof['granulation_pct']}% (non-viable {prof['non_viable_pct']}%)\n"
        f"I (Infection) : {inf}\nM (Moisture)  : {moi}\nE (Edge)      : {edg}\n"
        f"Rule-based wound type: {cls['wound_type']} "
        f"(referral={cls['referral_required']}, antibiotic={cls['antibiotic_required']})"
    )
    if notes.strip():
        assessment_text += f"\nPatient notes: {notes.strip()}"

    return {
        "tissue_profile": prof, "infection_norm": inf, "moisture_norm": moi, "edge_norm": edg,
        "notes": notes, "diabetes": diabetes, "depth_self": depth_self,
        "classifier": cls, "narrative_query": nq,
        "ordered_chunks": ordered,
        "retrieved_contexts": [c.metadata.get("raw_text", c.page_content) for c in ordered],
        "assessment_text": assessment_text,
    }


def build_vlm_block(vlm_result: dict, image_provided: bool) -> str:
    """Identical to the app's vlm_block construction."""
    if image_provided and not vlm_result.get("error"):
        uf = vlm_result.get("urgency_flags") or []
        return (
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
    return ("\n(No usable wound image — recommendation is grounded in the CV T.I.M.E. labels "
            "and guidelines only.)\n")


def build_depth_block(wound_depth_final: str, dfu_flag: bool) -> str:
    b = f"\nWOUND DEPTH (resolved): {wound_depth_final}"
    if dfu_flag:
        b += "\nDFU CONTEXT: diabetic foot context active — add offloading + glycaemic + referral caveat."
    return b + "\n"


print("Per-case pipeline helpers defined (wrap the imported production functions).")

# %% [markdown]
# ## Cell 5 — VLM Caption Cache (A1 only)
#
# Generate the caption once per case (temp-0, near-deterministic) and freeze it. Snapshots are
# saved to disk for reproducibility and for Ms Saw's H1 caption-quality spot-check.

# %%
_CAPTION_CACHE: dict = {}

def get_caption(tc: dict, inputs: dict) -> dict:
    """Return the (cached) VLM caption result for a case. A1 arm only."""
    cid = tc["case_id"]
    if CACHE_CAPTIONS and cid in _CAPTION_CACHE:
        return _CAPTION_CACHE[cid]
    try:
        b64, mime = _img_to_b64(tc["image_ref"])
        demo_bits = []
        demo_bits.append("patient reports diabetes" if inputs["diabetes"] == "yes" else "no diabetes reported")
        if inputs["depth_self"] not in ("unknown", ""):
            demo_bits.append(f"patient says wound looks '{inputs['depth_self']}'")
        vlm = mm.generate_vlm_caption(
            image_b64=b64, mime=mime, time_payload_text=inputs["assessment_text"],
            demographics_text="; ".join(demo_bits), vlm_model_key=VLM_MODEL)
    except Exception as e:
        vlm = mm._empty_vlm_result(error=str(e))
    if CACHE_CAPTIONS:
        _CAPTION_CACHE[cid] = vlm
    return vlm


def resolve_depth_and_dfu(tc, inputs, vlm_result, image_provided):
    """Mirror the app's depth + DFU resolution (differs by arm: A1 uses VLM signals)."""
    vlm_depth = vlm_result["depth"].get("label", "undetermined") if image_provided else "undetermined"
    wound_depth_final = mm.resolve_wound_depth(vlm_depth, inputs["depth_self"])
    loc = ((vlm_result.get("anatomical_location", "") if image_provided else "") + " " + inputs["notes"]).lower()
    is_foot = any(w in loc for w in ["foot", "ankle", "plantar", "heel", "toe"])
    et = (vlm_result["etiology"].get("label", "") or "").lower() if image_provided else ""
    dfu = ((inputs["diabetes"] == "yes" and is_foot) or "diabetic_foot" in et
           or inputs["classifier"].get("etiology") == "diabetic_foot")
    return wound_depth_final, dfu


print("VLM caption cache + depth/DFU resolver defined.")

# %% [markdown]
# ## Cell 6 — Clinical-Correctness Checker (patient schema, v5)
#
# Adapts the FYP1 deterministic safety checker to the **patient-friendly schema**. Scores three
# things from the generated guide, per the ablation map's secondary metrics:
#   1. **Dressing-class correctness** — recommended classes ⊆ `allowed_dressings` AND ∩ `contraindicated` = ∅
#   2. **Antibiotic correctness** — matches `antibiotic_required` (patient phrasing)
#   3. **Referral correctness** — matches `referral_required` (patient phrasing)
# Overall PASS = all applicable checks pass.

# %%
DRESSING_ALIASES = {
    "film": ["film", "transparent film"], "hydrocolloid": ["hydrocolloid", "renocare"],
    "foam": ["foam", "renofoam"], "tulle": ["tulle", "paraffin"], "hydrogel": ["hydrogel", "dermacyn"],
    "alginate": ["alginate", "alginogel", "flaminal"], "alginogel": ["alginogel", "flaminal"],
    "hydrofiber": ["hydrofibre", "hydrofiber", "aquacel"], "silver": ["silver"],
    "iodine": ["iodine", "povidone", "cadexomer"], "charcoal": ["charcoal", "activated carbon", "zorflex"],
    "polymeric_membrane": ["polymeric membrane", "polymem"],
    "hydroconductive": ["hydroconductive", "drawtex"],
}
def _forms(tok): return DRESSING_ALIASES.get(tok, [tok.replace("_", " ")])

_PATIENT_POS_RE = re.compile(r"^##\s*(dressing you need|example products|step-?by-?step)", re.I | re.M)
_PATIENT_REC_RE = re.compile(r"^##\s*(dressing you need|example products)", re.I | re.M)  # excludes Step-by-Step
_PATIENT_AVOID_RE = re.compile(r"^##\s*(dressings to avoid)", re.I | re.M)

def _section_text(answer: str, header_re) -> str:
    lines, keep, out = answer.split("\n"), False, []
    for ln in lines:
        s = ln.strip()
        if s.startswith("##"):
            keep = bool(header_re.match(s)) and not _PATIENT_AVOID_RE.match(s)
            continue
        if keep:
            out.append(ln.lower())
    return " ".join(out)

def _positive_text(answer: str) -> str:
    """'Dressing You Need' / 'Example Products' / 'Step-by-Step' (used for the allowed-dressing check)."""
    t = _section_text(answer, _PATIENT_POS_RE)
    return t if t else answer.lower()

def _rec_only_text(answer: str) -> str:
    """Recommendation sections ONLY ('Dressing You Need' + 'Example Products') — excludes 'Step-by-Step',
    where cautions like 'no tape' / 'not compression' live, so a warned-against item isn't misread as recommended."""
    t = _section_text(answer, _PATIENT_REC_RE)
    return t if t else answer.lower()

def _avoid_text(answer: str) -> str:
    lines, keep, out = answer.split("\n"), False, []
    for ln in lines:
        s = ln.strip()
        if s.startswith("##"):
            keep = bool(_PATIENT_AVOID_RE.match(s)); continue
        if keep: out.append(ln.lower())
    return " ".join(out)

def _recommended(token, answer): return any(f.lower() in _positive_text(answer) for f in _forms(token))
_NEG_RE = re.compile(r"\b(no|not|never|avoid|without|don'?t|isn'?t|aren'?t|nor)\b[\w\s,/()-]{0,22}$")
def _neg_before(text, idx): return bool(_NEG_RE.search(text[max(0, idx-32):idx]))
def _recommended_rec(token, answer):                                     # contraindicated check (negation-aware)
    t = _rec_only_text(answer)
    for f in _forms(token):
        for m in re.finditer(re.escape(f.lower()), t):
            if not _neg_before(t, m.start()): return True                # positive mention (not "without <tok>")
    return False

def check_clinical(answer: str, tc: dict) -> dict:
    if not answer or answer.startswith("ERROR"):
        return {"overall": "FAIL", "_error": "empty/error answer"}
    res = {}
    # 1) contraindicated not positively recommended
    for contra in tc.get("contraindicated_dressings", []):
        base = contra.split("(")[0].strip()              # "iodine (if thyroid)" → "iodine"
        bad  = _recommended_rec(base, answer)            # scan recommendation sections only (not Step-by-Step cautions)
        res[f"avoid_{base}"] = {"result": "FAIL" if bad else "PASS",
                                "reason": f"{base} {'in' if bad else 'not in'} positive recommendation"}
    # 2) at least one allowed dressing recommended
    allow_hit = any(_recommended(t, answer) for t in tc.get("allowed_dressings", []))
    res["allowed_dressing_present"] = {"result": "PASS" if allow_hit else "FAIL",
        "reason": f"allowed hit={allow_hit}"}
    # 3) antibiotic correctness (patient phrasing)
    lo = answer.lower()
    if tc.get("antibiotic_required"):
        ok = any(k in lo for k in ["antibiotic", "swab", "culture", "antimicrobial"])
        res["antibiotic"] = {"result": "PASS" if ok else "FAIL", "reason": f"abx-keyword={ok}"}
    # 4) referral correctness (patient phrasing)
    if tc.get("referral_required"):
        ok = any(k in lo for k in ["see a doctor", "refer", "hospital", "urgent", "specialist", "clinic"])
        res["referral"] = {"result": "PASS" if ok else "FAIL", "reason": f"referral-keyword={ok}"}
    res["overall"] = "FAIL" if any(v.get("result") == "FAIL" for v in res.values() if isinstance(v, dict)) else "PASS"
    return res

print("Patient-schema clinical-correctness checker defined "
      "(dressing-class · antibiotic · referral).")

# %% [markdown]
# ## Cell 7 — RAGAS Evaluation Helper (FA + AR)
#
# Identical contract to FYP1. `retrieved_contexts` = the **KB guideline chunks only** (the caption
# is a generation input, not a context — so FA measures guideline grounding consistently for A0/A1).

# %%
def run_ragas(questions, contexts, answers, references) -> dict:
    samples = [SingleTurnSample(user_input=q, retrieved_contexts=[str(c) for c in ctx],
                                response=a, reference=r)
               for q, ctx, a, r in zip(questions, contexts, answers, references)
               if not a.startswith("ERROR")]
    if not samples:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "per_sample_fa": [], "per_sample_ar": [], "n": 0}
    print(f"    RAGAS on {len(samples)} samples (FA + AR)…")
    result = evaluate(EvaluationDataset(samples),
                      metrics=[Faithfulness(llm=ragas_llm),
                               AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embed)])
    df = result.to_pandas()
    fa_c = next((c for c in df.columns if "faithfulness" in c.lower()), None)
    ar_c = next((c for c in df.columns if "answer_relevancy" in c.lower()), None)
    def _m(s): v = s.dropna().tolist(); return round(sum(v)/len(v), 4) if v else 0.0
    return {"faithfulness": _m(df[fa_c]) if fa_c else 0.0,
            "answer_relevancy": _m(df[ar_c]) if ar_c else 0.0,
            "per_sample_fa": df[fa_c].tolist() if fa_c else [],
            "per_sample_ar": df[ar_c].tolist() if ar_c else [], "n": len(samples)}

print("RAGAS helper defined (FA + AR; contexts = KB chunks only).")

# %% [markdown]
# ## Cell 8 — Generate one answer for one arm

# %%
def generate_arm(arm: str, tc: dict, inputs: dict) -> dict:
    """Produce one recommendation for the given arm (A0 = no caption, A1 = caption)."""
    multimodal = ARM_CONFIG[arm]["multimodal"]
    vlm_result, image_provided, vlm_cost, vlm_lat = mm._empty_vlm_result(), False, 0.0, 0.0

    if multimodal:
        vlm_result = get_caption(tc, inputs)
        image_provided = not vlm_result.get("error")
        vlm_cost, vlm_lat = vlm_result.get("cost_usd", 0.0), vlm_result.get("latency_ms", 0.0)

    wound_depth_final, dfu_flag = resolve_depth_and_dfu(tc, inputs, vlm_result, image_provided)
    vlm_block   = build_vlm_block(vlm_result, image_provided)
    depth_block = build_depth_block(wound_depth_final, dfu_flag)

    messages = mm.build_patient_messages(
        inputs["ordered_chunks"], inputs["assessment_text"], inputs["narrative_query"],
        inputs["classifier"], vlm_block, depth_block, GEN_LLM)

    llm = mm.make_llm(GEN_LLM)
    t0 = time.perf_counter()
    try:
        resp   = llm.invoke(messages)
        answer = mm._strip_thinking(resp.content)
        inp, out = mm._extract_tokens(resp)
    except Exception as e:
        answer, inp, out = f"ERROR: {e}", 0, 0
    gen_lat  = (time.perf_counter() - t0) * 1000
    gen_cost = mm._compute_cost(mm.MODEL_REGISTRY, GEN_LLM, inp, out)

    return {
        "answer": answer, "image_provided": image_provided,
        "wound_depth_final": wound_depth_final, "dfu_flag": dfu_flag,
        "vlm_etiology": vlm_result["etiology"].get("label") if image_provided else None,
        "vlm_caption": vlm_result.get("caption", "") if image_provided else "",
        "gen_input_tokens": inp, "gen_output_tokens": out,
        "gen_cost_usd": round(gen_cost, 8), "vlm_cost_usd": round(vlm_cost, 8),
        "total_cost_usd": round(gen_cost + vlm_cost, 8),
        "gen_latency_ms": round(gen_lat, 1), "vlm_latency_ms": round(vlm_lat, 1),
    }

print("Per-arm generation function defined.")

# %% [markdown]
# ## Cell 9 — One full run (both arms, all imaged cases)
#
# Retrieval + assessment are computed ONCE per case and shared by A0 and A1 (the caption is the only
# difference), then both arms generate. Returns records + RAGAS per arm.

# %%
def run_one_pass(run_idx: int) -> dict:
    print(f"\n{'='*70}\n  G4-A — RUN {run_idx}/{N_RUNS}\n{'='*70}")
    arm_records = {a: [] for a in ARMS}

    for i, tc in enumerate(testset):
        cid, cat = tc["case_id"], tc["category"]
        inputs = build_case_inputs(tc)                      # shared pipeline
        print(f"  [{i+1:>2}/{len(testset)}] {cid:<34} ({cat}) | ret={len(inputs['ordered_chunks'])}", end="")
        for arm in ARMS:
            out = generate_arm(arm, tc, inputs)
            safety = check_clinical(out["answer"], tc)
            rec = {
                "run": run_idx, "arm": arm, "case_id": cid, "category": cat,
                "wound_type_expected": tc.get("wound_type_expected"),
                "wound_type_predicted": inputs["classifier"]["wound_type"],
                "narrative_query": inputs["narrative_query"],
                "reference": tc.get("reference", ""),
                "retrieved_contexts": inputs["retrieved_contexts"],
                "answer": out["answer"],
                "image_provided": out["image_provided"], "vlm_etiology": out["vlm_etiology"],
                "wound_depth_final": out["wound_depth_final"], "dfu_flag": out["dfu_flag"],
                "safety_checks": safety, "safety_overall": safety.get("overall", "N/A"),
                "antibiotic_required": tc.get("antibiotic_required", False),
                "referral_required": tc.get("referral_required", False),
                "gen_cost_usd": out["gen_cost_usd"], "vlm_cost_usd": out["vlm_cost_usd"],
                "total_cost_usd": out["total_cost_usd"],
                "gen_latency_ms": out["gen_latency_ms"], "vlm_latency_ms": out["vlm_latency_ms"],
            }
            arm_records[arm].append(rec)
            print(f" | {arm}:{safety.get('overall','?')[:4]}", end="")
        print(flush=True)
        time.sleep(1.0)

    out = {}
    for arm in ARMS:
        recs = arm_records[arm]
        print(f"\n  RAGAS — {arm} (run {run_idx}):")
        rg = run_ragas([r["narrative_query"] for r in recs],
                       [r["retrieved_contexts"] for r in recs],
                       [r["answer"] for r in recs],
                       [r["reference"] for r in recs])
        print(f"    {arm}: FA={rg['faithfulness']:.4f}  AR={rg['answer_relevancy']:.4f}  n={rg['n']}")
        out[arm] = {"records": recs, "ragas": rg}
    return out

print("Single-pass orchestrator defined.")

# %% [markdown]
# ## Cell 10 — Execute all 3 runs

# %%
all_runs = []          # list of {arm: {records, ragas}}
t_start = time.perf_counter()
for r in range(1, N_RUNS + 1):
    all_runs.append(run_one_pass(r))
print(f"\n✅ All {N_RUNS} runs done in {(time.perf_counter()-t_start)/60:.1f} min")

# Freeze the caption snapshots (reproducibility + H1 caption review)
cap_snap = {cid: {"vlm_model": VLM_MODEL, "caption": v.get("caption", ""),
                  "etiology": v.get("etiology", {}), "depth": v.get("depth", {}),
                  "time_crossvalidation": v.get("time_crossvalidation", ""),
                  "urgency_flags": v.get("urgency_flags", []), "error": v.get("error", "")}
            for cid, v in _CAPTION_CACHE.items()}
with open(RESULTS_DIR / "G4A_vlm_captions.json", "w", encoding="utf-8") as f:
    json.dump(cap_snap, f, indent=2, ensure_ascii=False)
print(f"   Froze {len(cap_snap)} VLM caption snapshots → G4A_vlm_captions.json")

# %% [markdown]
# ## Cell 11 — Aggregate per arm (mean ± SD across 3 runs)

# %%
def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return 0.0, 0.0
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0)

agg = {}
for arm in ARMS:
    fa  = [all_runs[r][arm]["ragas"]["faithfulness"]     for r in range(N_RUNS)]
    ar  = [all_runs[r][arm]["ragas"]["answer_relevancy"] for r in range(N_RUNS)]
    # safety pass-rate per run
    sp  = [sum(1 for x in all_runs[r][arm]["records"] if x["safety_overall"] == "PASS")
           / len(all_runs[r][arm]["records"]) for r in range(N_RUNS)]
    lat = [statistics.mean([x["gen_latency_ms"] for x in all_runs[r][arm]["records"]]) for r in range(N_RUNS)]
    cost= [sum(x["total_cost_usd"] for x in all_runs[r][arm]["records"]) for r in range(N_RUNS)]
    agg[arm] = {
        "FA":          _mean_sd(fa),
        "AR":          _mean_sd(ar),
        "SafetyPass":  _mean_sd(sp),
        "gen_latency_ms": _mean_sd(lat),
        "total_cost_usd_per_run": _mean_sd(cost),
        "fa_runs": fa, "ar_runs": ar, "safety_runs": [round(s, 4) for s in sp],
    }

print(f"\n{'='*72}\n  G4-A RESULTS — mean ± SD over {N_RUNS} runs ({len(testset)} imaged cases)\n{'='*72}")
print(f"  {'Arm':<6}{'FA':<18}{'AR':<18}{'Safety Pass':<16}{'Latency(ms)':<14}")
for arm in ARMS:
    a = agg[arm]
    print(f"  {arm:<6}"
          f"{a['FA'][0]:.4f}±{a['FA'][1]:.4f}   "
          f"{a['AR'][0]:.4f}±{a['AR'][1]:.4f}   "
          f"{a['SafetyPass'][0]*100:.1f}%±{a['SafetyPass'][1]*100:.1f}   "
          f"{a['gen_latency_ms'][0]:.0f}")

# %% [markdown]
# ## Cell 12 — The headline: A1 − A0 deltas (does the caption help?)

# %%
def _delta(arm1, arm0, key):
    return round(agg[arm1][key][0] - agg[arm0][key][0], 4)

dFA  = _delta("A1", "A0", "FA")
dAR  = _delta("A1", "A0", "AR")
dSAF = _delta("A1", "A0", "SafetyPass")
print(f"\n{'='*60}\n  G4-A HEADLINE  (A1 with caption  −  A0 no caption)\n{'='*60}")
print(f"  Δ Faithfulness   : {dFA:+.4f}  ({dFA*100:+.2f} pp)")
print(f"  Δ Answer-Relevance: {dAR:+.4f}  ({dAR*100:+.2f} pp)")
print(f"  Δ Safety Pass    : {dSAF*100:+.2f} pp")
print(f"\n  Interpretation:")
print(f"   - Positive ΔFA/ΔAR ⇒ the VLM caption improves grounded recommendation quality.")
print(f"   - A small/negative global ΔFA is EXPECTED if the caption adds visual claims absent")
print(f"     from the KB chunks — look at the per-category breakdown (Cell 13) where the caption")
print(f"     is designed to help (adversarial Cat G + cavity Cat D).")

# %% [markdown]
# ## Cell 13 — Per-category A0 vs A1 (the honest, decisive view)
#
# The caption should help most where the CV labels are wrong/incomplete. We average per-sample FA
# per category across runs for each arm and report the delta.

# %%
def per_category_fa():
    # collect per-sample FA aligned to case order, per arm, averaged across runs
    cat_of = {tc["case_id"]: tc["category"] for tc in testset}
    order  = [tc["case_id"] for tc in testset]
    out = {arm: defaultdict(list) for arm in ARMS}
    for arm in ARMS:
        for r in range(N_RUNS):
            recs = all_runs[r][arm]["records"]
            ps   = all_runs[r][arm]["ragas"]["per_sample_fa"]
            # per_sample_fa aligns with valid (non-ERROR) records in order
            valid = [rec for rec in recs if not rec["answer"].startswith("ERROR")]
            for rec, fa in zip(valid, ps):
                out[arm][cat_of[rec["case_id"]]].append(fa)
    cats = sorted({tc["category"] for tc in testset})
    print(f"\n  Per-category Faithfulness (mean over runs×cases)")
    print(f"  {'Cat':<5}{'A0 (no cap)':<14}{'A1 (caption)':<14}{'Δ (A1−A0)':<12}")
    rows = {}
    for c in cats:
        a0 = round(statistics.mean(out['A0'][c]), 4) if out['A0'][c] else 0.0
        a1 = round(statistics.mean(out['A1'][c]), 4) if out['A1'][c] else 0.0
        rows[c] = {"A0": a0, "A1": a1, "delta": round(a1 - a0, 4)}
        flag = "  ← caption helps" if a1 - a0 > 0.01 else ("  ← caption hurts" if a1 - a0 < -0.01 else "")
        print(f"  {c:<5}{a0:<14.4f}{a1:<14.4f}{a1-a0:<+12.4f}{flag}")
    return rows

per_cat = per_category_fa()
print("\n  NOTE: Cat G (adversarial label↔image) and Cat D (cavity) are where a positive Δ is the")
print("  cleanest evidence of the caption's value — the structured labels are wrong/incomplete there.")

# %% [markdown]
# ## Cell 14 — VLM cross-validation signal (qualitative, for the discussion chapter)
#
# Where did the VLM caption disagree with the CV labels? These are the cases the caption is meant to
# catch (the FYP2 clinical contribution). Listed for the write-up + Ms Saw's H1 caption review.

# %%
print("  VLM ↔ CV cross-validation notes (A1 captions):")
for cid, v in cap_snap.items():
    xv = (v.get("time_crossvalidation") or "").strip()
    uf = v.get("urgency_flags") or []
    if re.search(r"discrepan|however|but |not match|disagree|spreading|missed|infect", xv, re.I) or uf:
        print(f"   • {cid}: flags={uf} | {xv[:140]}")

# %% [markdown]
# ## Cell 15 — Save results + summary

# %%
for arm in ARMS:
    recs = [rec for r in range(N_RUNS) for rec in all_runs[r][arm]["records"]]
    with open(RESULTS_DIR / f"{EXP_ID}_{arm}_results_all.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"{EXP_ID}_{arm}_ragas.json", "w", encoding="utf-8") as f:
        json.dump({"faithfulness": agg[arm]["FA"], "answer_relevancy": agg[arm]["AR"],
                   "fa_runs": agg[arm]["fa_runs"], "ar_runs": agg[arm]["ar_runs"]}, f, indent=2)

summary = {
    "experiment": "G4-A — VLM caption vs no-caption",
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_runs": N_RUNS, "n_cases": len(testset),
    "generation_llm": GEN_LLM, "vlm_model": VLM_MODEL,
    "ragas_judge": {"llm": RAGAS_LLM_MODEL, "embed": RAGAS_EMBED_MODEL},
    "fixed": {"retrieval": "R1-C dense k=6", "embedding": "BGE v5", "prompt": "G1-F patient schema"},
    "arms": {arm: {"label": ARM_CONFIG[arm]["label"],
                   "FA_mean": agg[arm]["FA"][0], "FA_sd": agg[arm]["FA"][1],
                   "AR_mean": agg[arm]["AR"][0], "AR_sd": agg[arm]["AR"][1],
                   "SafetyPass_mean": agg[arm]["SafetyPass"][0], "SafetyPass_sd": agg[arm]["SafetyPass"][1],
                   "gen_latency_ms_mean": agg[arm]["gen_latency_ms"][0],
                   "total_cost_usd_per_run_mean": agg[arm]["total_cost_usd_per_run"][0]}
             for arm in ARMS},
    "delta_A1_minus_A0": {"FA": dFA, "AR": dAR, "SafetyPass": dSAF},
    "per_category_faithfulness": per_cat,
    "skipped_cases_no_image": skipped,
}
with open(RESULTS_DIR / "G4A_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# flat per-case CSV
try:
    import pandas as pd
    rows = [rec for r in range(N_RUNS) for arm in ARMS for rec in all_runs[r][arm]["records"]]
    pd.DataFrame([{k: rec[k] for k in ("run","arm","case_id","category","wound_type_expected",
                  "wound_type_predicted","image_provided","vlm_etiology","wound_depth_final",
                  "dfu_flag","safety_overall","total_cost_usd","gen_latency_ms","vlm_latency_ms")}
                  for rec in rows]).to_csv(RESULTS_DIR / "G4A_per_case.csv", index=False)
    print("Wrote G4A_per_case.csv")
except Exception as e:
    print(f"(CSV skipped: {e})")

print("\n✅ Saved: G4A_summary.json · G4A_{A0,A1}_results_all.json · G4A_{A0,A1}_ragas.json · "
      "G4A_vlm_captions.json")
print(f"\n  FINAL  →  A0 FA={agg['A0']['FA'][0]:.4f}  A1 FA={agg['A1']['FA'][0]:.4f}  (Δ {dFA:+.4f}) | "
      f"A0 AR={agg['A0']['AR'][0]:.4f}  A1 AR={agg['A1']['AR'][0]:.4f}  (Δ {dAR:+.4f})")
