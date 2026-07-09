# %% [markdown]
# # VerdaSense Ablation — Stage FYP2 / Experiment G4-B: VLM Comparison — CLOSED-SOURCE (Multi-Run)
#
# **Research question:** *Among proprietary vision models, does the choice matter?* Holding the
# **blind prompt** (the G4-P winner) and everything else fixed, we swap only the VLM.
#
# | Arm | VLM | Provider |
# |-----|-----|----------|
# | **B1** | `gpt-4o-mini` Vision | OpenAI |
# | **B2** | `gemini-2.5-flash` Vision | Google |
#
# (Open-source VLMs via OpenRouter are a **separate** experiment — see **G4-C**. G4-B stays
# closed-source-only so the two are not mixed.)
#
# **Fixed (so only the VLM changes):** VLM prompt = **blind / independent** (G4-P's P4 winner) ·
# Gen LLM = `gpt-4o-mini` · retrieval R1-C k=6 BGE-v5 · prompt G1-F patient schema ·
# RAGAS judge = `gpt-4o-mini` + `text-embedding-3-small`.
# **Runs:** 3. Captions are generated **fresh each run** (VLM reads are non-deterministic, so the
# 3-run *rate* is the honest measure). 34 cases × 2 VLMs × 3 runs = **204 VLM calls + 204 generations**.
#
# **Metrics (per VLM):**
#   - **Caption Infection-Accuracy** — VLM's visual infection read vs the (correct) CV label on the
#     27 non-adversarial cases (higher = better perception).
#   - **Tissue-bucket accuracy** — non-viable ≥25% vs <25% match.
#   - **Discrepancy-Detection Rate (VLM-DISC)** — on the 7 adversarial (cat G) cases, does the VLM
#     *disagree* with the wrong label on the axis that is actually wrong (infection OR tissue)?
#   - **Refusal rate** — fraction of images the VLM declined to read (empty/blocked responses).
#   - **FA / AR** downstream (RAGAS, 3 runs, mean ± SD) · **Safety** · **Cost / Latency** per VLM.
#
# **Headline contrast: B1 (GPT-4o-mini-V) vs B2 (Gemini-2.5-Flash-V).**
#
# > Note: `gemini-2.5-flash` is ~2× the input / ~4× the output price of `gpt-4o-mini` (per the app's
# > `VLM_REGISTRY`), so B2 must *earn* its cost in accuracy/detection to justify a switch.

# %% [markdown]
# ## Cell 0 — Environment + import the production pipeline

# %%
import os, sys, re, json, time, base64, statistics, warnings, datetime
from pathlib import Path
from collections import Counter, defaultdict

import torch
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv()

def _find_root(start):
    start = Path(start).resolve()
    for cand in [start, *start.parents]:
        if (cand / "wound_app_multimodal.py").exists():
            return cand
    return start
PROJECT_ROOT = _find_root(Path(__file__).parent if "__file__" in dir() else Path.cwd())
NOTEBOOK_DIR = PROJECT_ROOT / "RAGAS_EVAL" / "G4B_VLM_Comparison"
RESULTS_DIR  = NOTEBOOK_DIR / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TESTSET_PATH = PROJECT_ROOT / "ragas_testset" / "wound_testset_v5.json"

sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
print(f"Project root : {PROJECT_ROOT}\nImporting wound_app_multimodal (BGE + v5 KB, ~15–30 s)…")
import wound_app_multimodal as mm

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

print(f"✅ Pipeline imported. v5 KB chunks: {mm.db._collection.count()}")

# %% [markdown]
# ## Cell 1 — Configuration

# %%
EXP_ID   = "G4B"
N_RUNS   = 3
VARIANTS = ["B1", "B2"]
VLM_OF   = {"B1": "gpt-4o-mini", "B2": "gemini-2.5-flash"}   # closed-source only
GEN_LLM  = "gpt-4o-mini"                                     # generation model — fixed
RAGAS_LLM_MODEL, RAGAS_EMBED_MODEL = "gpt-4o-mini", "text-embedding-3-small"
CACHE_CAPTIONS = False

VARIANT_LABEL = {
    "B1": "GPT-4o-mini Vision (OpenAI)",
    "B2": "Gemini-2.5-Flash Vision (Google)",
}
print("G4-B configuration (closed-source VLM comparison under the fixed BLIND prompt)")
for v in VARIANTS: print(f"  {v}: {VARIANT_LABEL[v]}  (vlm_model={VLM_OF[v]})")
print(f"  Gen={GEN_LLM} · runs={N_RUNS} · judge={RAGAS_LLM_MODEL}+{RAGAS_EMBED_MODEL}")
for v in VARIANTS: assert VLM_OF[v] in mm.VALID_VLMS, f"{VLM_OF[v]} not a valid VLM"
assert GEN_LLM in mm.VALID_MODELS

# %% [markdown]
# ## Cell 2 — Load v5 testset (imaged cases) + mark adversarial

# %%
full = json.load(open(TESTSET_PATH, encoding="utf-8"))
testset = [tc for tc in full if tc.get("image_ref") and (PROJECT_ROOT / tc["image_ref"]).exists()]
for tc in testset:
    tc["_adversarial"] = (tc["category"] == "G")
n_adv = sum(tc["_adversarial"] for tc in testset)
print(f"{len(testset)} imaged cases · {n_adv} adversarial (cat G) · {len(testset)-n_adv} non-adversarial")
print(f"  categories: {dict(Counter(tc['category'] for tc in testset))}")

# %% [markdown]
# ## Cell 3 — RAGAS judge (fixed)

# %%
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0))
ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBED_MODEL))
print(f"RAGAS judge: {RAGAS_LLM_MODEL} + {RAGAS_EMBED_MODEL}")

# %% [markdown]
# ## Cell 4 — Shared pipeline (identical to G4-A / G4-P / the app up to retrieval)

# %%
def _img_to_b64(image_ref):
    p = PROJECT_ROOT / image_ref
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/png")
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime

def build_case_inputs(tc):
    tp = tc["time_payload"]
    prof = mm.interpret_tissue_percentages(tp["necrotic_pct"], tp["slough_pct"], tp["granulation_pct"])
    inf, moi, edg = mm.normalize_infection(tp["infection"]), mm.normalize_moisture(tp["moisture"]), mm.normalize_edge(tp["edge"])
    notes = tp.get("notes", "") or ""
    diabetic = bool(tc.get("demographics", {}).get("diabetic", False))
    diabetes = "yes" if diabetic else "no"
    depth_self = "cavity" if tc.get("wound_depth") == "cavity" else "surface"
    notes_rules = (notes + " | Patient reports diabetes.").strip(" |") if (diabetes == "yes" and "diabet" not in notes.lower()) else notes
    cls = mm.classify_wound(prof, inf, moi, notes_rules)
    nq  = mm.build_narrative_query(prof, inf, moi, edg, notes)
    chunks, _ = mm.retrieve_chunks_multiaxis(narrative_query=nq, tissue_profile=prof, infection_norm=inf,
                                             moisture_norm=moi, classifier=cls, notes=notes, top_n=6)
    ordered, _ = mm.order_sources(chunks, cls["wound_type"])
    assessment = (
        f"T.I.M.E. WOUND ASSESSMENT (from CV pipeline):\n"
        f"T (Tissue)    : {prof['clinical_tissue']} — Necrotic {prof['necrotic_pct']}%, "
        f"Slough {prof['slough_pct']}%, Granulation {prof['granulation_pct']}% "
        f"(non-viable {prof['non_viable_pct']}%)\n"
        f"I (Infection) : {inf}\nM (Moisture)  : {moi}\nE (Edge)      : {edg}\n"
        f"Rule-based wound type: {cls['wound_type']} "
        f"(referral={cls['referral_required']}, antibiotic={cls['antibiotic_required']})"
    )
    if notes.strip(): assessment += f"\nPatient notes: {notes.strip()}"
    return {"prof": prof, "inf": inf, "moi": moi, "edg": edg, "notes": notes, "diabetes": diabetes,
            "depth_self": depth_self, "classifier": cls, "narrative_query": nq, "ordered_chunks": ordered,
            "retrieved_contexts": [c.metadata.get("raw_text", c.page_content) for c in ordered],
            "assessment_text": assessment}
print("Shared pipeline helper defined.")

# %% [markdown]
# ## Cell 5 — The fixed BLIND VLM prompt (G4-P P4 winner) — identical for BOTH VLMs

# %%
SCHEMA = (
    'Return ONLY a JSON object (no markdown) with EXACTLY these keys:\n'
    '{"caption": "<short clinical caption>", '
    '"tissue": {"necrotic_pct": <int>, "slough_pct": <int>, "granulation_pct": <int>}, '
    '"infection": "Infected" | "Not infected" | "Undetermined", '
    '"moisture": "Dry" | "Moderate" | "High" | "Undetermined", '
    '"depth": "superficial" | "cavity" | "undetermined", '
    '"urgency_flags": ["<short visual red-flags; [] if none>"], '
    '"note": "<one line of your visual reasoning>"}'
)
VLM_SYSTEM_BLIND = ("You are an INDEPENDENT wound assessor. Assess this wound photograph on its own "
    "merits. You are NOT given any prior CV labels — estimate the tissue percentages, infection "
    "status, moisture level and depth PURELY from what you see in the image. Be objective and do not "
    "assume the wound is healthy; if you see signs of infection (erythema, pus, slough, odour cues), "
    "report 'Infected'.\n\n" + SCHEMA)

def vlm_human_blind(demo_text):
    return (f"Assess this wound photograph independently. Patient context: {demo_text}. "
            f"Estimate tissue %, infection, moisture and depth from the image alone, and return the JSON.")

print("Fixed blind VLM prompt defined (applied identically to B1 and B2).")

# %% [markdown]
# ## Cell 6 — Variant caption generator (same prompt, swap the VLM model)

# %%
_CAP_CACHE = {}

def generate_variant_caption(variant, tc, inputs):
    key = (variant, tc["case_id"])
    if CACHE_CAPTIONS and key in _CAP_CACHE:
        return _CAP_CACHE[key]
    vlm_model = VLM_OF[variant]
    demo_bits = ["patient reports diabetes" if inputs["diabetes"] == "yes" else "no diabetes reported"]
    if inputs["depth_self"] not in ("unknown", ""):
        demo_bits.append(f"patient says wound looks '{inputs['depth_self']}'")
    demo = "; ".join(demo_bits)
    out = {"variant": variant, "vlm_model": vlm_model, "error": "", "caption": "",
           "infection": "Undetermined", "moisture": "Undetermined", "depth": "undetermined",
           "tissue": {}, "urgency_flags": [], "note": "",
           "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0}
    try:
        b64, mime = _img_to_b64(tc["image_ref"])
        vlm = mm.make_llm(vlm_model)
        msgs = [SystemMessage(content=VLM_SYSTEM_BLIND),
                HumanMessage(content=[{"type": "text", "text": vlm_human_blind(demo)},
                                      {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}])]
        t0 = time.perf_counter()
        resp = vlm.invoke(msgs)
        raw = mm._strip_thinking(resp.content if isinstance(resp.content, str) else str(resp.content))
        p = mm._parse_vlm_json(raw)
        inp, otk = mm._extract_tokens(resp)
        out.update({"caption": p.get("caption", "") or raw[:600],
                    "infection": p.get("infection", "Undetermined"),
                    "moisture": p.get("moisture", "Undetermined"),
                    "depth": p.get("depth", "undetermined"),
                    "tissue": p.get("tissue", {}) or {},
                    "urgency_flags": p.get("urgency_flags", []) or [],
                    "note": p.get("note", ""),
                    "input_tokens": inp, "output_tokens": otk,
                    "cost_usd": round(mm._compute_cost(mm.VLM_REGISTRY, vlm_model, inp, otk), 8),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "error": "" if p else "non-JSON; raw used"})
    except Exception as e:
        out["error"] = str(e)
    if CACHE_CAPTIONS: _CAP_CACHE[key] = out
    return out
print("Variant caption generator defined (blind prompt fixed; VLM model swapped per arm).")

# %% [markdown]
# ## Cell 7 — Caption metrics: infection accuracy · tissue bucket · discrepancy detection
# (identical scoring to G4-P so the experiments are directly comparable)

# %%
def _norm_inf(s):
    s = (s or "").lower()
    if "not" in s or s in ("no", "none", "absent"): return "Not infected"
    if "infect" in s: return "Infected"
    return "Undetermined"

def caption_metrics(cap, tc):
    label_inf = _norm_inf(tc["time_payload"]["infection"])
    vlm_inf   = _norm_inf(cap.get("infection"))
    lab_nv = tc["time_payload"]["necrotic_pct"] + tc["time_payload"]["slough_pct"]
    t = cap.get("tissue", {}) or {}
    vlm_nv = (t.get("necrotic_pct", 0) or 0) + (t.get("slough_pct", 0) or 0)
    tissue_bucket_ok = ((lab_nv >= 25) == (vlm_nv >= 25)) if t else None
    adv = tc["_adversarial"]
    inf_disagree    = (vlm_inf != label_inf and vlm_inf != "Undetermined")
    tissue_disagree = (tissue_bucket_ok is False)
    axis = None
    if adv:
        parts = (["infection"] if inf_disagree else []) + (["tissue"] if tissue_disagree else [])
        axis = "+".join(parts) if parts else "none"
    return {
        "label_infection": label_inf, "vlm_infection": vlm_inf,
        "label_nv": lab_nv, "vlm_nv": vlm_nv,
        "infection_correct": (vlm_inf == label_inf) if (not adv and vlm_inf != "Undetermined") else None,
        "discrepancy_detected": (inf_disagree or tissue_disagree) if adv else None,
        "discrepancy_axis": axis,
        "tissue_bucket_ok": tissue_bucket_ok,
        "flagged_in_note": bool(re.search(r"disagree|discrepan|however|but |spreading|infect|mismatch|necro|slough",
                                          cap.get("note", ""), re.I)),
    }
print("Caption-metric functions defined.")

# %% [markdown]
# ## Cell 8 — Downstream generation + clinical check (negation-safe) + RAGAS

# %%
DRESSING_ALIASES = {
    "film": ["film","transparent film"], "hydrocolloid": ["hydrocolloid","renocare"], "foam": ["foam","renofoam"],
    "tulle": ["tulle","paraffin"], "hydrogel": ["hydrogel","dermacyn"], "alginate": ["alginate","alginogel","flaminal"],
    "alginogel": ["alginogel","flaminal"], "hydrofiber": ["hydrofibre","hydrofiber","aquacel"], "silver": ["silver"],
    "iodine": ["iodine","povidone","cadexomer"], "charcoal": ["charcoal","activated carbon","zorflex"],
    "polymeric_membrane": ["polymeric membrane","polymem"], "hydroconductive": ["hydroconductive","drawtex"],
    "npwt": ["npwt","negative pressure","vacuum"], "silicone_foam": ["silicone foam","silicone-coated foam"],
    "compression": ["compression"]}
def _forms(t): return DRESSING_ALIASES.get(t, [t.replace("_"," ")])
_POS = re.compile(r"^##\s*(dressing you need|example products|step-?by-?step)", re.I|re.M)
_REC = re.compile(r"^##\s*(dressing you need|example products)", re.I|re.M)
_AV  = re.compile(r"^##\s*(dressings to avoid)", re.I|re.M)
def _section(a, header_re):
    keep, out = False, []
    for ln in a.split("\n"):
        s = ln.strip()
        if s.startswith("##"): keep = bool(header_re.match(s)) and not _AV.match(s); continue
        if keep: out.append(ln.lower())
    return " ".join(out)
def _postxt(a): t=_section(a,_POS); return t if t else a.lower()
def _rectxt(a): t=_section(a,_REC); return t if t else a.lower()
_NEG_RE = re.compile(r"\b(no|not|never|avoid|without|don'?t|isn'?t|aren'?t|nor)\b[\w\s,/()-]{0,22}$")
def _neg_before(text, idx): return bool(_NEG_RE.search(text[max(0, idx-32):idx]))  # negation just before token?
def _rec(tok, a):     return any(f in _postxt(a) for f in _forms(tok))
def _rec_only(tok, a):                                                    # contraindicated check (negation-aware)
    t = _rectxt(a)
    for f in _forms(tok):
        for m in re.finditer(re.escape(f), t):
            if not _neg_before(t, m.start()): return True                 # positive mention (not "without <tok>")
    return False

def check_clinical(a, tc):
    if not a or a.startswith("ERROR"): return {"overall": "FAIL"}
    r = {}
    for c in tc.get("contraindicated_dressings", []):
        base = c.split("(")[0].strip(); r[f"avoid_{base}"] = {"result": "FAIL" if _rec_only(base, a) else "PASS"}
    r["allowed_present"] = {"result": "PASS" if any(_rec(t, a) for t in tc.get("allowed_dressings", [])) else "FAIL"}
    lo = a.lower()
    if tc.get("antibiotic_required"): r["antibiotic"] = {"result": "PASS" if any(k in lo for k in ["antibiotic","swab","culture","antimicrobial"]) else "FAIL"}
    if tc.get("referral_required"):   r["referral"]   = {"result": "PASS" if any(k in lo for k in ["see a doctor","refer","hospital","urgent","specialist","clinic"]) else "FAIL"}
    r["overall"] = "FAIL" if any(v.get("result")=="FAIL" for v in r.values() if isinstance(v,dict)) else "PASS"
    return r

def vlm_block_from_caption(cap):
    if cap.get("error") and not cap.get("caption"):
        return "\n(No usable wound image — grounded in CV labels + guidelines only.)\n"
    uf = cap.get("urgency_flags") or []
    return ("\nVLM VISUAL ASSESSMENT (direct observation of the wound photo — personalise 'Your Wound', "
            "cross-check the CV labels, inform urgency; do NOT cite as [S#]):\n"
            f"- Caption: {cap.get('caption','')}\n"
            f"- Visual infection read: {cap.get('infection','')}\n"
            f"- Visual moisture: {cap.get('moisture','')}\n"
            f"- Depth: {cap.get('depth','')}\n"
            f"- Note: {cap.get('note','')}\n"
            f"- Visual urgency flags: {', '.join(uf) if uf else 'none noted'}\n")

def run_ragas(qs, ctxs, ans, refs):
    S = [SingleTurnSample(user_input=q, retrieved_contexts=[str(c) for c in ct], response=a, reference=r)
         for q, ct, a, r in zip(qs, ctxs, ans, refs) if not a.startswith("ERROR")]
    if not S: return {"faithfulness": 0.0, "answer_relevancy": 0.0, "per_sample_fa": [], "n": 0}
    res = evaluate(EvaluationDataset(S), metrics=[Faithfulness(llm=ragas_llm),
                    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embed)])
    df = res.to_pandas()
    fa = next((c for c in df.columns if "faithfulness" in c.lower()), None)
    ar = next((c for c in df.columns if "answer_relevancy" in c.lower()), None)
    m = lambda s: round(sum(s.dropna())/max(1, len(s.dropna())), 4) if s.dropna().tolist() else 0.0
    return {"faithfulness": m(df[fa]) if fa else 0.0, "answer_relevancy": m(df[ar]) if ar else 0.0,
            "per_sample_fa": df[fa].tolist() if fa else [], "n": len(S)}
print("Downstream generation + negation-safe clinical checker + RAGAS defined.")

# %% [markdown]
# ## Cell 9 — One full pass over both VLMs × all cases

# %%
def run_one_pass(run_idx):
    print(f"\n{'='*70}\n  G4-B — RUN {run_idx}/{N_RUNS}\n{'='*70}")
    per_variant = {v: [] for v in VARIANTS}
    for i, tc in enumerate(testset):
        inputs = build_case_inputs(tc)
        print(f"  [{i+1:>2}/{len(testset)}] {tc['case_id']:<34}", end="")
        for v in VARIANTS:
            cap = generate_variant_caption(v, tc, inputs)
            cm  = caption_metrics(cap, tc)
            depth_final = mm.resolve_wound_depth(cap.get("depth", "undetermined"), inputs["depth_self"])
            depth_block = f"\nWOUND DEPTH (resolved): {depth_final}\n"
            msgs = mm.build_patient_messages(inputs["ordered_chunks"], inputs["assessment_text"],
                        inputs["narrative_query"], inputs["classifier"],
                        vlm_block_from_caption(cap), depth_block, GEN_LLM)
            llm = mm.make_llm(GEN_LLM); t0 = time.perf_counter()
            try:
                resp = llm.invoke(msgs); answer = mm._strip_thinking(resp.content); gi, go = mm._extract_tokens(resp)
            except Exception as e:
                answer, gi, go = f"ERROR: {e}", 0, 0
            safety = check_clinical(answer, tc)
            per_variant[v].append({
                "run": run_idx, "variant": v, "vlm_model": VLM_OF[v], "case_id": tc["case_id"],
                "category": tc["category"], "adversarial": tc["_adversarial"],
                "narrative_query": inputs["narrative_query"], "reference": tc.get("reference", ""),
                "retrieved_contexts": inputs["retrieved_contexts"], "answer": answer,
                "caption": cap.get("caption", ""), "vlm_infection": cm["vlm_infection"],
                "label_infection": cm["label_infection"], "caption_metrics": cm,
                "vlm_error": cap.get("error", ""), "safety_overall": safety.get("overall", "N/A"),
                "vlm_cost_usd": cap.get("cost_usd", 0.0), "vlm_latency_ms": cap.get("latency_ms", 0.0),
                "gen_cost_usd": round(mm._compute_cost(mm.MODEL_REGISTRY, GEN_LLM, gi, go), 8),
                "gen_latency_ms": round((time.perf_counter()-t0)*1000, 1)})
            print(f" | {v}", end="")
        print(flush=True); time.sleep(0.5)
    out = {}
    for v in VARIANTS:
        recs = per_variant[v]
        rg = run_ragas([r["narrative_query"] for r in recs], [r["retrieved_contexts"] for r in recs],
                       [r["answer"] for r in recs], [r["reference"] for r in recs])
        print(f"  RAGAS {v}: FA={rg['faithfulness']:.4f} AR={rg['answer_relevancy']:.4f} n={rg['n']}")
        out[v] = {"records": recs, "ragas": rg}
    return out
print("Single-pass orchestrator defined.")

# %% [markdown]
# ## Cell 10 — Execute 3 runs (fresh captions each run)

# %%
all_runs, t0 = [], time.perf_counter()
for r in range(1, N_RUNS + 1):
    all_runs.append(run_one_pass(r))
n_caps = N_RUNS * len(VARIANTS) * len(testset)
print(f"\n✅ {N_RUNS} runs in {(time.perf_counter()-t0)/60:.1f} min · "
      f"{n_caps} captions generated fresh (2 VLMs × {len(testset)} cases × {N_RUNS} runs)")

cap_dump = [{"run": r+1, "variant": rec["variant"], "vlm_model": rec["vlm_model"], "case_id": rec["case_id"],
             "vlm_infection": rec["vlm_infection"], "label_infection": rec["label_infection"],
             "caption": rec["caption"], "error": rec["vlm_error"]}
            for r in range(N_RUNS) for v in VARIANTS for rec in all_runs[r][v]["records"]]
json.dump(cap_dump, open(RESULTS_DIR / "G4B_captions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"   Froze {len(cap_dump)} captions → G4B_captions.json")

# %% [markdown]
# ## Cell 11 — Caption quality per VLM (infection accuracy · tissue bucket · VLM-DISC · refusals)

# %%
cap_stats = {}
for v in VARIANTS:
    inf_ok, disc, tiss, errs = [], [], [], 0
    for r in range(N_RUNS):
        for rec in all_runs[r][v]["records"]:
            cm = rec["caption_metrics"]
            if rec["vlm_error"]: errs += 1
            if cm["infection_correct"]    is not None: inf_ok.append(cm["infection_correct"])
            if cm["discrepancy_detected"] is not None: disc.append(cm["discrepancy_detected"])
            if cm["tissue_bucket_ok"]     is not None: tiss.append(cm["tissue_bucket_ok"])
    cap_stats[v] = {
        "infection_accuracy": round(sum(inf_ok)/len(inf_ok), 3) if inf_ok else None,
        "discrepancy_detection": round(sum(disc)/len(disc), 3) if disc else None,
        "tissue_bucket_acc": round(sum(tiss)/len(tiss), 3) if tiss else None,
        "n_nonadv": len(inf_ok), "n_adv": len(disc), "vlm_errors": errs}

print(f"\n{'='*82}\n  G4-B CAPTION QUALITY per VLM\n{'='*82}")
print(f"  {'Arm':<5}{'VLM':<26}{'Infection acc':<16}{'Tissue-bucket':<15}{'VLM-DISC':<14}{'refusals'}")
for v in VARIANTS:
    s = cap_stats[v]
    ia = f"{s['infection_accuracy']*100:.0f}% (n={s['n_nonadv']})" if s['infection_accuracy'] is not None else "—"
    tb = f"{s['tissue_bucket_acc']*100:.0f}%" if s['tissue_bucket_acc'] is not None else "—"
    dd = f"{s['discrepancy_detection']*100:.0f}% (n={s['n_adv']})" if s['discrepancy_detection'] is not None else "—"
    print(f"  {v:<5}{VLM_OF[v]:<26}{ia:<16}{tb:<15}{dd:<14}{s['vlm_errors']}")

# %% [markdown]
# ## Cell 12 — Downstream FA / AR / Safety + Cost / Latency per VLM

# %%
def _ms(xs): xs=[x for x in xs if x is not None]; return (round(statistics.mean(xs),4), round(statistics.stdev(xs),4) if len(xs)>1 else 0.0) if xs else (0.0,0.0)
agg = {}
for v in VARIANTS:
    fa = [all_runs[r][v]["ragas"]["faithfulness"] for r in range(N_RUNS)]
    ar = [all_runs[r][v]["ragas"]["answer_relevancy"] for r in range(N_RUNS)]
    sp = [sum(1 for x in all_runs[r][v]["records"] if x["safety_overall"]=="PASS")/len(all_runs[r][v]["records"]) for r in range(N_RUNS)]
    vlm_cost = [sum(x["vlm_cost_usd"] for x in all_runs[r][v]["records"]) for r in range(N_RUNS)]
    vlm_lat  = [statistics.mean([x["vlm_latency_ms"] for x in all_runs[r][v]["records"]]) for r in range(N_RUNS)]
    agg[v] = {"FA": _ms(fa), "AR": _ms(ar), "Safety": _ms(sp), "fa_runs": fa, "ar_runs": ar,
              "vlm_cost_per_run": _ms(vlm_cost), "vlm_latency_ms": _ms(vlm_lat)}
print(f"\n{'='*84}\n  G4-B DOWNSTREAM + COST (mean ± SD, {N_RUNS} runs, {len(testset)} cases)\n{'='*84}")
print(f"  {'Arm':<5}{'VLM':<24}{'FA':<16}{'AR':<16}{'Safety':<11}{'VLM $/run':<11}{'VLM ms'}")
for v in VARIANTS:
    a = agg[v]
    print(f"  {v:<5}{VLM_OF[v]:<24}{a['FA'][0]:.4f}±{a['FA'][1]:.4f}  {a['AR'][0]:.4f}±{a['AR'][1]:.4f}  "
          f"{a['Safety'][0]*100:.1f}%     ${a['vlm_cost_per_run'][0]:.4f}    {a['vlm_latency_ms'][0]:.0f}")

# %% [markdown]
# ## Cell 13 — HEADLINE: B1 vs B2

# %%
NTOT = len(testset) * N_RUNS
def f(x): return "—" if x is None else f"{x*100:.0f}%"
W = 32 + 16 * len(VARIANTS)
print(f"\n{'='*W}\n  G4-B HEADLINE — closed-source VLM comparison (blind prompt fixed)\n{'='*W}")
print("  " + f"{'Metric':<30}" + "".join(f"{v:<16}" for v in VARIANTS))
def row(label, fn): print("  " + f"{label:<30}" + "".join(f"{fn(v):<16}" for v in VARIANTS))
row("Refusal rate", lambda v: f"{100*cap_stats[v]['vlm_errors']/NTOT:.0f}% ({cap_stats[v]['vlm_errors']}/{NTOT})")
row("Infection acc (non-adv)", lambda v: f(cap_stats[v]['infection_accuracy']))
row("Tissue-bucket acc", lambda v: f(cap_stats[v]['tissue_bucket_acc']))
row("VLM-DISC (discrepancy)", lambda v: f(cap_stats[v]['discrepancy_detection']))
row("FA (downstream)", lambda v: f"{agg[v]['FA'][0]:.4f}")
row("AR (downstream)", lambda v: f"{agg[v]['AR'][0]:.4f}")
row("VLM cost / run", lambda v: f"${agg[v]['vlm_cost_per_run'][0]:.4f}")
row("VLM latency (ms)", lambda v: f"{agg[v]['vlm_latency_ms'][0]:.0f}")
print("\n  Read: the winner keeps HIGH VLM-DISC + accuracy with LOW refusals + cost.")

# %% [markdown]
# ## Cell 14 — Per-adversarial-case discrepancy per VLM (the VLM-DISC diagnostic)

# %%
adv_cases = [tc for tc in testset if tc["_adversarial"]]
cm_lookup = defaultdict(list)
for r in range(N_RUNS):
    for v in VARIANTS:
        for rec in all_runs[r][v]["records"]:
            cm_lookup[(v, rec["case_id"])].append(rec["caption_metrics"])
print("  Adversarial cases — discrepancy caught across the 3 runs (axis that is wrong):")
for tc in adv_cases:
    exp = (tc.get("escalation_flags_expected") or ["?"])[0]
    print(f"\n  {tc['case_id']}  | CV label infection='{tc['time_payload']['infection']}' "
          f"NV={tc['time_payload']['necrotic_pct']+tc['time_payload']['slough_pct']}%  | expect: {exp[:52]}")
    for v in VARIANTS:
        cms = cm_lookup[(v, tc["case_id"])]
        caught = sum(1 for c in cms if c["discrepancy_detected"])
        print(f"    {v} ({VLM_OF[v]:<18}): caught {caught}/{len(cms)}  axes={[c['discrepancy_axis'] for c in cms]}")

# %% [markdown]
# ## Cell 15 — Save results + summary

# %%
for v in VARIANTS:
    recs = [rec for r in range(N_RUNS) for rec in all_runs[r][v]["records"]]
    json.dump(recs, open(RESULTS_DIR / f"{EXP_ID}_{v}_results_all.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

summary = {
    "experiment": "G4-B — closed-source VLM comparison (blind prompt fixed)",
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_runs": N_RUNS, "n_cases": len(testset), "n_adversarial": n_adv,
    "generation_llm": GEN_LLM, "vlm_prompt": "blind (G4-P P4 winner)",
    "arms": {v: {"vlm_model": VLM_OF[v], "label": VARIANT_LABEL[v],
                 "refusal_rate": round(cap_stats[v]["vlm_errors"] / (len(testset) * N_RUNS), 3),
                 "vlm_errors": cap_stats[v]["vlm_errors"],
                 "infection_accuracy": cap_stats[v]["infection_accuracy"],
                 "tissue_bucket_acc": cap_stats[v]["tissue_bucket_acc"],
                 "discrepancy_detection": cap_stats[v]["discrepancy_detection"],
                 "FA_mean": agg[v]["FA"][0], "FA_sd": agg[v]["FA"][1],
                 "AR_mean": agg[v]["AR"][0], "AR_sd": agg[v]["AR"][1],
                 "Safety_mean": agg[v]["Safety"][0],
                 "vlm_cost_per_run": agg[v]["vlm_cost_per_run"][0],
                 "vlm_latency_ms": agg[v]["vlm_latency_ms"][0]} for v in VARIANTS},
}
json.dump(summary, open(RESULTS_DIR / "G4B_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("✅ Saved: G4B_summary.json · G4B_{B1,B2}_results_all.json · G4B_captions.json")

# %% [markdown]
# ## Cell 16 — How to read G4-B
#
# 1. **VLM-DISC + infection accuracy (Cell 11/13)** is the point: both VLMs use the *same blind
#    prompt*, so any gap is the model, not the framing.
# 2. **Refusal rate** is a first-class metric: a VLM that refuses graphic wound images is unusable
#    regardless of accuracy.
# 3. **Cost/latency (Cell 12/13):** `gemini-2.5-flash` is pricier — it must *earn* the switch with a
#    clear accuracy/detection win, else `gpt-4o-mini` is the better value (and stays the default).
# 4. **FA/AR (Cell 12)** should be similar (retrieval + prompt + gen LLM fixed) — the primary signal
#    is caption *quality/detection*, not downstream FA.
# 5. **Open-source VLMs are the separate G4-C experiment** — keep the two un-mixed.
