# %% [markdown]
# # VerdaSense Ablation — Stage FYP2 / Experiment G4-P: VLM Caption Prompt Strategy (Multi-Run)
#
# **Research question:** *How should the VLM be prompted to caption the wound so the recommendation
# improves — and, critically, so the caption actually **cross-validates** the CV labels instead of
# anchoring to them?*
#
# G4-A showed the caption trades a little Faithfulness for Answer-Relevance, but its headline job
# (catching CV-label errors) **did not happen** — the VLM was *given* the CV labels and echoed them
# (on the adversarial `cat_g` it said "agrees … not infected" even though the image is infected).
# G4-P ablates the **VLM prompt** to find the framing that fixes this.
#
# | Arm | Label | CV labels shown to VLM? | Framing |
# |-----|-------|-------------------------|---------|
# | **P1** | Appearance-only | yes (context) | describe what you see; no cross-validation, no treatment reasoning (R5-style) |
# | **P2** | Current production | yes | T.I.M.E. cross-validation + dressing implications (the shipped `VLM_SYSTEM_PROMPT`) |
# | **P3** | Minimal / terse | yes | 1–2 sentence caption + structured fields only |
# | **P4** | **Blind / independent** | **NO** | assess tissue / infection / moisture / depth **from the image alone** — the anti-anchoring arm |
#
# **Fixed:** VLM = `gpt-4o-mini`-Vision · Gen LLM = `gpt-4o-mini` · retrieval R1-C k=6 BGE-v5 ·
# prompt G1-F patient schema · RAGAS judge = `gpt-4o-mini` + `text-embedding-3-small`.
# **Runs:** 3. Captions are generated **fresh each run** (not cached): the VLM's discrepancy detection
# is non-deterministic, so the 3-run *rate* is the honest measure. ~21 cases × 4 variants × 3 runs ≈
# **250 VLM calls + 250 generations (~$0.6–0.8)**.
#
# **Metrics:**
#   - **Caption Infection-Accuracy** — does the VLM's *visual* infection read match the CV label on
#     the 14 non-adversarial cases (higher = better)?
#   - **Discrepancy-Detection Rate** — on the adversarial case(s), does the VLM's read *disagree*
#     with the wrong CV label (higher = better)? ← the metric FA cannot capture
#   - **Tissue-bucket accuracy** — non-viable ≥25% vs <25% match
#   - **FA / AR** downstream (RAGAS, 3 runs, mean ± SD)
#
# **Headline contrast: P2 (current, given labels) vs P4 (blind).** If P4 lifts discrepancy detection
# without hurting non-adversarial accuracy, the blind prompt is the fix.
#
# The v5 testset now has **7 adversarial cases** (category G) across three discrepancy directions —
# missed infection (2), missed necrosis (2), and CV over-call (3) — so the discrepancy-detection rate is
# over 7 cases × 3 runs = 21 samples. Detection is scored on the axis that is actually wrong (infection
# OR tissue-bucket), so the necrosis cases are credited via tissue, not infection.

# %% [markdown]
# ## Cell 0 — Environment + import the production pipeline

# %%
import os, sys, re, json, time, base64, statistics, warnings, datetime
from pathlib import Path
from collections import Counter, defaultdict

import torch
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv()

# Robustly locate the project root (works whether cwd is the notebook dir, the project root,
# or anywhere in between) by walking up until wound_app_multimodal.py is found.
def _find_root(start):
    start = Path(start).resolve()
    for cand in [start, *start.parents]:
        if (cand / "wound_app_multimodal.py").exists():
            return cand
    return start
PROJECT_ROOT = _find_root(Path(__file__).parent if "__file__" in dir() else Path.cwd())
NOTEBOOK_DIR = PROJECT_ROOT / "RAGAS_EVAL" / "G4P_VLM_Prompt_Strategy"
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
EXP_ID   = "G4P"
N_RUNS   = 3
VARIANTS = ["P1", "P2", "P3", "P4"]
GEN_LLM  = "gpt-4o-mini"
VLM_MODEL= "gpt-4o-mini"
RAGAS_LLM_MODEL, RAGAS_EMBED_MODEL = "gpt-4o-mini", "text-embedding-3-small"
# Generate a FRESH caption each run: the VLM's discrepancy detection is NON-DETERMINISTIC
# (the same image+prompt read "agrees/not infected" in G4-A but "Infected" on retry) — so the
# 3-run RATE, not a single cached caption, is the honest measure. ~180 VLM calls (~$0.3).
CACHE_CAPTIONS = False

VARIANT_LABEL = {
    "P1": "Appearance-only (labels shown, no cross-validation)",
    "P2": "Current production (cross-validation + dressing implications)",
    "P3": "Minimal / terse",
    "P4": "Blind / independent (CV labels NOT shown) — anti-anchoring",
}
print("G4-P configuration")
for v in VARIANTS: print(f"  {v}: {VARIANT_LABEL[v]}")
print(f"  VLM={VLM_MODEL} · Gen={GEN_LLM} · runs={N_RUNS} · judge={RAGAS_LLM_MODEL}+{RAGAS_EMBED_MODEL}")
assert VLM_MODEL in mm.VALID_VLMS and GEN_LLM in mm.VALID_MODELS

# %% [markdown]
# ## Cell 2 — Load v5 testset (imaged cases) + mark adversarial

# %%
full = json.load(open(TESTSET_PATH, encoding="utf-8"))
testset = [tc for tc in full if tc.get("image_ref") and (PROJECT_ROOT / tc["image_ref"]).exists()]
for tc in testset:
    tc["_adversarial"] = (tc["category"] == "G")     # label intentionally wrong vs image
n_adv = sum(tc["_adversarial"] for tc in testset)
print(f"{len(testset)} imaged cases · {n_adv} adversarial (cat G) · "
      f"{len(testset)-n_adv} non-adversarial")
print(f"  categories: {dict(Counter(tc['category'] for tc in testset))}")
if n_adv < 3:
    print("  ⚠ few adversarial cases — Discrepancy-Detection Rate is illustrative until VLM-DISC set is built.")

# %% [markdown]
# ## Cell 3 — RAGAS judge (fixed)

# %%
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0))
ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBED_MODEL))
print(f"RAGAS judge: {RAGAS_LLM_MODEL} + {RAGAS_EMBED_MODEL}")

# %% [markdown]
# ## Cell 4 — Shared pipeline (identical to G4-A / the app up to retrieval)

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
# ## Cell 5 — The four VLM prompt strategies (the core of G4-P)
#
# All variants must return the SAME JSON schema (so the metrics parse uniformly). Only the framing
# and whether the CV labels are shown change. P4 is *blind* — it never sees the CV labels.

# %%
SCHEMA = (
    'Return ONLY a JSON object (no markdown) with EXACTLY these keys:\n'
    '{"caption": "<short clinical caption>", '
    '"tissue": {"necrotic_pct": <int>, "slough_pct": <int>, "granulation_pct": <int>}, '
    '"infection": "Infected" | "Not infected" | "Undetermined", '
    '"moisture": "Dry" | "Moderate" | "High" | "Undetermined", '
    '"depth": "superficial" | "cavity" | "undetermined", '
    '"urgency_flags": ["<short visual red-flags; [] if none>"], '
    '"note": "<one line: for given-label prompts, agreement/disagreement vs the CV labels; '
    'for the blind prompt, your reasoning>"}'
)

VLM_SYSTEM = {
    "P1": ("You are a wound-photo describer. Report ONLY what you visually observe — tissue "
           "colours and rough proportions, any visible signs of infection, moisture sheen, and "
           "apparent depth. Do NOT reason about dressings or treatment.\n\n" + SCHEMA),
    "P2": ("You are a clinical wound-assessment assistant. Examine the wound photograph and "
           "CROSS-VALIDATE it against the provided CV T.I.M.E. labels: state whether the visual "
           "appearance agrees, and flag any discrepancy (especially infection and tissue). Then note "
           "the dressing properties the wound appears to need. Be clinically precise.\n\n" + SCHEMA),
    "P3": ("You are a wound-assessment assistant. Give a BRIEF visual assessment (1–2 sentence "
           "caption) plus the structured fields. Keep it terse.\n\n" + SCHEMA),
    "P4": ("You are an INDEPENDENT wound assessor. Assess this wound photograph on its own merits. "
           "You are NOT given any prior CV labels — estimate the tissue percentages, infection "
           "status, moisture level and depth PURELY from what you see in the image. Be objective and "
           "do not assume the wound is healthy; if you see signs of infection (erythema, pus, "
           "slough, odour cues), report 'Infected'.\n\n" + SCHEMA),
}

def vlm_human(variant, assessment_text, demo_text):
    """P1–P3 are shown the CV labels; P4 (blind) is NOT."""
    if variant == "P4":
        return (f"Assess this wound photograph independently. Patient context: {demo_text}. "
                f"Estimate tissue %, infection, moisture and depth from the image alone, and return the JSON.")
    return (f"Patient T.I.M.E. assessment from the CV pipeline:\n{assessment_text}\n\n"
            f"Patient context: {demo_text}\nExamine the wound photograph and return the JSON described above.")

print("Four VLM prompt strategies defined (P1 appearance · P2 current · P3 terse · P4 blind).")

# %% [markdown]
# ## Cell 6 — Variant caption generator (calls the VLM directly, per prompt) + cache

# %%
_CAP_CACHE = {}   # (variant, case_id) -> parsed caption dict

def generate_variant_caption(variant, tc, inputs):
    key = (variant, tc["case_id"])
    if CACHE_CAPTIONS and key in _CAP_CACHE:
        return _CAP_CACHE[key]
    demo_bits = ["patient reports diabetes" if inputs["diabetes"] == "yes" else "no diabetes reported"]
    if inputs["depth_self"] not in ("unknown", ""):
        demo_bits.append(f"patient says wound looks '{inputs['depth_self']}'")
    demo = "; ".join(demo_bits)
    out = {"variant": variant, "error": "", "caption": "", "infection": "Undetermined",
           "moisture": "Undetermined", "depth": "undetermined", "tissue": {}, "urgency_flags": [],
           "note": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0}
    try:
        b64, mime = _img_to_b64(tc["image_ref"])
        vlm = mm.make_llm(VLM_MODEL)
        msgs = [SystemMessage(content=VLM_SYSTEM[variant]),
                HumanMessage(content=[{"type": "text", "text": vlm_human(variant, inputs["assessment_text"], demo)},
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
                    "cost_usd": round(mm._compute_cost(mm.VLM_REGISTRY, VLM_MODEL, inp, otk), 8),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "error": "" if p else "non-JSON; raw used"})
    except Exception as e:
        out["error"] = str(e)
    if CACHE_CAPTIONS: _CAP_CACHE[key] = out
    return out
print("Variant caption generator + cache defined.")

# %% [markdown]
# ## Cell 7 — Caption metrics: infection accuracy · tissue bucket · discrepancy detection

# %%
def _norm_inf(s):
    s = (s or "").lower()
    if "not" in s or s in ("no", "none", "absent"): return "Not infected"
    if "infect" in s: return "Infected"
    return "Undetermined"

def caption_metrics(cap, tc):
    """Compare the VLM's VISUAL read to the case's CV label.
    Adversarial cases span 3 discrepancy DIRECTIONS (missed infection / missed necrosis / over-call),
    so 'detected' is scored on whichever axis is actually wrong — infection OR tissue-bucket."""
    label_inf = _norm_inf(tc["time_payload"]["infection"])
    vlm_inf   = _norm_inf(cap.get("infection"))
    # tissue bucket (non-viable ≥25 vs <25)
    lab_nv = tc["time_payload"]["necrotic_pct"] + tc["time_payload"]["slough_pct"]
    t = cap.get("tissue", {}) or {}
    vlm_nv = (t.get("necrotic_pct", 0) or 0) + (t.get("slough_pct", 0) or 0)
    tissue_bucket_ok = ((lab_nv >= 25) == (vlm_nv >= 25)) if t else None
    adv = tc["_adversarial"]

    inf_disagree    = (vlm_inf != label_inf and vlm_inf != "Undetermined")
    tissue_disagree = (tissue_bucket_ok is False)      # VLM's NV bucket differs from the label's
    axis = None
    if adv:
        parts = (["infection"] if inf_disagree else []) + (["tissue"] if tissue_disagree else [])
        axis = "+".join(parts) if parts else "none"
    return {
        "label_infection": label_inf, "vlm_infection": vlm_inf,
        "label_nv": lab_nv, "vlm_nv": vlm_nv,
        # non-adversarial accuracy: VLM read should MATCH the (correct) label
        "infection_correct": (vlm_inf == label_inf) if (not adv and vlm_inf != "Undetermined") else None,
        # adversarial: VLM should DISAGREE on the axis that is wrong (infection OR tissue)
        "discrepancy_detected": (inf_disagree or tissue_disagree) if adv else None,
        "discrepancy_axis": axis,
        "tissue_bucket_ok": tissue_bucket_ok,
        "flagged_in_note": bool(re.search(r"disagree|discrepan|however|but |spreading|infect|mismatch|necro|slough",
                                          cap.get("note", ""), re.I)),
    }
print("Caption-metric functions defined.")

# %% [markdown]
# ## Cell 8 — Downstream generation (caption → recommendation) + clinical check + RAGAS

# %%
DRESSING_ALIASES = {
    "film": ["film","transparent film"], "hydrocolloid": ["hydrocolloid","renocare"], "foam": ["foam","renofoam"],
    "tulle": ["tulle","paraffin"], "hydrogel": ["hydrogel","dermacyn"], "alginate": ["alginate","alginogel","flaminal"],
    "alginogel": ["alginogel","flaminal"], "hydrofiber": ["hydrofibre","hydrofiber","aquacel"], "silver": ["silver"],
    "iodine": ["iodine","povidone","cadexomer"], "charcoal": ["charcoal","activated carbon","zorflex"],
    "polymeric_membrane": ["polymeric membrane","polymem"], "hydroconductive": ["hydroconductive","drawtex"]}
def _forms(t): return DRESSING_ALIASES.get(t, [t.replace("_"," ")])
_POS = re.compile(r"^##\s*(dressing you need|example products|step-?by-?step)", re.I|re.M)
_AV  = re.compile(r"^##\s*(dressings to avoid)", re.I|re.M)
def _postxt(a):
    keep, out = False, []
    for ln in a.split("\n"):
        s = ln.strip()
        if s.startswith("##"): keep = bool(_POS.match(s)) and not _AV.match(s); continue
        if keep: out.append(ln.lower())
    return " ".join(out) if out else a.lower()
def _rec(tok, a): return any(f in _postxt(a) for f in _forms(tok))
def check_clinical(a, tc):
    if not a or a.startswith("ERROR"): return {"overall": "FAIL"}
    r = {}
    for c in tc.get("contraindicated_dressings", []):
        base = c.split("(")[0].strip(); r[f"avoid_{base}"] = {"result": "FAIL" if _rec(base, a) else "PASS"}
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
print("Downstream generation + clinical checker + RAGAS defined.")

# %% [markdown]
# ## Cell 9 — One full pass over all variants × cases

# %%
def run_one_pass(run_idx):
    print(f"\n{'='*70}\n  G4-P — RUN {run_idx}/{N_RUNS}\n{'='*70}")
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
                "run": run_idx, "variant": v, "case_id": tc["case_id"], "category": tc["category"],
                "adversarial": tc["_adversarial"], "narrative_query": inputs["narrative_query"],
                "reference": tc.get("reference", ""), "retrieved_contexts": inputs["retrieved_contexts"],
                "answer": answer, "caption": cap.get("caption", ""), "vlm_infection": cm["vlm_infection"],
                "label_infection": cm["label_infection"], "caption_metrics": cm,
                "safety_overall": safety.get("overall", "N/A"),
                "vlm_cost_usd": cap.get("cost_usd", 0.0),
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
# ## Cell 10 — Execute 3 runs (captions cached across runs)

# %%
all_runs, t0 = [], time.perf_counter()
for r in range(1, N_RUNS + 1):
    all_runs.append(run_one_pass(r))
n_caps = N_RUNS * len(VARIANTS) * len(testset)
print(f"\n✅ {N_RUNS} runs in {(time.perf_counter()-t0)/60:.1f} min · "
      f"{n_caps} captions generated fresh (4 variants × {len(testset)} cases × {N_RUNS} runs)")

# freeze every caption (run × variant × case) — reproducibility + shows the non-determinism
cap_dump = [{"run": r+1, "variant": rec["variant"], "case_id": rec["case_id"],
             "vlm_infection": rec["vlm_infection"], "label_infection": rec["label_infection"],
             "caption": rec["caption"]}
            for r in range(N_RUNS) for v in VARIANTS for rec in all_runs[r][v]["records"]]
json.dump(cap_dump, open(RESULTS_DIR / "G4P_captions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"   Froze {len(cap_dump)} captions → G4P_captions.json")

# %% [markdown]
# ## Cell 11 — Caption metrics per variant (computed once from cached captions)

# %%
cap_stats = {}
for v in VARIANTS:
    inf_ok, disc, tiss = [], [], []
    for r in range(N_RUNS):                       # aggregate over ALL runs (captures VLM noise)
        for rec in all_runs[r][v]["records"]:
            cm = rec["caption_metrics"]
            if cm["infection_correct"]    is not None: inf_ok.append(cm["infection_correct"])
            if cm["discrepancy_detected"] is not None: disc.append(cm["discrepancy_detected"])
            if cm["tissue_bucket_ok"]     is not None: tiss.append(cm["tissue_bucket_ok"])
    cap_stats[v] = {
        "infection_accuracy": round(sum(inf_ok)/len(inf_ok), 3) if inf_ok else None,
        "discrepancy_detection": round(sum(disc)/len(disc), 3) if disc else None,
        "tissue_bucket_acc": round(sum(tiss)/len(tiss), 3) if tiss else None,
        "n_nonadv": len(inf_ok), "n_adv": len(disc)}

print(f"\n{'='*74}\n  G4-P CAPTION QUALITY (from cached captions)\n{'='*74}")
print(f"  {'Var':<5}{'Infection acc':<16}{'Tissue-bucket':<16}{'Discrepancy det.':<18}")
for v in VARIANTS:
    s = cap_stats[v]
    ia = f"{s['infection_accuracy']*100:.0f}% (n={s['n_nonadv']})" if s['infection_accuracy'] is not None else "—"
    tb = f"{s['tissue_bucket_acc']*100:.0f}%" if s['tissue_bucket_acc'] is not None else "—"
    dd = f"{s['discrepancy_detection']*100:.0f}% (n={s['n_adv']})" if s['discrepancy_detection'] is not None else "—"
    print(f"  {v:<5}{ia:<16}{tb:<16}{dd:<18}  {VARIANT_LABEL[v]}")

# %% [markdown]
# ## Cell 12 — FA / AR per variant (mean ± SD over runs) + safety

# %%
def _ms(xs): xs=[x for x in xs if x is not None]; return (round(statistics.mean(xs),4), round(statistics.stdev(xs),4) if len(xs)>1 else 0.0) if xs else (0.0,0.0)
agg = {}
for v in VARIANTS:
    fa = [all_runs[r][v]["ragas"]["faithfulness"] for r in range(N_RUNS)]
    ar = [all_runs[r][v]["ragas"]["answer_relevancy"] for r in range(N_RUNS)]
    sp = [sum(1 for x in all_runs[r][v]["records"] if x["safety_overall"]=="PASS")/len(all_runs[r][v]["records"]) for r in range(N_RUNS)]
    agg[v] = {"FA": _ms(fa), "AR": _ms(ar), "Safety": _ms(sp), "fa_runs": fa, "ar_runs": ar}
print(f"\n{'='*66}\n  G4-P DOWNSTREAM (mean ± SD, {N_RUNS} runs, {len(testset)} cases)\n{'='*66}")
print(f"  {'Var':<5}{'FA':<18}{'AR':<18}{'Safety':<14}")
for v in VARIANTS:
    a = agg[v]
    print(f"  {v:<5}{a['FA'][0]:.4f}±{a['FA'][1]:.4f}   {a['AR'][0]:.4f}±{a['AR'][1]:.4f}   {a['Safety'][0]*100:.1f}%")

# %% [markdown]
# ## Cell 13 — HEADLINE: does the blind prompt (P4) fix the anchoring? (P2 vs P4)

# %%
print(f"\n{'='*60}\n  G4-P HEADLINE — P2 (current) vs P4 (blind)\n{'='*60}")
for m, key, fmt in [("Infection accuracy (non-adv)", "infection_accuracy", "pct"),
                    ("Discrepancy detection (adv)",  "discrepancy_detection", "pct")]:
    p2, p4 = cap_stats["P2"][key], cap_stats["P4"][key]
    def f(x): return "—" if x is None else f"{x*100:.0f}%"
    print(f"  {m:<32} P2={f(p2):<8} P4={f(p4):<8}")
print(f"  {'FA (downstream)':<32} P2={agg['P2']['FA'][0]:.4f}  P4={agg['P4']['FA'][0]:.4f}")
print(f"  {'AR (downstream)':<32} P2={agg['P2']['AR'][0]:.4f}  P4={agg['P4']['AR'][0]:.4f}")
print("\n  Read: if P4 raises discrepancy detection WITHOUT hurting non-adversarial infection")
print("  accuracy, the blind prompt is the fix — the VLM stops anchoring to the CV labels.")

# %% [markdown]
# ## Cell 14 — The anchoring diagnostic: infection read per variant on the adversarial case
#
# The clean evidence. On `cat_g` the CV label is "Not infected" but the image is infected. A variant
# that reads "Infected" here **caught the discrepancy**; one that reads "Not infected" **anchored**.

# %%
adv_cases = [tc for tc in testset if tc["_adversarial"]]
# lookup: (variant, case_id) -> list of caption_metrics dicts across runs
cm_lookup = defaultdict(list)
for r in range(N_RUNS):
    for v in VARIANTS:
        for rec in all_runs[r][v]["records"]:
            cm_lookup[(v, rec["case_id"])].append(rec["caption_metrics"])
print("  Adversarial cases — discrepancy caught across the 3 runs (scored on the axis that is wrong):")
for tc in adv_cases:
    exp = (tc.get("escalation_flags_expected") or ["?"])[0]
    print(f"\n  {tc['case_id']}  | CV label: infection='{tc['time_payload']['infection']}', "
          f"NV={tc['time_payload']['necrotic_pct']+tc['time_payload']['slough_pct']}%  | expect: {exp[:55]}")
    for v in VARIANTS:
        cms = cm_lookup[(v, tc["case_id"])]
        caught = sum(1 for c in cms if c["discrepancy_detected"])
        axes   = [c["discrepancy_axis"] for c in cms]
        print(f"    {v}: caught {caught}/{len(cms)} runs  (axes: {axes})")

# %% [markdown]
# ## Cell 15 — Save results + summary

# %%
for v in VARIANTS:
    recs = [rec for r in range(N_RUNS) for rec in all_runs[r][v]["records"]]
    json.dump(recs, open(RESULTS_DIR / f"{EXP_ID}_{v}_results_all.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

summary = {
    "experiment": "G4-P — VLM caption prompt strategy",
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_runs": N_RUNS, "n_cases": len(testset), "n_adversarial": n_adv,
    "vlm_model": VLM_MODEL, "generation_llm": GEN_LLM,
    "variants": {v: {"label": VARIANT_LABEL[v],
                     "infection_accuracy": cap_stats[v]["infection_accuracy"],
                     "tissue_bucket_acc": cap_stats[v]["tissue_bucket_acc"],
                     "discrepancy_detection": cap_stats[v]["discrepancy_detection"],
                     "FA_mean": agg[v]["FA"][0], "FA_sd": agg[v]["FA"][1],
                     "AR_mean": agg[v]["AR"][0], "AR_sd": agg[v]["AR"][1],
                     "Safety_mean": agg[v]["Safety"][0]} for v in VARIANTS},
    "headline_P2_vs_P4": {
        "P2_discrepancy": cap_stats["P2"]["discrepancy_detection"],
        "P4_discrepancy": cap_stats["P4"]["discrepancy_detection"],
        "P2_infection_acc": cap_stats["P2"]["infection_accuracy"],
        "P4_infection_acc": cap_stats["P4"]["infection_accuracy"]},
}
json.dump(summary, open(RESULTS_DIR / "G4P_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("✅ Saved: G4P_summary.json · G4P_{P1..P4}_results_all.json · G4P_captions.json")

# %% [markdown]
# ## Cell 16 — How to read G4-P
#
# 1. **Discrepancy detection (Cell 11 / 13 / 14)** is the point. If **P4 (blind) > P2 (current)**,
#    you have direct evidence that the VLM was *anchoring* to the CV labels and that a blind prompt
#    restores genuine cross-validation — the FYP2 clinical contribution.
# 2. **Guard against a trade-off:** check P4's **non-adversarial infection accuracy** (Cell 11). If
#    blind assessment also *invents* infection on clean wounds (false positives), that's a cost —
#    report both numbers. The ideal variant is high on both.
# 3. **FA/AR (Cell 12)** should be similar across variants (retrieval + prompt fixed) — the caption's
#    downstream effect is second-order to G4-A. The primary G4-P signal is caption *quality*, not FA.
# 4. **Caveat:** with 1 adversarial case, discrepancy detection is 0% or 100% — a single point.
#    Expand the VLM-DISC adversarial set (8–10 cases) before quoting a rate in the thesis. The
#    per-variant **non-adversarial accuracy** (n≈14) is already meaningful.
# 5. **Next:** feed the winning prompt back into `wound_app_multimodal.VLM_SYSTEM_PROMPT`, then
#    re-run **G4-A** — you should see the caption start *catching* CV errors (a positive story),
#    and G4-B can compare GPT-4o-V vs Gemini-V under the winning prompt.
