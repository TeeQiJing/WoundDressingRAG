# VerdaSense — Complete Ablation Summary (FYP1 + FYP2)

**A single reference for every ablation, its result, and what it proved.**
System: a hybrid **multimodal RAG** clinical decision-support tool for wound-dressing recommendation, grounded in the T.I.M.E. framework (Tissue, Infection, Moisture, Edge) and the MOH wound-type algorithm (WT1–8).

**Evaluation protocol (constant across all experiments):** RAGAS judge = `gpt-4o-mini` + `text-embedding-3-small` (never changed) · 3 independent runs, mean ± SD · retrieval metrics are deterministic (SD from the LLM judge only). **Design philosophy: every architectural choice is decided by an ablation, not assumed.**

**Metrics glossary:**
- *Retrieval:* **CR** Context Recall · **CP** Context Precision · **HR@6** Hit-Rate · **MRR** · **NDCG@6**
- *Generation:* **FA** Faithfulness (primary — hallucination resistance) · **AR** Answer Relevancy · **Safety** (deterministic rule-checker: dressing-class + antibiotic + referral)
- *Multimodal:* **VLM-DISC** discrepancy-detection rate (does the VLM catch a wrong label from the image?) · **VLM-ACC** caption accuracy vs ground truth · **Refusal rate**

---

## Part 0 — The two stages at a glance

| Stage | Experiments | Question answered |
|---|---|---|
| **FYP1 — Retrieval** | R1–R5 | How to retrieve the right guideline evidence? |
| **FYP1 — Generation** | G1–G3 | How to generate a faithful, safe recommendation? |
| **FYP2 — Multimodal** | G4-P, G4-A, G4-B, G4-C | Does adding a wound photo (VLM) help, how, and with which model? |
| **FYP2 — Human eval** | VLM-DISC, VLM-ACC, **H1** | Does it hold up to a clinician? |

---

## Part 1 — FYP1 Retrieval Ablations (R1–R5)

| # | Question | Arms | Winner | Key result | What it proved |
|---|---|---|---|---|---|
| **R1** | Query strategy | A: keyword · B: NL query · C: multi-axis sub-queries | **R1-C** multi-axis | CR **0.868** vs R1-A 0.824 (**+4.4 pp**); leads every quality metric | Decomposing the query into 3 axes (pinned wound-type + dressing-mechanism + patient-notes) retrieves more clinically-relevant content than a single query. |
| **R2** | Retrieval method | A: dense · B: BM25 · C: hybrid (RRF) · rerank | **R2-A** dense-only | BM25-alone had *higher* CR (0.906) but **hybrid fusion *under*performed dense** (0.870 < 0.880); dense chosen on the balanced metric picture + determinism | Counter-intuitive: lexical/hybrid retrieval did **not** help — dense-only is the robust, reproducible choice. (A finding worth reporting *because* it refutes the hypothesis.) |
| **R3** | Top-K | k = 2 · 4 · 6 · 8 | **R3-C** k=6 | CR **peaks at k=6 (0.870)** and *drops* at k=8 (0.861); HR@6 0.906; ~930 context words (cost-efficient) | More chunks ≠ better: beyond k=6 the extra low-ranked chunks dilute context quality. k=6 is the accuracy **and** cost optimum. |
| **R4** | Embedding model | A: MedEmbed · **B: BGE-large** · E5 | **R4-B** BGE-large-en-v1.5 | CR **0.895**, best of the three; deterministic IR metrics reproduced exactly | A strong general embedding (BGE-large) beats a domain "medical" embedding for this guideline-retrieval task. |
| **R5** | **Caption in retrieval?** | A: none · B: caption→retrieval · C/D variants | **R5-A** (no caption in retrieval) | Injecting the VLM caption *into retrieval* **hurt**: **−6.6 pp CR, −18.75 pp HR@6** | **The pivotal FYP1→FYP2 result.** Visual-appearance language and guideline text are different semantic registers BGE can't bridge — so the caption must feed **generation only**, never retrieval. *This single result defines the entire FYP2 multimodal architecture (Paradigm B).* |

**Retrieval chain locked:** R1-C multi-axis → R2-A dense → R3-C k=6 → R4-B BGE-large → R5 caption excluded from retrieval.

---

## Part 2 — FYP1 Generation Ablations (G1–G3)

| # | Question | Winner | Key result | What it proved |
|---|---|---|---|---|
| **G1** | Prompt strategy | **G1-C** grounded system prompt | **FA 0.69 (zero-shot) → 0.81 (grounded), +12 pp**; passes the safety gate | Explicit grounding instruction is what turns raw RAG output into a *faithful, contraindication-aware* recommendation. This is the measured value of the "RAG" layer. |
| **G2** | Closed-source LLM | **G2-D** Gemini-2.5-Flash (best FA) | FA **0.815**, Safety **90.6%** — best closed-source; system ships **user-selectable** LLM (gpt-4o-mini default for cost) | With an optimal prompt, LLM choice still moves faithfulness/safety; the pipeline is model-agnostic by design. |
| **G3** | Open-source LLM (7 models, OpenRouter, reasoning-off) | *no open model matched closed-source* | **No open-source LLM reached FA ≥ 0.75 + the safety gate** simultaneously | For the *generation* stage, open-source LLMs under-deliver on faithfulness/safety vs closed-source — closed-source generation is justified. *(Contrast: for the **vision** stage, G4-C later shows open models are competitive — a nice asymmetry.)* |

**Generation chain locked:** G1-C grounded prompt · user-selectable LLM (gpt-4o-mini default) · deterministic safety checker.

---

## Part 3 — FYP2 Multimodal Ablations (the core novelty)

The FYP2 question: **does adding a wound photo help, and can the VLM catch errors the labels/rules miss?** Caption feeds **generation only** (R5), and is generated **blind** (see G4-P).

### G4-P — How to prompt the VLM (blind vs label-shown)
| Prompt | VLM-DISC | Infection acc (non-adv) | FA |
|---|---|---|---|
| P1 appearance / P3 terse | 14% | 100%* | 0.61 / 0.63 |
| P2 cross-validation (old prod) | 19% | 100%* | 0.62 |
| **P4 blind (winner)** | **100%** | 79% | 0.63 |

**Result:** only the **blind** prompt cross-validates — P4 caught **100% (21/21)** of planted label↔image discrepancies vs **14–19%** when the labels are shown. Label-shown prompts **anchor** (echo the label) — infection-axis anchoring ~**98%** (caught 1/45). Blind's nominal 79% accuracy is honest (the label-shown "100%" is label-echoing, not perception). **Blind costs nothing downstream** (FA 0.63 ≈ best; Safety identical). **Proved:** the VLM only becomes a genuine error-checker when the CV labels are *withheld* — a non-obvious, decisive design choice. (Now live in the app.)

### G4-A — Does the caption help generation? (caption vs no-caption, 34 cases)
| | A0 no caption | A1 caption | Δ |
|---|---|---|---|
| Faithfulness | 0.613 | 0.605 | **−0.8 pp** (within noise) |
| Safety | 91.2% | 90.2% | −1 pp (one case) |
| Answer Relevancy | 0.455 | 0.418 | −3.7 pp |

**Result:** adding the blind caption is **FA-/safety-neutral**. The expansion to 34 cases was decisive — the pilot's per-category "wins" (Cat B/F +12 pp) were **small-n artifacts that collapsed at n=6**; no category shows an FA *improvement* → **FA is the wrong lens** (it penalises the caption's out-of-KB visual claims, e.g. Cat G −5.8 pp). **Directionality finding (novel):** the caption is an **asset when the image reveals danger the labels miss** (Cat G → it escalates the advice), but a **liability when danger is in the patient's notes and the image looks clean** (it can pull advice toward "clean"). **Proved:** the caption must stay **advisory** — never override notes/label-driven escalation. Its value is error-catching (G4-P), not FA.

### G4-B — Which closed-source VLM? (GPT-4o-mini-V vs Gemini-2.5-Flash-V)
| | GPT-4o-mini-V | Gemini-2.5-Flash-V |
|---|---|---|
| **Refusals** | **0%** | **41%** |
| VLM-DISC | **100%** | 47.6%\* |
| Tissue acc | **86%** | 67%\* |
| Cost / latency | **cheaper / faster** | +36% / +64% |

**Result:** GPT-4o-mini wins on every axis. **Gemini refused 41% of clinical wound images** (empty `BlockedReason.OTHER` responses, concentrated on infected/necrotic/adversarial wounds). A `safety_settings=BLOCK_NONE` test recovered **0/5** — the block is **non-configurable** on the Developer API. **Proved:** the pricier proprietary model is *disqualified* — not on accuracy, but on a non-negotiable content filter. **Deployment lesson: reliability on the graphic input distribution is a first-class model-selection criterion.**

### G4-C — Which open-source VLM? (Qwen2.5-VL, Qwen3-VL, Gemma-3, Gemma-4, OpenRouter, reasoning-off)
| Model | Refusals | Infection acc | Tissue acc | VLM-DISC | Over-call |
|---|---|---|---|---|---|
| **Qwen2.5-VL-72B (best open)** | ~0% | **0.76** | 0.85 | 0.71 | 21% |
| Qwen3-VL-235B | 0% | 0.68 | 0.79 | 0.57 | 38% |
| Gemma-4-26B | 0% | 0.60 | 0.78 | 0.86 | 62% |
| Gemma-3-27B | ~0% | 0.49 | 0.35 | **1.00** | **95%** |

**Three results:** (1) **open models eliminate the refusal problem** — all ~0% vs Gemini's 41% → a self-hostable open VLM structurally avoids the content filter. (2) **Methodological finding — VLM-DISC is gameable:** Gemma-3 scores a perfect 100% VLM-DISC but earns it by over-calling "Infected" on **95% of clean wounds** (49% accuracy) → **DISC must always be read *with* accuracy + over-call rate.** (3) **Best open = Qwen2.5-VL-72B** — infection accuracy (76%) *exceeds* GPT-4o-mini (73%), tissue ≈ equal, **6× cheaper**, **self-hostable = patient images never leave your infrastructure (data sovereignty)** — at the cost of lower discrepancy-sensitivity. **Bigger ≠ better** (Qwen3-VL-235B lost to the smaller Qwen2.5-VL-72B). **Proved:** GPT-4o-mini is the best single choice, and Qwen2.5-VL-72B is a genuinely viable, privacy-preserving open alternative.

\* Gemini/label-shown accuracy figures are inflated/partial (label-echoing or computed only on non-refused images) — see the individual write-ups.

---

## Part 4 — Human & caption-level metrics

| Item | Result | Status |
|---|---|---|
| **VLM-DISC** (does the VLM catch wrong labels?) | GPT-4o-mini **100%** (genuine — backed by high accuracy) | ✅ |
| **VLM-ACC** (caption vs ground truth) | GPT-4o-mini **infection 73% · tissue 86%** (moisture/depth excluded as low-confidence — a still photo can't convey exudate/depth) | ✅ (via G4-B) |
| **H1 — blinded clinician review (Ms Saw)** | 34-case one-pass review package prepped (`ragas_testset/h1_review.html`); yields the **Clinical Concordance Rate** | ⏳ *the decisive remaining deliverable* |

---

## Part 5 — The ablation-best configuration (the final deployed system)

| Component | Winner | Setting | Decided by |
|---|---|---|---|
| Query strategy | R1-C | Multi-axis sub-queries (A pinned + B mechanism + C notes) | R1 |
| Retrieval | R2-A | Dense only (`similarity_search`) | R2 |
| Top-K | R3-C | k = 6 | R3 |
| Embedding | R4-B | `BAAI/bge-large-en-v1.5` | R4 |
| Caption placement | R5 | Generation only (never retrieval) | R5 |
| Prompt | G1-C | Grounded system prompt | G1 |
| Generation LLM | G2-D | User-selectable (gpt-4o-mini default) | G2/G3 |
| **VLM prompt** | **G4-P4** | **Blind (labels withheld)** | G4-P |
| **VLM model** | **G4-B1** | **`gpt-4o-mini`-Vision** (Qwen2.5-VL-72B = open alt) | G4-B/G4-C |
| Safety layer | — | Deterministic `classify_wound` (MOH algorithm) — rules decide referral/antibiotic | G1/G4-A |

---

## Part 6 — What the whole chain proves (the thesis narrative)

1. **It is a genuine RAG system** — retrieval is systematically tuned (R1–R4) and grounding measurably improves faithfulness (**FA 0.69 → 0.81**, G1).
2. **It is genuinely hybrid** — a deterministic rules layer (`classify_wound` → WT1–8 → pinned retrieval) fixes the dressing *category* + safety decisions, while the RAG layer supplies the *why/how/when*. Each layer has a number behind it.
3. **It is genuinely multimodal, and the image does real work** — the blind VLM caption catches **100%** of wrong labels (G4-P/VLM-DISC) at **no faithfulness or safety cost** (G4-A): a measured clinical safety-net a rules-only pipeline cannot provide. The caption's role is **cross-validation + personalisation, not classification** — a choice *forced by evidence* (R5), and kept **advisory** (G4-A directionality).
4. **Every design decision is evidence-based, not assumed** — blind prompting (G4-P), caption-feeds-generation-not-retrieval (R5), k=6 (R3), BGE (R4), multi-axis (R1), grounded prompt (G1).
5. **The methodology is self-critical** — the ablations honestly surface where the system's limits are (FA-is-the-wrong-lens, DISC-is-gameable, WT3/4-photographability, the caption-directionality liability), which is what distinguishes rigorous work from a demo.

---

## Part 7 — Honest limitations (state these before an examiner does)

- **No whole-system head-to-head vs an external baseline** — every component is the *ablation-winner*, but "best system" is **not** claimed. The defensible claim is "rigorously ablated + evidence-based," not "state-of-the-art."
- **The multimodal contribution is invisible to FA/AR** (flat) — its value is VLM-DISC + safety, reported as such; FA is structurally the wrong lens.
- **Clinical correctness currently rests on AI-authored gold** — until **H1** (Ms Saw) validates it, the system is *well-engineered*, not yet *clinician-validated*. H1 is the single highest-leverage remaining task.
- **Etiology + wound-depth deferred** (supervisor-approved scoping), and moisture/depth are low-confidence from a photo.
- **Open-source generation LLMs (G3) underperformed** — closed-source is relied on for generation faithfulness (though open *vision* models are competitive, G4-C).

---

*Per-experiment write-ups: `MDs/Retrieval Ablation/` (R1–R5), `MDs/Generation Ablation/` (G1–G3, G4-A/B/C/P). Ablation plan: `MDs/FYP2 Migration/VerdaSense_FYP2_Ablation_Map_v5.md`. Best config also in `CLAUDE.md`.*
