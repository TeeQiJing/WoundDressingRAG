# VerdaSense — Viva Q&A Cheat-Sheet

**How to use:** every likely hard question, with a crisp answer and **the one number to cite**. The golden rule — *never claim "best"; claim "rigorously ablated + evidence-based," and back every architectural word with a result.*

**Your 15-second elevator pitch:**
> *"VerdaSense is a hybrid multimodal RAG system for wound-dressing recommendation. A deterministic rules layer (the MOH wound-type algorithm) fixes the safety-critical decisions; a RAG layer grounds the rationale in clinical guidelines; and a blind vision model cross-validates the labels against the wound photo. Every design decision — retrieval, prompt, VLM, model — is chosen by an ablation, not assumed."*

---

## A. Architecture & "is it really hybrid / multimodal / RAG?"

**Q: How is this genuinely *hybrid* and not just an LLM with a prompt?**
Two layers, each with a *measured* job. Layer 1 (rules): `classify_wound` → wound type 1–8 → a *pinned* retrieval axis fixes the dressing **category** and the referral/antibiotic decision deterministically (it matches the MOH algorithm). Layer 2 (RAG): supplies the *why/how/when*. → *Cite: grounding lifts faithfulness **FA 0.69 → 0.81** (G1); the rules layer makes referral/antibiotic deterministic (safety ~90%).*

**Q: How is it genuinely *multimodal* — the image doesn't even pick the dressing?**
By design. The image's job is **cross-validation + personalisation, not classification** — a choice *forced by evidence*: putting the caption into retrieval *hurt* it (R5: −18.75 pp HR@6), because guideline text and visual language are different semantic registers. So the blind caption feeds generation, where it catches label errors and personalises. → *Cite: the blind caption catches **100%** of deliberately-wrong labels (G4-P / VLM-DISC), at **no** faithfulness or safety cost (G4-A). The image genuinely changes the output — it escalates the advice when it spots danger the labels missed.*

**Q: Isn't your knowledge base too small for "real" RAG (companies use thousands of docs)?**
RAG's purpose isn't scale — it's **grounding, provenance, and updatability**, which matter *more* in a safety-critical clinical setting. A small, curated, authoritative KB is a *deliberate design choice* for control over what the model may say. And my retrieval is **structured** (multi-axis + a rule-pinned axis + metadata filters), beyond textbook top-k. → *Cite: grounding cuts hallucination (FA 0.69→0.81); every claim carries a `[S#]` citation (auditable); the retrieval ablations show quality measurably drives output (R1 +4.4 pp CR; R5 caption-in-retrieval −18.75 pp HR@6). The architecture scales to a larger KB unchanged.*

**Q: Why not just fine-tune one model to do everything?**
Fine-tuning loses **traceability** (can't cite a source), **updatability** (guidelines change → retrain), and **auditability** — all essential for a clinical tool. And fine-tuning the VLM as a *classifier* wasn't the goal — the task is captioning/cross-validation, and R5 showed visual features don't help retrieval. RAG + rules keeps every decision inspectable.

---

## B. The multimodal contribution (the part examiners will probe)

**Q: Why didn't Faithfulness (FA) improve when you added the caption? Doesn't that mean it's useless?**
No — **FA is structurally the wrong lens.** FA measures whether the answer sticks to the *guideline text (KB)*. The caption's job is to add what the guidelines *don't* contain — visual facts ("this looks infected"). So a correct visual observation gets *penalised* by FA for going off-KB. → *Cite: on the adversarial cases the caption's FA drops (Cat G −5.8 pp) precisely because it adds correct out-of-KB flags; its real value is VLM-DISC (100%) and end-to-end escalation, neither visible to FA. Analogy: grading a nurse only on "did you quote the manual" — the one who also looks at the wound scores lower on quoting but is the better nurse.*

**Q: VLM-DISC = 100% sounds too good. Is it real?**
Yes, and I can *prove* it's genuine — because I found that **VLM-DISC alone is gameable**. A model that reads "abnormal" on everything scores 100% by flagging everything (Gemma-3: 100% DISC but 95% over-call, 49% accuracy — useless). GPT-4o-mini's 100% is *backed by* high accuracy (73% infection / 86% tissue) and only moderate over-call → genuine sensitivity. → *Cite: G4-C, `fig3_disc_vs_accuracy.png`. This self-critical check is itself a contribution.*

**Q: Does the image ever make things *worse*?**
Yes — one case, and I characterise it honestly: when danger is in the patient's *notes* and the photo genuinely looks clean, the caption's "looks clean" read can pull advice toward clean management (G4-A directionality finding). → *Design rule: the caption is **advisory** and never overrides notes/label-driven escalation. I even hardened the production prompt to enforce this.*

**Q: Why "blind" prompting? Isn't giving the model the labels more informative?**
The opposite, for cross-validation. When shown the labels, the VLM **anchors** (echoes them) — catching only ~2% of infection-axis errors. Withholding the labels is the *only* thing that restores genuine perception. → *Cite: G4-P — blind 100% vs label-shown 14–19% VLM-DISC; `fig1_g4p_vlm_disc.png`. Non-obvious, empirically demonstrated.*

---

## C. Model choices

**Q: Isn't Gemini-2.5-Flash newer/better than GPT-4o-mini? Why not use it?**
Tested it head-to-head — **Gemini refused 41% of clinical wound images** (empty, content-blocked responses on the infected/necrotic ones), and the block is **non-configurable** (`safety_settings=BLOCK_NONE` recovered 0/5; `BlockedReason.OTHER`). A safety-net that won't look at the dangerous wounds is worse than useless. → *Cite: G4-B, `fig2_refusal_rate.png`. Reliability on the graphic input distribution is a first-class selection criterion — the "obvious newer model" is actively wrong here.*

**Q: Could you self-host / avoid sending patient photos to a US API?**
Yes — tested 4 open-source VLMs; **Qwen2.5-VL-72B matches GPT-4o-mini on accuracy** (infection 76% vs 73%, tissue 85%≈86%), is 6× cheaper, self-hostable, and had ~0% refusals → **data sovereignty** (images never leave your infrastructure). → *Cite: G4-C, `fig4_caption_accuracy.png`. GPT-4o-mini stays the best single choice; Qwen2.5-VL is the recommended privacy-preserving alternative.*

**Q: Bigger model = better, right?**
No — Qwen3-VL-235B (the largest) *lost* to the smaller Qwen2.5-VL-72B on every caption metric. Calibration matters more than parameter count for this niche visual task. → *Cite: G4-C.*

---

## D. Evaluation rigor

**Q: How do you know your pipeline choices are right and not arbitrary?**
Every one is an ablation winner, 3 runs each, fixed RAGAS judge: R1-C multi-axis (+4.4 pp CR) · R2-A dense (hybrid *didn't* help) · R3-C k=6 (CR peaks then drops at k=8) · R4-B BGE-large · R5 caption-out-of-retrieval · G1-C grounded (FA +12 pp) · G4-P4 blind · G4-B1 GPT-4o-mini-V. → *The ablation chain **is** the defense. See `MDs/VerdaSense_Complete_Ablation_Summary.md`.*

**Q: Isn't RAGAS just an LLM judging an LLM?**
Yes, so I control for it: the judge is *fixed* (`gpt-4o-mini` + `text-embedding-3-small`) and *never changed* across all experiments, so it's a constant — only *relative* comparisons are claimed, and each is a 3-run mean ± SD. Safety is a *deterministic* rule-checker, not LLM-judged. And the decisive validation is human (H1), not RAGAS.

**Q: Is it clinically validated?**
Not yet — that's **H1**, a blinded one-pass review by a wound-care clinician (Ms Saw) over all 34 curated cases, yielding a **Clinical Concordance Rate**. I'm honest about this: today it's *well-engineered*; H1 is what makes it *clinician-validated*. *(The review package is built and ready.)*

---

## E. Limitations (say these *before* the examiner does — it reads as command of the work)

- **No whole-system head-to-head vs an external baseline** → I claim "each component is the ablation-winner," **not** "best system."
- **The multimodal value is invisible to FA/AR** → reported via VLM-DISC + safety, by design.
- **Clinical correctness currently rests on AI-authored gold** until H1.
- **Etiology + wound-depth deferred** (supervisor-approved scoping); moisture/depth are low-confidence from a photo (a finding, not an oversight — WT3/4 infection is peri-wound, not bed-visible).
- **Open-source generation LLMs (G3) under-delivered** on faithfulness → closed-source relied on for generation (though open *vision* models are competitive — G4-C).

---

## F. If you get stuck — three sentences that always work

1. *"I don't claim it's the best system; I claim every design decision is justified by an ablation, and I can show you the number."*
2. *"The multimodal layer's value is a measured clinical safety-net — 100% error-detection at no faithfulness or safety cost — which is exactly the failure a rules-only pipeline can't self-detect."*
3. *"I characterise its limits honestly — where FA is the wrong metric, where a metric is gameable, where the caption is a liability — and the decisive clinical validation is the clinician review, H1."*

---

*Companion docs: `MDs/VerdaSense_Complete_Ablation_Summary.md` (all numbers), the four `G4*_Analysis.md` write-ups, and `report_figures/` (fig1–fig4).*
