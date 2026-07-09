# VerdaSense FYP2 — Paper Analysis + Supervisor Meeting Brief

**Prepared for:** Supervisor meeting (FYP1 progress + FYP2 proposal)
**Author:** Tee Qi Jing (23004894)
**Paper analysed:** Abdolahnejad, Mashayekhi, … Joshi, Hong (Skinopathy Inc.), *"A Mobile AI-enhanced Platform for Standardized Wound Assessment and Clinical Decision Support"*, medRxiv preprint, posted **23 Jan 2026** (`papers/`).
**Companion doc:** `VerdaSense_FYP2_Master_Plan.md` (the authoritative FYP2 plan — this brief sits on top of it).

---

## 0. TL;DR — the one thing to say in the meeting

> *"I found a January-2026 preprint (Skinopathy, Canada) that is the closest published system to what I'm building — they do tissue segmentation, etiology classification, size estimation and a rule-based product recommendation engine, deployed on mobile. It validates almost every component of my FYP2 plan. But their recommendation engine is **rule-based only**, and they list **RAG + LLM as future work they have not done yet**. My FYP1 already does the RAG layer they're proposing for the future, and my FYP2 adds a VLM that reads the wound image at the generation stage — which their system also doesn't do for recommendation. So I'm not behind this paper; on the decision-support layer I'm one step ahead of it. The paper mainly tells me where to be **realistic**: don't fine-tune new CV models I can't get data for (use a VLM instead), and borrow their proven patterns for size-calibration, product mapping, and mobile deployment."*

That single framing does three jobs: shows you did the literature scan, de-risks your scope, and turns a "someone already did this" scare into a differentiation argument.

---

## 1. What the Paper Actually Did (deep analysis)

### 1.1 System overview

An end-to-end mobile wound platform = **3 CNNs + calibration + rule engine + mobile/cloud stack**. Pipeline:

```
Smartphone photo (+ fiducial marker)  +  adaptive clinician questionnaire
        │
        ├─ CNN-1  EfficientNet-B7  → wound etiology (arterial / venous / pressure / other)
        ├─ CNN-2  gated stager     → pressure-injury Stage I–IV  (only fires if etiology=pressure, τ=0.85)
        ├─ CNN-3  DeepLabv3+ResNet → tissue segmentation (epithelial / granulation / slough / eschar / bg)
        ├─ Fiducial marker (Hough circle + HSV) → px→mm calibration → size (L/W/area mm²)
        │
        └─ Hybrid Assessment (CV outputs + clinician inputs as 6th parameter, JSON)
               │
               └─ Rule-based decision tree (3 tiers):
                     1) cleansing/prep   2) primary dressing   3) secondary/adjunctive
                     → product-specific (e.g. MAXORB II/Ag+, PluroGel)
```

Deployed: **Flutter** frontend (MVC), **Go/Gin + PostgreSQL** backend, **Docker + Kubernetes**, Azure AD auth, part of a larger EMR "OS".

### 1.2 Headline results

| Component                             | Result                                                                             | Notes                                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Dataset                               | **1,648** de-identified clinical photos                                      | arterial/venous/DFU/pressure/mixed; 32% Monk Skin Tone 7–10; 348 images with tissue masks |
| Etiology classifier (EfficientNet-B7) | **91.75%** mean acc (4 classes); precision 0.85–0.94, recall 0.77–0.97     | venous best (distinct morphology + location)                                               |
| Pressure staging (gated)              | **67% (Stage III) – 92% (Stage I)**; ~81% overall                           | deep stages worst — eschar obscures depth                                                 |
| Tissue segmentation (DeepLabv3)       | mean**Dice 0.64 ± 0.06**, pixel acc **98%**                           | eschar best (DSC 0.64), epithelialisation worst (0.42)                                     |
| Size estimation (fiducial)            | Pearson**r = 0.73** (n=53), MAE **3.7 ± 2.1 mm**, 84.2% within ±5 mm | marker detected in**93%** of images                                                  |
| Recommendation engine                 | concordance vs 2 wound specialists, expert panel on**n=23** cases            | rule-based, 3-tier, product-specific                                                       |

### 1.3 The paper's own stated limitations (this is gold for you)

1. **Dataset still narrow** — MST 4–6 only 4%; no pediatric; hard anatomical sites (toe/heel) weak.
2. **Fiducial marker is a failure point** — 6% miss rate; contamination risk near infected wounds; they suggest marker-free / coin / LiDAR alternatives as *future work*.
3. **Segmentation weak on heterogeneous / deep / biofilm wounds** — exactly the clinically important edge cases.
4. **Rule engine cannot learn or explain** — "future iterations could… [use] **large language models**" and **"Retrieval-augmented generation (RAG) architectures, wherein LLMs query external knowledge bases of wound care protocols and product specifications, could further personalize recommendations while maintaining interpretability and enabling clinicians to trace the evidentiary basis for each suggestion."** ← *They are describing VerdaSense as their future work.*
5. **No depth classification** — depth only ever appears as a *future* burns/multispectral idea; their RGB pipeline does **not** estimate depth (pressure staging is a weak proxy).
6. **Retrospective validation only** — no prospective clinician-in-loop study yet.

> **Quote to keep in your pocket for the viva** (paper, Future Directions): *"Retrieval-augmented generation (RAG) architectures … could further personalize recommendations while maintaining interpretability and enabling clinicians to trace the evidentiary basis for each suggestion."* You can literally say: *"This sentence is my FYP1."*

---

## 2. Head-to-Head: Skinopathy Paper vs VerdaSense

| Dimension                                   | Skinopathy (the paper)                                      | VerdaSense (you)                                                                                    | Who's ahead / verdict                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Tissue decomposition**              | DeepLabv3+ResNet, Dice 0.64                                 | Senior's T-SegNet (LoRA-MobileSAM) + K-Means %                                                      | Parity — both produce tissue %; theirs is end-to-end, yours modular                                              |
| **I/M/E status**                      | Not modelled (infection only via questionnaire)             | IME-Net (Infection/Moisture/Edge)                                                                   | **You're richer** on T.I.M.E. axes                                                                          |
| **Etiology**                          | Dedicated CNN (1,648 imgs, 91.75%)                          | Planned:**VLM-based**, not a new CNN                                                          | They're more accurate*but* needed a huge dataset; your VLM route is the only feasible one for FYP scope (§4.1) |
| **Depth**                             | **Not done** (future work)                            | Planned: binary superficial/deep                                                                    | Open field — but do it with VLM+self-report, not a fine-tuned CNN (§4.2)                                        |
| **Size / calibration**                | Fiducial marker, r=0.73, ±5 mm                             | Not in scope                                                                                        | **Their advantage.** Optional borrow or explicitly de-scope (§4.3)                                         |
| **Recommendation engine**             | **Rule-based decision tree only**                     | **Hybrid: rules + RAG** over 9-source KB, + patient free-text notes                           | **You're ahead** — this is your core novelty                                                               |
| **Evidence grounding / traceability** | None (rules have no citations)                              | `[Source N]` citations to MOH/WCM/SFP/DyaMed                                                      | **You're ahead**                                                                                            |
| **Image at recommendation stage**     | No — recommendation runs off structured CV labels          | **FYP2: VLM caption cross-validates CV labels**                                               | **You're ahead** (this is the multimodal contribution)                                                      |
| **Product specificity**               | MAXORB II/Ag+, PluroGel (proprietary taxonomy)              | **DyaMed**: Flaminal, Zorflex, Drawtex, Dermacyn, RenoFoam/Care (9 monographs + class bridge) | Parity in concept — yours locally validated by Ms Saw                                                            |
| **Output tiering**                    | 3-tier (cleanse → primary → secondary)                    | 9-section → FYP2 patient-friendly dual-mode                                                        | Convergent design — adopt their 3-tier spine (§4.5)                                                             |
| **Human eval**                        | Expert panel, n=23                                          | H1 with Ms Saw (Likert + multimodal vs unimodal)                                                    | Parity — yours adds the comparative arm                                                                          |
| **Deployment**                        | Flutter + Go + k8s + EMR, mobile, prospective pilot planned | FastAPI demo; mobile = SE-student collaboration                                                     | **Their advantage** — use them as your deployment blueprint (§4.6)                                        |
| **Audience**                          | Clinician-facing (EMR, oversight)                           | **Patient self-care** + clinician view                                                        | Different niche — your patient-facing framing is a distinct contribution                                         |

**Net read:** The paper is a strong, well-engineered *systems* paper with the CV breadth and deployment maturity you don't (and shouldn't try to) match in 6 months. Your differentiation is the **decision-support intelligence**: RAG grounding, citations, free-text patient context, and VLM cross-validation of CV labels — all of which the paper either lacks or explicitly defers to future work.

---

## 3. How the Paper *De-risks* Your FYP2 (use these as confidence points)

1. **Your hybrid (rules + RAG) framing is validated.** They call theirs a "hybrid assessment framework" too — CV outputs + clinician inputs + rules. You extend "hybrid" to mean **rules + retrieval**, which is the natural next step they themselves point to.
2. **Product-specific recommendation is a published, accepted pattern.** They name MAXORB/PluroGel; you name DyaMed products. Reviewers won't see your product gallery as gimmicky — it mirrors a peer system.
3. **3-tier output (cleanse → primary → secondary) is a clinical convention**, not your invention. Adopt it; it strengthens your Part 13 schema and matches DyaMed's 4-step protocol.
4. **Expert-panel concordance (n=23) is an accepted validation method** at preprint level. Your H1 with Ms Saw is methodologically sufficient — you don't need a 200-patient trial.
5. **Etiology classification matters but is hard** — they needed 1,648 labelled images. This is your strongest argument to **not** fine-tune your own etiology CNN.

---

## 4. Realistic, Feasible Guidance Per FYP2 Component

> Guiding principle from the paper: **the expensive parts are the CV models and the labelled data.** A research lab with clinic partnerships spent 1,648 images to hit 91.75% on a 4-class problem. You are one student in 6 months. So every time the plan says "fine-tune a CV model," the realistic answer is "use a VLM unless a senior already trained it."

### 4.1 Etiology classification → **use VLM, do NOT fine-tune a CNN** ✅ decision needed

- **Why:** The paper's 91.75% required a curated 1,648-image, clinician-labelled, etiology-stratified dataset. You don't have that and can't build it in time with verification.
- **And it barely changes the output:** Ms Saw confirmed almost all etiologies share the same dressing algorithm except vascular — which your WT5/7/8 + referral logic already catches (Master Plan Part 12).
- **Recommendation:** Let the **VLM caption infer etiology + anatomical location** (zero-shot, no training), plus a one-tap "Do you have diabetes?" question. Feed as a **flag into generation** for management caveats (offloading, compression, referral), not as a dressing-category driver. This is Master Plan §Pain-Point-3 exactly — the paper *reinforces* it.
- **Viva line:** *"A dedicated etiology CNN needs ~1,600 labelled images for ~92% accuracy (Skinopathy 2026). Since etiology rarely changes the dressing (per my clinical collaborator), I get the management-caveat value at near-zero data cost using a VLM, and spend my effort on the part that actually changes the recommendation."*

### 4.2 Wound depth → **VLM estimate + patient self-report, NOT a fine-tuned binary CNN** ✅ decision needed

- **Why:** The paper — with all their resources — did **not** attempt RGB depth classification, and explicitly note depth needs multispectral / LiDAR. A binary superficial/deep CNN needs you to manually label images **and** get clinician verification, both of which are slow and the labels themselves are noisy (depth from a single 2D photo is ill-posed).
- **Recommendation:** Keep depth as **VLM caption field + optional patient self-report button** (Master Plan §Pain-Point-3). Combine into `wound_depth: superficial | cavity` → drives cavity-filler guidance in the prompt + the R6 metadata-filter ablation. No new model, no labelling project.
- **Fallback if supervisor insists on a CV model:** Frame depth-CNN as *stretch/future work*; deliver the VLM version first so the FYP isn't blocked on a labelling effort. Don't make a deliverable depend on data you don't yet have.

### 4.3 Size estimation & calibration → **optional borrow; default = explicitly de-scope** ⚠️ raise it

- The paper's **fiducial-marker calibration (Hough circle + HSV, r=0.73, ±5 mm, 93% detection)** is the most *cleanly reusable* CV idea in the paper and is well-specified enough to reimplement.
- **But** it's pure CV, it's outside your RAG/decision-support thesis, and size doesn't change the dressing *category*. It mainly helps **longitudinal monitoring**, which you've descoped (single-turn).
- **Recommendation:** Mention it as a **collaboration item with the SE/CV student** or future work. Only adopt it if your supervisor specifically wants a quantitative CV contribution in your chapter. Don't let it eat VLM/RAG time.

### 4.4 Product gallery → **feasible now; your KB already supports it** ✅ low-risk

- Your `DYAMED_clinical_protocol_kept.json` is already ingested: **22 chunks = 9 product monographs + 8 WT protocols + selection trees/T.I.M.E. map**, each product carrying `dressing_class` + `moh_category` (the brand→generic bridge, Master Plan Part 14). E.g. Flaminal Forte → *Alginate/Alginogel (moderate-heavy)*; RenoFoam → *Foam*; Zorflex → *Charcoal*.
- **Recommendation:** Build the static **dressing-class → product(s) → image/availability** JSON straight off these chunks (one afternoon). UI shows product card after the recommendation. This directly parallels the paper's product-specific output and is **lower-risk than anything CV**.
- **Safety guardrail (keep this):** Type comes from rules; product names are **quoted from KB only, never invented** (Master Plan §13.3). State this in the meeting — it's the answer to "what if it recommends the wrong brand?"

### 4.5 Output design → **adopt the paper's 3-tier spine inside your patient-friendly schema** ✅

- The paper's **(1) cleanse/prep → (2) primary dressing → (3) secondary/adjunctive** maps cleanly onto your Part 13 sections and the DyaMed 4-step protocol. Use it as the ordering backbone so the patient view reads as a sequence of actions, not a wall of text.
- Keep your **dual-mode render** (citations hidden in patient view, shown in eval/clinician view) — this is *more* advanced than the paper, which is clinician-only. It's also what keeps RAGAS Faithfulness measurable.

### 4.6 Mobile deployment → **their architecture is your blueprint; keep it out of your scope** ✅

- Concrete stack to hand the SE student: **Flutter (MVC)** frontend + **Go/Gin + PostgreSQL** (or your FastAPI) backend + **Docker/Kubernetes** + token auth + async inference orchestrator + guided capture (image-quality + marker feedback) + adaptive branching questionnaire.
- **Your FYP boundary stays:** the **AI dressing-recommendation engine** (RAG + VLM + rules). Deployment = collaboration / future work. Say this explicitly so the supervisor doesn't expand your scope into app engineering.

---

## 5. The Strategic Insight to Lead With

The paper's Discussion proposes, as **future work it has not built**:

- RAG over a wound-care KB for traceable recommendations ✅ *you did this in FYP1*
- LLM-generated narrative recommendations ✅ *your generation layer*
- Patient-specific factors (comorbidities, allergies) beyond wound appearance ✅ *your Sub-query C free-text notes*
- Expansion beyond product selection to debridement / contraindications ✅ *your G1-E pathway*

So your honest positioning is: **"The leading mobile wound platform (Jan 2026) reaches the edge of rule-based decision support and names RAG + LLM grounding as the next frontier. VerdaSense is a focused, working instance of exactly that frontier — and FYP2 pushes one step further by letting a VLM read the wound image to cross-validate the CV labels at generation time, which even their future-work section doesn't propose."**

That is a genuinely defensible novelty claim, and now it's backed by a citable contemporary paper.

---

## 6. What To Propose in Tomorrow's Meeting (agenda)

**Suggested 25–30 min structure:**

1. **FYP1 recap (3 min)** — the working unimodal system + headline metrics (CR 0.897, FA 0.814, Safety 90.6%); ablation-best config locked.
2. **Literature checkpoint (4 min)** — the Skinopathy paper; the head-to-head table (§2); the "they list RAG as future work" slide. *Frame as differentiation, not threat.*
3. **FYP2 thesis (3 min)** — multimodal RAG: VLM reads the image to cross-validate CV labels at generation; hybrid (rules + RAG) framing.
4. **The 3 decisions I want sign-off on (8 min)** — see §7.
5. **What's already done (2 min)** — v5 KB (160 chunks, 9 sources, BGE + MedEmbed), DyaMed ingested with class bridge, testset v5 in progress.
6. **Timeline + risks (4 min)** — §8 / Master Plan Part 7.
7. **Asks (2 min)** — H1 logistics with Ms Saw; whether to include size-calibration; SE-student deployment hand-off.

---

## 7. Decisions To Get Your Supervisor To Confirm

Bring these as explicit yes/no asks (don't leave them ambiguous):

| #            | Decision                                         | Your recommendation                              | Why it matters                                                               |
| ------------ | ------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| **D1** | Etiology: VLM zero-shot vs fine-tune a CNN       | **VLM** (no new dataset)                   | Paper needed 1,648 labelled imgs; etiology rarely changes dressing           |
| **D2** | Depth: VLM + self-report vs fine-tune binary CNN | **VLM + self-report** first; CNN = stretch | Labelling + clinician verification is slow; even the paper avoided RGB depth |
| **D3** | Size/calibration: include or de-scope            | **De-scope / SE collaboration**            | Pure CV, off-thesis, doesn't change dressing category                        |
| **D4** | Mobile app: your scope or SE student's           | **SE student / future work**               | Keeps you on the AI engine, not app eng                                      |
| **D5** | Primary FYP2 deliverable priority                | **G4 (VLM caption) + H1 human eval**       | These are the novel, defensible, examinable results                          |

If the supervisor pushes back on D1/D2 (wants "real CV models"), your fallback: *deliver the VLM version as the baseline that ships, position the CNN as an ablation/stretch that depends on a labelling effort you'll scope separately* — so no core deliverable is blocked on data you don't have yet.

---

## 8. Practical Planning & Risk Notes

**Sequencing (aligns with Master Plan Part 7):**

- **Now → Week 2:** finalise testset v5; verify referral/antibiotic ground truth; RCH population filter. (Foundations — unblocks everything.)
- **Weeks 2–4:** G1-E prompt (debridement WT5–8, depth, sepsis gate); re-baseline on v5. *Cheap, fast, documents Ms Saw's clinical review.*
- **Weeks 4–8:** VLM captioner (GPT-4o-V + Gemini-V), integrate into generation only (retrieval unchanged — R5 settled this). Add depth + DFU flag + product gallery JSON.
- **Weeks 8–14:** G4-A/B/C/D + R6 ablations; build adversarial-discrepancy + cavity testsets.
- **Weeks 12–18:** H1 with Ms Saw (highest-stakes deliverable — send early).
- **Weeks 18–26:** writing + viva.

**Top risks & mitigations:**

| Risk                                                     | Mitigation                                                                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Scope creep into CV model training (etiology/depth/size) | Lock D1–D3 tomorrow; default to VLM/de-scope                                                                 |
| VLM caption hallucination                                | Measure Caption Accuracy Rate + adversarial discrepancy rate; products quoted-only; type-from-rules guardrail |
| H1 depends on Ms Saw's availability                      | Send form early (Week 14), keep it ≤30 min, treat her DyaMed photos as her primary expert input              |
| testset v5 not comparable to FYP1 metrics                | State explicitly that v5 rebases generation metrics (Master Plan §13.5)                                      |
| "Someone already published this" (Skinopathy)            | Lead with §5 differentiation; cite their RAG-as-future-work line                                             |

**Slide deck (keep it ~8 slides):**

1. Title + FYP1 result headline.
2. FYP1 architecture (unimodal RAG, ablation-best).
3. The gap → FYP2 thesis (multimodal, hybrid).
4. Literature: Skinopathy paper + head-to-head table (§2).
5. "They list RAG as future work" → your differentiation (§5).
6. FYP2 architecture diagram (Master Plan Part 2).
7. The 3 decisions (D1–D3) — ask for sign-off.
8. Timeline + what's already done (v5 KB, DyaMed, testset).

---

## 9. One-Paragraph Script To Open The Meeting

> *"In FYP1 I built and ablated a working unimodal RAG system for wound-dressing recommendation — grounded in the T.I.M.E. framework, citation-traceable to Malaysian MOH and clinical guidelines, hitting CR 0.90 / FA 0.81 / Safety 90.6%, beating single-query retrieval and ungrounded prompting. For FYP2 I'm extending it to multimodal: a vision-language model reads the wound photo directly and cross-validates the upstream CV labels at the generation stage, so the system can flag, say, a missed infection. While preparing this I found a January-2026 preprint from Skinopathy that's the closest published system — and it's useful in two ways. It validates my components (tissue segmentation, etiology, product-specific recommendation, hybrid framing), and it shows where to stay realistic: they needed 1,648 labelled images for their etiology CNN, so I'll use a VLM instead of fine-tuning my own. Most importantly, their own paper names RAG over a clinical KB as future work they haven't built — which is exactly my FYP1. So I'm proposing to spend FYP2 on the parts that are genuinely novel and feasible: the VLM cross-validation layer, the DyaMed product grounding, the patient-friendly output, and a human clinical evaluation with my collaborator. I have three scope decisions I'd like your sign-off on today."*

---

## 10. FYP2 Scope Confirmation Checklist (tick through with supervisor)

Use this as a live tick-list in the meeting. Group A = "confirm these are the deliverables"; B = "already built — show in demo"; C = "decisions still open" (mirrors §7 D1–D5); D = "confirm these are OUT"; E = "clinical-validation items for Ms Saw". Boxes are for the supervisor to tick.

### A — Core FYP2 deliverables (confirm IN scope)

- [ ] **Multimodal RAG** — VLM reads the wound photo and feeds the *generation* stage only (retrieval unchanged; R5/Paradigm B). *Primary contribution.*
- [ ] **VLM ↔ CV cross-validation** — system flags discrepancies between CV labels and the visual appearance (e.g. missed infection). *Core novelty claim.*
- [ ] **Etiology via VLM** (zero-shot, **no CNN trained**) — used for management caveats + referral, not as the dressing driver.
- [ ] **Wound depth** = VLM estimate + patient self-report (**no depth CNN, no labelling project**).
- [X] **Patient-friendly output (G1-F)** — short, plain-language, dual-mode render (Dev = evidence/citations, Prod = product gallery).
- [ ] **DyaMed product grounding + static gallery** (placeholder images for now).
- [X] **KB v5 (BGE)** is the active FYP2 store — 160 chunks / 9 sources.
- [X] **H1 human clinical evaluation with Ms Saw** — *highest-priority deliverable.*
- [ ] **Ablations:** G1-E/G1-F (prompt), G4-A/B/C/D (caption), R6 (depth filter), optional R-KB (pruning).

### B — Already implemented in the prototype (confirm enough for demo)

- [X] `wound_app_multimodal.py` + `wound_index_multimodal.html` running (port 8001), beside the untouched unimodal pair.
- [X] VLM caption + etiology + depth shown in UI; **Multimodal On/Off** live A/B (= G4-A demo).
- [X] **Dev/Prod toggle**; **token-by-token streaming**; exudate-aware gallery; contraindication consistency guard.

### C — Decisions still open (get an explicit yes/no — see §7)

- [ ] **D1** Etiology = VLM (recommended) vs train a CNN.
- [ ] **D2** Depth = VLM + self-report (recommended) vs train a binary CNN (stretch only).
- [ ] **~~D3~~** ~~Size/fiducial calibration —~~ **~~de-scope~~** ~~(recommended) vs include as CV collaboration.~~
- [ ] **D4** Mobile app = SE-student / future work (recommended) vs in your scope.
- [ ] **D5** Primary deliverable priority = **G4 caption + H1** (recommended).
- [ ] **D6 (new)** Real product-gallery images & purchase links — future work vs in scope this semester.

### D — Confirm explicitly OUT of scope

- [ ] ~~Conversational / multi-turn RAG (descoped → future work).~~
- [ ] Fine-tuning any new CV model (etiology / depth / size).
- [ ] Wound size estimation & fiducial calibration (unless D3 flips).
- [ ] Mobile deployment & cloud/EMR backend (SE-student collaboration).
- [ ] ~~Longitudinal / serial-image healing tracking~~.

### E — Clinical-validation items to send Ms Saw (H1 Part D — KB reconciliation)

- [ ] **C1 — WT1 / charcoal:** does the MOH "charcoal" exclusion cover low-adherent carbon contact layers (Zorflex LA), or only odour-control charcoal dressings? *(KB source conflict; default = follow MOH, drop Zorflex for WT1.)*
- [ ] **C2 — WT2 high-exudate secondary:** prefer **Drawtex** (WT2 protocol) or **Gauze & Gamgee** (exudate tree)? *(Both grounded in her DyaMed material.)*
- [ ] Confirm we should **not edit her transcribed protocol chunks** to resolve conflicts — reconcile at generation instead (source-fidelity).

> Full detail on every item above is in `VerdaSense_FYP2_Master_Plan.md` **Part 17** (prototype build + guardrails + the two KB source-conflict findings).

---

*Generated as a meeting-prep companion to `VerdaSense_FYP2_Master_Plan.md`. Cross-reference Parts 1, 2, 7, 12, 13, 14, 17 of the master plan for full detail on each FYP2 component.*
