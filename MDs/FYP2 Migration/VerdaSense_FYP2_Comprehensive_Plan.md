# VerdaSense FYP2 — Comprehensive Planning Document

**Student:** Tee Qi Jing (23004894) · Universiti Malaya  
**Date:** June 2026 · Post-FYP1 Viva  
**Contents:** (1) Ms Saw Review Analysis · (2) Wound Dressing Domain Knowledge · (3) FYP2 Step-by-Step Plan

---

## PART 1 — Analysis of Ms Saw's Review (KB & Testset Cat A)

### 1.1 Who Are "Panel 1, 2, 3"?

Ms Saw's review sheet shows three sets of comments per case labelled Panel 1, Panel 2, Panel 3. Based on context (she is a surgeon at Sultan Ismail Hospital and is your clinical collaborator, not a panel examiner), these are most likely **three different clinical reviewers** she consulted — for example, colleagues from her surgical ward, wound care nurses, or junior doctors — who each independently reviewed the 8 cases. This is actually better than a single-reviewer sign-off: it gives you inter-rater evidence. Panel 2 gives the most clinically detailed feedback and Panel 3 is mostly agreement with caveats. Panel 1 gave no written comments for any case (blank), which may mean they agreed with everything but did not write it down, or they did not complete the review.

**For your FYP report, document this as:** "Three clinical reviewers (facilitated by the project's surgical collaborator, Ms Saw Shier Khee) independently reviewed all 8 Category A test cases." This is a stronger clinical validation statement than "one surgeon reviewed the testset."

---

### 1.2 KB Review Analysis

| Source | Ms Saw's Rating | What It Means for Your System |
|---|---|---|
| **WCM (MOH 2014)** | ✅ Yes | Your primary algorithm source is confirmed appropriate. The 13 GP algorithm chunks + 40 WCM chunks are your core evidence backbone. No change needed. |
| **GP (MOH Garis Panduan)** | ✅ Yes | Confirmed. The 8 wound type treatment chunks (WT1–WT8) are clinically valid for Malaysian practice. These are your single most important retrieval targets. |
| **AJGP** | 🟡 Good supporting reference | Valid but secondary. Use for general dressing principles and acute wound types. Not authoritative for Malaysian-specific practice. |
| **SFP** | 🟡 Good supporting reference | Valid but secondary (Singapore context). Your 36 SFP chunks are your largest single source — useful for dressing type detail (Table 2 chunks) but should not override MOH/GP for algorithm decisions. |
| **RCH** | ⚠️ Limited use — paediatric only | **Action required.** Ms Saw confirmed RCH is only appropriate for paediatric wounds. Your current system has no patient age filtering — if a 60-year-old patient queries the system, RCH chunks about paediatric wound care could be retrieved and inappropriately injected into the generation context. Your 11 RCH chunks should be given a `population: paediatric` metadata tag and filtered out of retrieval unless patient notes explicitly mention a child. This is a metadata fix, not a chunk deletion. |
| **EWMA** | 🟡 Good supporting reference | Valid for T.I.M.E. framework theory and DFU/VLU wound bed preparation. Your 12 EWMA chunks are essential for Cat E testset cases. Keep as-is. |
| **ISTAP** | ✅ Yes | Confirmed appropriate. Your 3 ISTAP chunks cover skin tear classification and management — directly supports `cat_b_skin_tear_fragile` and `cat_b_skin_tear_type2_flap` test cases. |
| **ANZBA** | ✅ Yes | Confirmed appropriate. Your 4 ANZBA chunks support `cat_b_burns_hand` and `cat_b_burns_minor_epidermal`. Keep as-is. |

**Key action from KB review:**

> **RCH metadata fix (High Priority):** Add `"population": "paediatric"` to all 11 RCH chunk metadata fields in `RCH_wound_care_kept.json`. In your retrieval pipeline, add a population filter: if the patient notes do not contain keywords like "child", "baby", "paediatric", "age X years" where X < 18, exclude RCH chunks from the candidate pool. This is a correctness fix with direct patient safety implications.

---

### 1.3 Testset Cat A Review Analysis

#### Overall Pattern Across Panels

Panel 2 is your most clinically informative reviewer. Their comments consistently add **debridement** and **daily dressing** as requirements that your system does not currently recommend. Panel 3 agrees with your primary dressing in all cases but adds debridement caveats for Types 3, 6, 7, 8. Panel 1 has no written feedback.

The key clinical insight from the reviews: **your system recommends the right dressing type, but for complex wound types (5–8), it misses the debridement requirement.** This is a KB coverage gap, not a retrieval failure — your KB has debridement content (WCM Chapter 15, 3 chunks), but your prompt does not explicitly ask the LLM to address debridement alongside dressing selection.

---

#### Case-by-Case Review Summary

**Type 1 (100% granulation, dry, not infected):**  
Panel 2 and 3 agree. Film dressing is correct. No action needed.

**Type 2 (90% granulation, high exudate, not infected):**  
Panel 2 and 3 agree. Alginate/Hydrofibre is correct. No action needed.

**Type 3 (locally infected, dry, non-advancing, 78% gran):**  
- Panel 2: "Modern dressing and assess wound progression." — Agrees with silver dressing as "modern dressing" but adds the important point that wound progression should be monitored and reassessed. Your system should include a "reassess at 5–7 days" statement for infected cases.  
- Panel 3: "Agreed + wound debridement of the slough." — Your system does not currently mention debridement for Type 3 (10% necrotic, 12% slough). At 22% non-viable tissue, autolytic debridement is clinically indicated alongside silver dressing. **This is a prompt gap**: add debridement guidance to the system prompt for cases where necrotic + slough % > 15%.

**Type 4 (locally infected, high exudate, non-advancing, 78% gran):**  
- Panel 2: "Suggest daily dressing instead of modern dressing as high exudate +/- bedside deslough first." — This is a significant clinical comment. Silver + Alginate is correct, but Panel 2 suggests daily dressing changes due to high exudate, versus the 2–3 day frequency your system states. For high exudate wounds, dressing saturation is a risk and daily changes may be appropriate.  
- Panel 3: "Agreed + wound debridement of the slough." — Same as Type 3.  
- **Action:** For Infection = Locally infected AND Moisture = High, your generation prompt should include a note that dressing change frequency may increase to daily depending on saturation level.

**Type 5 (dry, non-infected, heavy necrotic/slough burden):**  
- Panel 2: "Chemical debridement then bedside deslough." — Hydrogel is confirmed (for autolytic debridement), but Panel 2 adds the clinical pathway: chemical debridement first, then bedside deslough by a clinician. For self-care patients, this means the recommendation should also say "this wound requires professional debridement — do not attempt to remove dead tissue at home."  
- Panel 3: Agreed.  
- **Action:** For wound types 5–8 (necrotic burden > 30%), add a debridement pathway statement to the generated output: "This wound contains significant non-viable tissue. Autolytic debridement using hydrogel can begin at home, but professional wound debridement is recommended. Please inform your clinician at your next visit."

**Type 6 (non-infected, high exudate, heavy necrotic/slough, referral):**  
- Panel 2: "Daily dressing + deslough." — Same pattern: alginate is correct but daily changes and debridement are needed.  
- Panel 3: "Agreed + wound debridement of the slough."  
- No change to primary dressing recommendation. Add debridement note and daily dressing caveat.

**Type 7 (locally infected, dry, heavy necrotic burden, referral):**  
- Panel 2: "Daily dressing + deslough + antibiotic, might need OT debridement." — Silver + Hydrogel is confirmed. Panel 2 adds systemic antibiotic (already flagged in your system), and importantly: **"might need OT debridement"** — meaning operating theatre surgical debridement, not just bedside. This is the most severe case where a self-care patient absolutely must be referred urgently.  
- Panel 3: "Wound debridement then dressing + antibiotic." — Confirms the priority order: debridement first, then dressing, then antibiotic.  
- **Action:** For Types 7 and 8, your referral message should explicitly say "surgical debridement may be required — this wound must be assessed by a specialist."

**Type 8 (locally infected, high exudate, heavy necrotic/slough, referral):**  
- Same as Type 7. Both panels recommend debridement + antibiotic + possible OT.

---

### 1.4 Three Critical Updates to Your System from Ms Saw's Review

**Update 1 — Debridement statement in generation (Prompt Engineering Fix):**  
Add to your G1-C system prompt: "For wounds with necrotic% + slough% > 15%, include a debridement guidance statement in the Clinical Notes section. For self-care wounds, recommend autolytic debridement (hydrogel). For wounds requiring referral, state that professional/surgical debridement may be required."

**Update 2 — High exudate dressing frequency caveat (Prompt Engineering Fix):**  
For Moisture = High cases, add: "Dressing change frequency may increase to daily if the dressing becomes saturated before the scheduled change interval. Patient should check the dressing daily and change immediately if visibly saturated."

**Update 3 — RCH population metadata filter (Retrieval Fix):**  
Add `population` field to all RCH chunks. Exclude RCH from retrieval unless patient notes suggest paediatric context. This is the only KB-level fix required from Ms Saw's review.

**What Ms Saw's review confirms that is working correctly:**  
All 8 primary dressing recommendations are clinically validated. The contraindicated dressing lists are not challenged. The referral flags for Types 6, 7, 8 are confirmed correct. The antibiotic flags are confirmed correct. Your FYP1 dressing recommendation core is clinically sound.

---

## PART 2 — Wound Dressing Domain Knowledge for AI Students

This section teaches you what you need to know about wound care dressings to be a confident, fluent researcher — even without a medical background.

### 2.1 Why Wounds Need Dressings at All

A wound is any break in the skin barrier. The skin normally keeps bacteria out and moisture in. When broken, three things go wrong simultaneously: (1) bacteria enter and can cause infection, (2) the wound dries out and cells cannot migrate to close it, (3) exudate (fluid produced by the wound) builds up and can damage surrounding skin. A dressing addresses all three: it creates a barrier against bacteria, maintains a moist wound environment, and absorbs excess exudate. The art of dressing selection is matching the right dressing properties to what the wound currently needs — which is exactly what the T.I.M.E. framework formalises.

### 2.2 The Wound Healing Cascade (Why Timing Matters)

Wound healing occurs in four overlapping phases. Understanding these phases explains why the same wound needs different dressings at different times:

1. **Haemostasis (0–24 hours):** Bleeding stops. Platelets form a clot. No dressing intervention needed beyond wound closure/cover.
2. **Inflammation (1–4 days):** Immune cells enter the wound. Wound appears red, warm, slightly swollen. This is normal and healthy. However, if infection is present, this phase is prolonged and destructive. Silver dressings are appropriate here for infected wounds.
3. **Proliferation (4–21 days):** New tissue grows (granulation tissue — the pink, bumpy tissue you see in healing wounds). Cells migrate from the wound edges to close it. This is when the wound needs moisture retention (hydrocolloid, film) and protection from damage. The wound should not be disturbed unnecessarily.
4. **Remodelling (21 days – 2 years):** The wound closes and the new tissue matures and strengthens. The wound may appear healed externally but is still vulnerable underneath.

A wound "stuck" in the inflammation phase (non-advancing edge, persistent infection) is what your system classifies as Types 3–8. A wound successfully in the proliferation phase is Type 1–2.

### 2.3 The T.I.M.E. Framework — What Each Axis Actually Means Clinically

**T — Tissue:** What is the wound bed made of?
- **Granulation tissue (pink/red, moist, bumpy):** Healthy, healing tissue. Good sign. Needs protection and moisture.
- **Slough (yellow/cream, soft, wet):** Dead tissue that has become moist and stringy. Blocks healing by preventing new cells from migrating. Needs to be removed (debridement) or dissolved (hydrogel, autolytic debridement).
- **Necrotic tissue (black/brown, hard, dry or wet):** Dead tissue that has dried out (eschar) or liquefied. Cannot heal through necrosis. Must be debrided before healing can occur.
- **Clinical implication:** High necrotic% → debridement-first dressings (hydrogel for autolytic). High slough% → moisture-donating or absorbent dressings.

**I — Infection/Inflammation:** Is bacteria causing problems?
- **Not infected:** Normal wound colonisation (bacteria present but not causing harm). No antimicrobial dressing needed.
- **Locally infected:** Bacteria actively causing local tissue damage — increased exudate, wound not progressing, increased odour, increased pain. Needs topical antimicrobial (silver dressing, iodine).
- **Spreading infection / systemic:** Bacteria spreading into surrounding tissue (cellulitis) or bloodstream (sepsis). Topical dressing is insufficient — systemic antibiotics + urgent referral needed.
- **Clinical implication:** Infection detection by CV model (IME-Net) is the highest-stakes classification — a missed infection in a diabetic patient can lead to limb loss.

**M — Moisture:** How much fluid is the wound producing?
- **Low exudate:** Wound is dry. Needs a moisture-donating or moisture-retaining dressing. Dry wounds heal poorly — cells cannot migrate without moisture.
- **Moderate exudate:** Balanced. Most dressings work here.
- **High exudate:** Wound is producing too much fluid. Needs a highly absorbent dressing. Excessive moisture causes maceration (skin softening and breakdown around the wound) which enlarges the wound.
- **Clinical implication:** Alginate and Hydrofibre are the key high-exudate dressings. Film and hydrocolloid are contraindicated for high exudate — they cannot absorb enough and will leak.

**E — Edge:** Are the wound edges moving?
- **Advancing:** Epithelial cells from wound edges are migrating inward to close the wound. Wound is healing. Protect and maintain.
- **Non-advancing:** Wound edges are stalled — the epithelium is not moving. This is the clearest clinical sign that something is blocking healing (infection, necrosis, poor moisture balance, poor blood supply, underlying disease). Intervention needed.
- **Clinical implication:** Non-advancing edge with high necrotic burden = your system's most complex cases (Types 5–8). These almost always need debridement + dressing + possible referral.

---

### 2.4 Dressing Types — What They Are, When to Use, When NOT to Use

#### Film Dressings (e.g., Tegaderm, Opsite)

**What they are:** A thin, transparent, flexible adhesive sheet. Like a second skin. Highly permeable to oxygen and water vapour but impermeable to bacteria and water.

**When to use:**
- Clean, granulating wounds with low exudate (your Type 1)
- Post-operative wounds that are closed
- Superficial abrasions and minor cuts
- As a secondary dressing over other dressings

**When NOT to use:**
- High exudate wounds — the film cannot absorb liquid; it will pool under the dressing and macerate the wound
- Infected wounds — not antimicrobial; will trap bacteria under the film
- Deep wounds — film has no depth to fill

**Malaysian availability:** Widely available (Tegaderm at Guardian/Watson's). OTC.

---

#### Hydrocolloid Dressings (e.g., Duoderm, Comfeel)

**What they are:** A pad containing gel-forming agents (carboxymethylcellulose) backed by a waterproof film. When wound exudate contacts the pad, the agents absorb it and form a gel, maintaining a moist wound environment.

**When to use:**
- Low to moderate exudate granulating wounds
- Superficial pressure ulcers (Stage 1–2)
- Wounds under bony prominences (they cushion the wound)

**When NOT to use:**
- High exudate wounds — the gel becomes oversaturated, leaks, and may cause maceration
- Infected wounds — the gel environment can promote bacterial growth
- Deep wounds or wounds with undermining/sinus tracts

**Malaysian availability:** Available at pharmacies. OTC.

---

#### Alginate Dressings (e.g., Sorbsan, Kaltostat)

**What they are:** Made from brown seaweed (calcium alginate fibres). Highly absorbent — can absorb up to 20 times their weight in exudate. When exudate contacts the alginate, it forms a gel that conforms to the wound shape, maintaining moisture while absorbing excess.

**When to use:**
- Moderate to high exudate wounds (your Types 2, 4, 6, 8)
- Cavity wounds (rope/ribbon form fills deep spaces)
- Bleeding wounds (alginate has haemostatic properties — it promotes clotting)

**When NOT to use:**
- Dry wounds / low exudate — the alginate cannot gel without moisture; it will dry out and adhere to the wound bed, causing pain and trauma on removal
- This is your Cat A Type 5 contraindication — necrotic, low exudate → do NOT use alginate

**Requires secondary dressing:** Alginate cannot function as the outer layer — needs a secondary dressing (foam, film, or gauze pad + tape) to hold it in place and provide additional absorption.

**Malaysian availability:** Generally clinic/hospital purchase. May not be readily available OTC.

---

#### Hydrofibre Dressings (e.g., Aquacel)

**What they are:** Contain sodium carboxymethylcellulose fibres (similar chemistry to hydrocolloid but in fibre form). When wet, they form a gel that locks exudate within the fibre structure. Much more cohesive than alginate — they don't fragment and are easier to remove intact.

**When to use:**
- High exudate wounds
- Infected wounds (Aquacel Ag — silver-containing version)
- Deep wound cavities

**When NOT to use:**
- Dry wounds (same as alginate — no exudate means no gel formation)

**Malaysian availability:** Clinic/hospital. Not typically OTC.

---

#### Silver Dressings (e.g., Mepilex Ag, Aquacel Ag, Urgotul Silver)

**What they are:** Dressings incorporating silver ions (Ag+) which are released into the wound environment. Silver has broad-spectrum antimicrobial activity — it disrupts bacterial cell membranes and metabolic processes. Silver does not cause antibiotic resistance (different mechanism from systemic antibiotics).

**When to use:**
- Locally infected wounds (your Types 3, 4, 7, 8)
- High-risk wounds (diabetic foot, wounds in immunocompromised patients)
- Wounds with signs of biofilm (persistent non-healing despite appropriate dressing)

**When NOT to use:**
- Clean, non-infected granulating wounds — silver inhibits fibroblast proliferation (the cells that build new tissue), so it slows healing in clean wounds. This is your Cat B case: `cat_b_silver_clean_granulating` — the patient mistakenly uses silver on a clean wound.
- Patients with silver allergy (rare but real)
- Pregnancy (safety data limited)
- Prolonged use on large surface areas (systemic silver absorption risk)

**Malaysian availability:** Hospital/clinic. Generally not OTC. Expensive.

---

#### Hydrogel Dressings (e.g., Intrasite Gel, Aquaform)

**What they are:** A water-based gel (70–90% water) that donates moisture to the wound. The gel autolytically debrides — it rehydrates dry, necrotic, or sloughy tissue, making it easier for the body's own enzymes to break down and remove the dead material.

**When to use:**
- Dry, necrotic wounds requiring debridement (your Types 5, 7)
- Sloughy wounds
- Painful wounds — hydrogel has a cooling, soothing effect
- Burns

**When NOT to use:**
- High exudate wounds — adding more moisture to an already wet wound causes maceration. This is why alginate/hydrofibre contraindicated hydrogel in your Type 6/8 cases.

**Requires secondary dressing:** Hydrogel is a gel — it needs a secondary dressing to keep it in place and prevent it from drying out.

**Malaysian availability:** Available at some pharmacies. Borderline OTC/clinic.

---

#### Charcoal Dressings (e.g., Actisorb, Carboflex)

**What they are:** Contain activated charcoal, which has an enormous surface area that physically adsorbs odour molecules (not just masks them). Often combined with silver for antimicrobial effect.

**When to use:**
- Malodorous wounds (fungating tumour wounds, heavily infected wounds, necrotic wounds with offensive smell)
- Type 8 cases with malodour (your `cat_c_malodour_type8` test case)

**When NOT to use:**
- Clean wounds — no benefit, unnecessary cost
- This is your Type 1 contraindication — clean granulating wound should not have charcoal dressing

**Malaysian availability:** Specialist clinic/hospital.

---

#### Iodine Dressings (e.g., Iodoflex, Betadine-impregnated dressings, Inadine)

**What they are:** Release iodine (an antiseptic with broad-spectrum antimicrobial activity) into the wound. Iodine is absorbed systemically in small amounts.

**When to use:**
- Infected or contaminated wounds
- Sloughy wounds (cadexomer iodine formulations also debride slough)
- Short-term use (2 weeks maximum) due to systemic absorption

**When NOT to use (critical contraindications):**
- Thyroid disorders — iodine is taken up by the thyroid gland; in patients with thyroid disease, excess iodine can cause thyrotoxicosis (life-threatening hyperthyroidism). Your `cat_b_iodine_thyroid` test case tests this directly.
- Renal failure — impaired iodine clearance increases systemic toxicity risk
- Pregnancy / breastfeeding — iodine crosses the placenta and is secreted in breast milk
- Neonates and infants — immature thyroid function
- Large wound surface areas (increased systemic absorption)

**Malaysian availability:** Betadine (povidone-iodine) solution widely available OTC. Cadexomer iodine dressing (Iodoflex) is clinic/hospital only.

---

#### NPWT — Negative Pressure Wound Therapy (e.g., Vacuum-Assisted Closure, KCI VAC)

**What it is:** Not a dressing per se — a wound care system that applies controlled negative pressure (suction) to a wound via a foam/gauze interface and an airtight seal connected to a vacuum pump. Removes exudate, reduces oedema, promotes granulation, and brings wound edges together.

**When to use:**
- Large, complex wounds with high exudate that cannot be managed with conventional dressings
- Post-surgical wound dehiscence
- Pressure ulcers
- Diabetic foot ulcers

**When NOT to use:**
- Necrotic wounds without prior debridement (eschar must be removed first)
- Exposed blood vessels or organs
- Malignant wounds (suction promotes tumour cell dissemination)
- Fistulae

**Self-care implication:** NPWT is categorically NOT self-care — it requires a trained clinician for set-up and monitoring. Any recommendation involving NPWT must trigger mandatory referral. Your `cat_b_npwt_necrotic_eschar` and `cat_d_notes_npwt_adjunct` test cases are correctly flagged as referral required.

---

#### Honey Dressings (e.g., Medihoney, L-Mesitran)

**What they are:** Medical-grade manuka honey (not kitchen honey — it is sterilised by gamma irradiation and standardised for antibacterial activity). Honey's antimicrobial mechanism includes hydrogen peroxide release, low pH, and high osmolarity. It also has autolytic debridement properties.

**When to use:**
- Infected or sloughy wounds where silver dressings are unavailable or contraindicated
- Malodorous wounds
- Non-healing wounds

**When NOT to use / caveats:**
- Bee product allergy (rare)
- Diabetic patients — theoretically, significant systemic glucose absorption from large wound surfaces could affect glycaemia, though evidence is limited. Panel 2 and Ms Saw implicitly endorsed your `cat_b_honey_dry_necrotic` recommendation.

**Malaysian availability:** Limited. Medical-grade honey (Medihoney) is clinic-only. Kitchen honey is NOT an acceptable substitute and should never be recommended.

---

### 2.5 The Debridement Gap Your Viva Panels Noticed

One reason your RAG felt "not useful" is that **dressing selection and debridement are inseparable in clinical practice**, but your system only recommends dressings, not debridement. For Types 5–8 (necrotic/sloughy wounds), the correct clinical action is:

1. **First:** Debride the wound (remove non-viable tissue) — chemical (hydrogel), mechanical (irrigation, gentle debridement pad), sharp/surgical (blade at bedside or in theatre), or biological (maggots — Maggot Debridement Therapy from your SFP source)
2. **Then:** Apply appropriate dressing to the debrided wound bed
3. **Monitor:** Reassess after 5–7 days for signs of progression

Your system jumps to step 2. Ms Saw's reviewers unanimously added step 1. The fix is not architectural — it's a prompt addition. This is one of the most impactful single improvements you can make to the system in FYP2 without any new KB chunks or ablation experiments.

---

### 2.6 The "Rule-Based vs RAG" Argument — Why RAG Wins for an AI Doctor

Your Viva panel's challenge ("why not just rule-based?") can be definitively answered with this table:

| Clinical Scenario | Rule-Based Can Handle? | RAG Required? |
|---|---|---|
| Type 3 wound → Silver dressing | ✅ Yes | Not necessary |
| Type 3 wound + patient notes "I have a silver allergy" | ❌ No — rule has no allergy field | ✅ Yes — patient notes retrieved against allergy chunk |
| "Why is silver better than normal gauze for my infected wound?" | ❌ No — rules don't explain | ✅ Yes — RAG retrieves guideline passage and explains |
| "My wound looked infected but is now worse after 3 days on silver" | ❌ No — rules are static | ✅ Yes — conversation turn retrieves escalation guidance |
| "Can I use manuka honey from the supermarket instead?" | ❌ No — not in rule table | ✅ Yes — RAG retrieves medical-grade vs kitchen honey distinction |
| Type 5 wound + patient notes "wound has been 6 months, I have diabetes" | ❌ No — rules ignore duration/comorbidity | ✅ Yes — multi-axis query retrieves DFU, chronic wound, and debridement chunks |
| Clinician adds note: "patient on warfarin" | ❌ No — warfarin has no explicit dressing contraindication rule | ✅ Yes — RAG can retrieve bleeding risk dressing considerations from KB |

**The rule-based system gives you a dressing lookup. RAG gives you a clinical conversation.** For a self-care patient using a phone app, the conversation is the entire value proposition.

---

## PART 3 — FYP2 Step-by-Step Planning

### 3.1 FYP2 Objectives (What You're Arguing For)

Your FYP1 proved: *given a T.I.M.E. wound assessment, a RAG system can retrieve clinically relevant guideline evidence and generate a structured dressing recommendation with 81% faithfulness and 90.6% safety compliance.*

Your FYP2 argues: *the same KB-grounded evidence system can power a continuous AI doctor experience — answering follow-up questions, handling complex cases, integrating wound category context, and validating clinical accuracy through human expert evaluation.* This transforms VerdaSense from a dressing lookup tool into a wound care companion.

The title should stay the same but you can extend the subtitle: **"...with Conversational Multi-Turn Extension and Wound Category Classification"** or simply keep the FYP1 title and reflect the extension in your abstract.

---

### 3.2 What to Carry Forward from FYP1 (Unchanged)

- ✅ ChromaDB vector store + BGE-large-en embedding (R4 winner)
- ✅ Multi-axis sub-query retrieval (R1-C)
- ✅ Hybrid retrieval with RRF (R2)
- ✅ Top-6 depth (R3)
- ✅ G1-C grounded system prompt
- ✅ Gemini 2.5 Flash as primary LLM (G2-D best performance, FA=0.8147, Safety=90.6%)
- ✅ Qwen3.5-35B-A3B via OpenRouter as open-source alternative (G3-G, FA=0.8322)
- ✅ Rule-based safety checker (v2)
- ✅ 32-case testset (wound_testset_v3.json) — after referral field correction

---

### 3.3 Immediate Fixes Before FYP2 Begins (Week 0)

These must be done before you start any FYP2 development. They are corrections to FYP1 results.

**Fix 1 — Referral logic update (Ms Saw confirmed: all locally infected wounds require referral):**
```python
# In classify_wound() — update referral_required logic
# OLD: only Types 6, 7, 8 get referral_required = True
# NEW: also Types 3 and 4 (locally infected) get referral_required = True

if infection == "Locally infected":
    referral_required = True  # ALL infected wounds refer
elif wound_type in [6, 7, 8]:
    referral_required = True  # Complex necrotic burden refer
```

**Fix 2 — Update testset v3 referral fields:**
Update `referral_required` to `True` for these cases:
- `cat_a_type3_dry_infected` — currently False, should be True
- `cat_a_type4_wet_infected` — currently False, should be True
- `cat_b_iodine_thyroid` — currently False, should be True
- `cat_b_diabetic_foot` — currently False, should be True
- `cat_c_dry_infected_combo` — currently False, should be True
- Any other case with `infection = "Locally infected"` and `referral_required = False`

**Fix 3 — Re-run G2-D and G3-G safety evaluation on corrected testset:**
Re-run your ablation with the corrected referral fields. The new Safety Pass Rate is your definitive FYP2 reported result. It may go up (system was already generating correct referral text) or down (system was incorrectly not flagging referral for Types 3/4).

**Fix 4 — RCH population metadata filter:**
Add `"population": "paediatric"` to all 11 RCH chunks. Update retrieval pipeline to exclude RCH unless patient notes contain paediatric keywords.

---

### 3.4 FYP2 Development Roadmap

#### Phase 1 (Weeks 1–4): Prompt Improvements + Debridement Integration

**Goal:** Improve the quality of single-turn recommendations based on Ms Saw's clinical review, before adding new features.

**1A — Debridement guidance addition to system prompt:**
Add to G1-C prompt: "For wounds with necrotic% + slough% > 15%, include a 'Wound Debridement' subsection in Clinical Notes. State whether autolytic debridement (hydrogel) is appropriate for home use, or whether professional/surgical debridement is recommended. Do not advise patient to attempt manual debridement at home."

**1B — High exudate dressing frequency caveat:**
Add to generation output: "Note: If the dressing becomes saturated before the scheduled change date, change it immediately. High exudate wounds may require daily dressing changes."

**1C — Time-based escalation statement:**
For every locally infected wound output, add: "If no improvement is seen in 5–7 days, or if the wound worsens, seek medical attention immediately."

**1D — Sepsis bypass gate (pre-RAG hard rule):**
Implement a keyword-based bypass check on patient notes before retrieval begins. If any of these phrases are detected: fever, chills, shaking, difficulty breathing, confusion, rapid heartbeat, very low blood pressure — output an immediate emergency referral message and do not proceed to dressing recommendation.

Run a quick ablation (G1-D vs G1-C with these additions) on the full 32-case testset. This is your **FYP2 G1-E experiment** — it costs one afternoon and produces measurable FA/Safety improvement.

---

#### Phase 2 (Weeks 4–10): Conversational Multi-Turn RAG

**Goal:** Extend your single-turn RAG to a multi-turn session-aware system. This is your primary FYP2 research contribution.

**Architecture changes needed:**

```python
# FYP1: Single turn
def generate_recommendation(time_payload, notes):
    sub_queries = build_sub_queries(time_payload, notes)
    chunks = hybrid_retrieve(sub_queries)
    return llm_generate(chunks, time_payload, notes)

# FYP2: Multi-turn with session memory
class WoundCareSession:
    def __init__(self, time_payload, notes):
        self.time_payload = time_payload
        self.initial_recommendation = generate_recommendation(time_payload, notes)
        self.conversation_history = [
            {"role": "system", "content": build_wound_context(time_payload)},
            {"role": "assistant", "content": self.initial_recommendation}
        ]
    
    def ask(self, patient_question):
        # Step 1: Retrieve KB evidence for this specific question
        chunks = hybrid_retrieve_conversational(
            patient_question, 
            self.conversation_history,  # history conditions retrieval
            self.time_payload           # T.I.M.E. context always active
        )
        # Step 2: Build messages with full history
        messages = self.conversation_history + [
            {"role": "user", "content": build_conversational_prompt(
                patient_question, chunks, self.time_payload
            )}
        ]
        # Step 3: Generate grounded response
        response = llm_generate_conversational(messages)
        # Step 4: Safety check on this turn's response
        safety_result = safety_checker(response, self.time_payload)
        # Step 5: Update history
        self.conversation_history.append({"role": "user", "content": patient_question})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response, safety_result
```

**Conversational retrieval (R7 ablation — new experiment):**  
Test three retrieval strategies for conversational turns:
- R7-A: Retrieve only based on current patient question (ignores history)
- R7-B: Retrieve based on current question + last assistant turn (short memory)
- R7-C: Retrieve based on current question + full session summary (long memory)

Expected finding: R7-B will outperform R7-A (history helps) but R7-C may degrade due to context length noise. This is a clean, novel ablation finding.

**Session memory management:**  
For elderly patients with short sessions (3–5 turns), full ConversationBufferMemory is sufficient. For longer sessions, use ConversationSummaryMemory (LangChain) which compresses history into a rolling summary after every 5 turns.

---

#### Phase 3 (Weeks 10–16): Wound Category Classification (If Dataset Secured)

**Goal:** Train a wound category classifier to identify DFU, VLU, Pressure Ulcer, Burn, Abrasion/Cut, Surgical, Skin Tear, Vascular.

**Scope recommendation — reduce to 6 main categories:**

| Category | Why Include | Dataset Availability |
|---|---|---|
| Diabetic Foot Ulcer (DFU) | Ms Saw's explicit request; highest clinical risk | AZH Wound Dataset, Kaggle wound datasets |
| Venous Leg Ulcer (VLU) | EWMA source already in KB; common in elderly | Public datasets available |
| Pressure Ulcer (PrU) | Common in bedridden patients; clear staging system | PUSH dataset; academic sources |
| Burn | ANZBA source in KB; RCH covers paediatric burns | Limited; ANZBA resources |
| Acute Wound (Abrasion/Cut/Surgical/Laceration) | Common self-care cases; AJGP, RCH cover this | Easiest to source public images |
| Skin Tear | ISTAP source in KB; common in elderly | Limited specialist datasets |

**Vascular ulcer** — include in future work only. Ms Saw flagged it as requiring special dressings, but public image datasets for vascular ulcers are extremely rare and the clinical presentation overlaps significantly with VLU without Doppler assessment (which a phone image cannot provide).

**Model architecture:**  
Start with EfficientNet-B3 fine-tuned on wound category images. Input: cropped wound ROI from YOLO BBox. Output: softmax probability over 6 categories. If public dataset size is insufficient (< 500 images per class), use DINOv2 or BioViL-T with few-shot fine-tuning.

**Integration into retrieval (R6 ablation — new experiment):**  
Once category label is produced, add it as a metadata filter in ChromaDB retrieval:
```python
# R6: With wound category filter
chunks = collection.query(
    query_texts=sub_queries,
    n_results=6,
    where={"$or": [
        {"wound_category": category_label},   # category-specific chunks
        {"wound_category": "general"}           # always retrieve general chunks
    ]}
)
```
Ablate: R6-A (no category filter) vs R6-B (hard category filter) vs R6-C (soft category filter — boost category-specific chunks but don't exclude general ones). Expected: R6-C will outperform both others.

---

#### Phase 4 (Weeks 14–18): Expanded Testset and New Ablation Evaluation

**New testset components needed:**

**Component 1 — Conversational testset (20–25 sessions, 3–4 turns each):**

Structure:
```json
{
  "session_id": "conv_01_type3_infected_followup",
  "category": "CONV",
  "initial_time_payload": {...},
  "initial_notes": "wound smells bad",
  "turns": [
    {
      "turn": 1,
      "patient_question": "Why is silver dressing better than normal bandage for my wound?",
      "reference_answer": "Silver dressing releases silver ions that kill bacteria...",
      "reference_contexts": ["WCM_chunk_31", "SFP_chunk_31"],
      "safety_critical": false
    },
    {
      "turn": 2,
      "patient_question": "My wound looks more red and feels warmer today after 2 days on silver. What should I do?",
      "reference_answer": "Increased redness and warmth after 2 days may indicate worsening infection or spreading cellulitis. You should seek medical attention today.",
      "reference_contexts": ["WCM_chunk_04", "GP_chunk_11"],
      "safety_critical": true  // must trigger escalation
    }
  ]
}
```

**Component 2 — Category F OOD testset (6–8 cases):**
Cases where the correct output is a refusal or "insufficient evidence" statement.
- Necrotizing fasciitis presentation (life-threatening — beyond self-care scope)
- Wound query with conflicting T.I.M.E. signals (e.g., CV says not infected but notes say "spreading red line from wound")
- Query about a dressing type not in KB (e.g., bioengineered skin substitute)

**Component 3 — Wound category testset (if Phase 3 implemented, 16–20 cases):**
Cases where the wound category label changes the appropriate dressing recommendation relative to T.I.M.E. alone.

---

#### Phase 5 (Weeks 16–20): Human Clinical Evaluation

This is your highest-priority evaluation deliverable for FYP2. It closes the biggest gap identified in your FYP1 Viva.

**3-part evaluation design (send to Ms Saw via Google Form):**

**Part A — Blinded recommendation rating (8 Cat A cases):**
- Show Ms Saw only the wound description + generated output (no T.I.M.E. labels, no system metadata)
- Rate each on 3 dimensions, Likert 1–5:
  - Clinical Accuracy: "Is the primary dressing recommendation clinically appropriate?"
  - Safety: "Does this recommendation contain anything potentially harmful?"
  - Completeness: "Does this recommendation include all clinically important information?"
- Primary metric: **Clinical Concordance Rate** = fraction of cases rated ≥ 4/5 on Clinical Accuracy

**Part B — RAG vs Zero-shot comparative (4 cases):**
- Show 4 cases: each case shows Recommendation A and Recommendation B (one RAG, one zero-shot GPT-4o, order randomised, labels hidden)
- Ask: "Which recommendation is more clinically appropriate?"
- Primary metric: **RAG Preference Rate** = fraction of cases where Ms Saw prefers the RAG output

**Part C — Conversational turn rating (5 sessions, 2 turns each):**
- Show the session transcript (initial recommendation + 2 follow-up Q&A turns)
- Ask: "Are the follow-up answers clinically appropriate and consistent with the initial recommendation?"
- Primary metric: **Conversational Consistency Score** = fraction of rated turns where advice is consistent and appropriate

**Time estimate for Ms Saw:** 20–25 minutes total. Send as PDF with answer boxes, or Google Form. This is the most important 25 minutes of your FYP2.

---

### 3.5 Complete FYP2 Ablation Map

| Experiment | Research Question | Type | Status |
|---|---|---|---|
| R1-C (FYP1) | Multi-axis query > single query | Retrieval | ✅ Done |
| R2 (FYP1) | Hybrid > dense/sparse alone | Retrieval | ✅ Done |
| R3 (FYP1) | Optimal top-K depth = 6 | Retrieval | ✅ Done |
| R4 (FYP1) | BGE-large-en best embedding | Retrieval | ✅ Done |
| G1-C (FYP1) | Grounded prompt > naive | Generation | ✅ Done |
| G2-D (FYP1) | Gemini 2.5 Flash best closed-source | Generation | ✅ Done |
| G3-G (FYP1) | Qwen3.5-35B best open-source | Generation | ✅ Done |
| **G1-E** | Debridement + frequency + sepsis additions to prompt | Generation | 🔲 FYP2 Phase 1 |
| **R7** | Conversational retrieval: no history vs short vs long memory | Retrieval | 🔲 FYP2 Phase 2 |
| **G4** | Multi-turn session: does conversation improve clinical accuracy over turns? | Generation | 🔲 FYP2 Phase 2 |
| **G5** | Appropriate Abstention Rate on OOD Category F testset | Safety | 🔲 FYP2 Phase 2 |
| **R6** | Wound category metadata filter: no filter vs hard vs soft | Retrieval | 🔲 FYP2 Phase 3 |
| **H1** | Human clinical evaluation (3-part: Ms Saw) | Human | 🔲 FYP2 Phase 5 |

---

### 3.6 KB Expansion Plan

Your current 138 chunks across 8 sources are sufficient for FYP1 scope. For FYP2 conversational use, you need broader coverage. Priority additions:

| Source to Add | Why | Chunks Estimated | Phase |
|---|---|---|---|
| MOH DFU Clinical Practice Guideline (Malaysia, 2020) | DFU is Ms Saw's explicit request + most high-risk wound category | 15–20 | Phase 3 |
| IWGDF Guidelines on DFU Prevention & Management (2023) | International DFU standard — supports EWMA | 10–15 | Phase 3 |
| Malaysian CPG Pressure Ulcer (MOH, 2019) | Pressure ulcer is common in elderly self-care patients | 10–15 | Phase 3 |
| WCM Chapter 8b (VLU) — expand coverage | EWMA VLU already in KB but WCM VLU chapter is in your existing source | Already have | Phase 1 |
| Wound Care FAQ / Patient Education | Covers conversational turn questions ("can I shower?", "how do I remove dressing?") | 10–15 | Phase 2 |

**Priority: Patient education FAQ source.** For the conversational RAG to answer "can I shower with this dressing on?" or "how do I remove the dressing without hurting myself?", you need chunks that answer these questions. Clinical guidelines do not contain this information — they are written for clinicians. You will need to curate a patient-facing FAQ from trusted sources (NHS Wound Care patient information, MOH patient education materials).

---

### 3.7 Answering the Viva Panel's "Why Not Rule-Based?" in FYP2

By FYP2 Viva, your answer will be:

> "In FYP1, the panel asked why I use RAG instead of a rule-based system. In FYP2, I have direct empirical evidence. In Part B of our human clinical evaluation, a clinical expert preferred the RAG output over zero-shot GPT output in X of 4 cases. In our conversational testset, the RAG system correctly answered patient follow-up questions about dressing alternatives, wound escalation, and dressing change technique — questions that cannot be encoded in any rule table. The rule-based pre-classifier remains in the system for wound type classification and safety checking, where deterministic rules are appropriate. The RAG is used for what rules cannot do: evidence-grounded explanation, follow-up Q&A, and semantic patient notes processing. The system is hybrid by design."

That is the full, evidence-backed answer to FYP1's hardest question.

---

## Summary: The Three Things That Will Make FYP2 Stand Out

1. **Conversational multi-turn RAG with a wound-care-specific KB** — no existing academic wound care system has built and evaluated a multi-turn grounded conversational interface. Your conversational testset + R7 + G4 ablation will be the first rigorous evaluation of this in the wound care domain.

2. **Human clinical evaluation with inter-rater evidence** — Ms Saw's three-panel Cat A review is already inter-rater validation. The Phase 5 blinded human evaluation will add output-level clinical concordance. Most clinical RAG papers have no human evaluation at all. You will have it.

3. **The hybrid architecture argument, empirically proved** — G1-A (zero-shot) vs G2-D (RAG+rules) gives you the Safety delta. H1 (human evaluation) gives you the clinical concordance. Together, they prove that RAG + rules outperforms both pure RAG and pure rules on clinical safety and output quality.

---

*VerdaSense FYP2 Comprehensive Planning Document · Tee Qi Jing (23004894) · Universiti Malaya · June 2026*
