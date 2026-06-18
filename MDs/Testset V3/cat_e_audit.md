# Category E — Gold Standard Audit Report
**Based on direct comparison of `reference` text vs actual chunk `text` + `ai_summary` content**
**All 8 source _kept.json files read verbatim. Every finding is chunk-traceable.**

---

## Legend
- ✅ Correct — grounded in cited chunk
- ⚠️ Hallucination / Ungrounded — stated in reference but NOT in any cited chunk
- 🔧 Fix needed — factual mismatch or wording contradicts the source
- 📝 POV / style issue on patient notes
- ➕ Chunk should be added to reference_contexts
- ➖ Chunk should be removed from reference_contexts
- 🛡️ contraindicated_dressings field check

---

## CASE E1 — `cat_e_vlu_chronic_ewma`
**Current reference_contexts:** `EWMA_VLU_TISSUE`, `EWMA_VLU_MOISTURE`, `EWMA_VLU_INFECTION`,
`EWMA_VLU_EDGE`, `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`

### Patient Notes POV Check
> "I am 65 years old. I have a wound on my lower left leg that has been there for about 14 weeks.
> The doctor checked the blood flow in my leg and said it is safe to use a compression bandage.
> I have been using the compression bandage for 8 weeks but the wound has not got any smaller.
> There is a lot of fluid and the skin around the wound looks white and soggy."

✅ Fully 1st-person patient-facing language. No clinical jargon. Realistic patient description. **No change needed.**

---

### Hallucination Check — E1

| Reference Statement | Source | Status |
|---|---|---|
| Wound Type 6: NV >25%, not infected, high exudate, referral required | GP_TYPE6 TEXT ✓ | ✅ |
| GP_TYPE6 dressing list: Alginate, Foam, Polymeric membrane, Hydrofiber | GP_TYPE6 TEXT ✓ | ✅ |
| GP_REFERRAL: Type 6 requires hospital referral | GP_REFERRAL TEXT ✓ | ✅ |
| "debridement of slough is the first priority" (Primary Dressing section) | EWMA_VLU_TISSUE TEXT — source says "majority of uncomplicated VLU have relatively little necrotic tissue and do NOT require debridement"; only complex ulcers of long duration may need it | 🔧 |
| Alginate: absorbs exudate, change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: manages heavy exuding wounds, reduces maceration | WCM_HYDROFIBRE TEXT ✓ — **WCM_HYDROFIBRE is NOT in reference_contexts** | ⚠️ |
| Foam as secondary: highly absorbent, change every 2–3 days | WCM_FOAM TEXT ✓ — **WCM_FOAM is NOT in reference_contexts** | ⚠️ |
| "Compression bandaging applied OVER dressing, as per VLU protocol" | EWMA_VLU_MOISTURE TEXT ✓ ("Compression therapy is fundamental to achieving moisture balance") | ✅ |
| T: "EWMA recommends mechanical or autolytic debridement first. Slough must be removed before epithelialisation can proceed." | EWMA_VLU_TISSUE TEXT — source says most VLU do NOT require debridement; autolytic debridement "is slow and clinical experience suggests this is NOT an effective form of debridement under compression" | 🔧 CONTRADICTS source |
| I: "Monitor for biofilm development" | EWMA_VLU_INFECTION TEXT — source mentions biofilm only in the general TIME framework (EWMA_TIME_FRAMEWORK), not in the VLU-specific infection chunk; EWMA_VLU_INFECTION does NOT mention biofilm specifically for VLU | ⚠️ |
| M: "white, soggy skin around the wound (periwound maceration) is present; alginate or hydrofibre manages exudate while a skin barrier protects the surrounding skin" | EWMA_VLU_MOISTURE TEXT ✓ ("Maceration may occur around the margins... white, soggy tissue"; "use paraffin-based products or zinc paste as a barrier"; "select appropriately sized dressing capable of handling high exudate levels such as foams and capillary action dressings") | ✅ (partially) |
| M: "skin barrier protects the surrounding skin" | EWMA_VLU_MOISTURE TEXT ✓ ("use paraffin-based products or zinc paste as a barrier") | ✅ |
| E: "failure to progress despite 8 weeks of compression; EWMA VLU guidelines indicate need for wound bed reassessment and consideration of advanced therapies" | EWMA_VLU_EDGE TEXT ✓ ("predict as early as the fourth week... which ulcers will fail to heal rapidly, as these patients benefit most from alternative care strategies") + EWMA_VLU_TISSUE ("the challenge for effective wound bed preparation is the early detection of those ulcers unlikely to heal by simple compression therapy alone") | ✅ |
| "Dry gauze: desiccates the wound bed and causes trauma on removal" (Contraindicated Dressings) | NOT stated in any cited chunk — EWMA_VLU_MOISTURE says "avoid hydrocolloids and films" and "adhesive dressings should be avoided" but does NOT mention dry gauze | ⚠️ |
| "Compression without prior blood flow assessment: CONTRAINDICATED" | EWMA_VLU_MOISTURE TEXT — source does NOT say compression is contraindicated without blood flow assessment; it discusses compression as cornerstone of care but does NOT state this contraindication explicitly | ⚠️ |
| Compression: "usually 1–2 times per week" (Dressing Change Frequency) | EWMA_VLU_MOISTURE TEXT — source says "bandages may need to be changed more frequently if soiled by excessive exudate" but does NOT state "1–2 times per week" as a specific frequency | ⚠️ |
| Referral: "possible advanced therapies such as growth factors, skin substitutes, or negative pressure wound therapy" | EWMA_VLU_EDGE TEXT ✓ ("Tissue-engineered products", "Growth factors", "Bioactive dressings") — these are named in the advanced therapies table | ✅ |
| Application Tips: "Apply skin barrier cream or film to white, soggy skin before applying dressing" | EWMA_VLU_MOISTURE TEXT ✓ ("use paraffin-based products or zinc paste as a barrier") — skin barrier is correct but "cream or film" is a slight expansion; source says "paraffin-based products or zinc paste" | 🔧 Minor — should specify paraffin-based or zinc paste to match KB |
| Application Tips: "compression bandaging on top of dressing — only because blood flow confirmed adequate" | EWMA_VLU_MOISTURE TEXT ✓ (compression cornerstone of care); no explicit "only because blood flow confirmed" in KB, but clinically defensible from context | 🔧 Minor |
| Clinical Notes: "Compression therapy is cornerstone of venous leg ulcer management" | EWMA_VLU_MOISTURE TEXT ✓ ("graduated, sustained multi-layer compression is the cornerstone of care") | ✅ |
| "Avoid hydrocolloids and films" for VLU | EWMA_VLU_MOISTURE TEXT ✓ ("Avoid hydrocolloids and films") | ✅ — but this is NOT in contraindicated_dressings field |
| Antibiotic: "may or may not be required, based on underlying cause" | GP_TYPE6 TEXT ✓ | ✅ |

---

### Key Issue 1 — Debridement framing CONTRADICTS EWMA_VLU_TISSUE

The reference states in Primary Dressing: *"Per EWMA VLU wound bed preparation guidelines, debridement of slough is the first priority."*

**EWMA_VLU_TISSUE TEXT explicitly states:** *"The majority of uncomplicated venous ulcers have relatively little necrotic tissue on the wound surface and **do not require debridement**."* Debridement is only recommended for complex ulcers with chronic fibrinous bases.

Likewise, Rationale T states: *"EWMA recommends mechanical or autolytic debridement first."* The source TEXT says: *"Autolytic debridement using dressings with a high water content, such as hydrogels and hydrocolloids, is slow and clinical experience suggests **this is not an effective form of debridement under compression**."*

This is a **direct factual contradiction of the source.**

**Fix:** Reframe T and Primary Dressing to reflect what the KB actually states — compression + appropriate dressings are first-line; debridement is only for complex/long-standing ulcers with chronic fibrinous base. Remove "debridement is the first priority" statement.

---

### Key Issue 2 — "Dry Gauze" Contraindication Not in KB

The reference states: *"Dry gauze: desiccates the wound bed and causes trauma on removal."* This specific contraindication is **not stated in any of the 7 cited chunks**. EWMA_VLU_MOISTURE says: *"Avoid hydrocolloids and films"* — not dry gauze.

**Fix:** Replace dry gauze with the KB-grounded contraindications. EWMA_VLU_MOISTURE explicitly states: *"Avoid hydrocolloids and films."* These should be in contraindicated_dressings and referenced in the Contraindicated Dressings section.

---

### Key Issue 3 — "Compression without blood flow assessment" Not in KB

Reference Contraindicated Dressings: *"Compression bandaging without prior blood flow assessment: compression must not be used if adequate blood flow has not been confirmed."* This specific contraindication is **not stated in any cited EWMA VLU chunk.** EWMA_VLU_MOISTURE discusses compression as cornerstone of care without stating this contraindication.

**Fix:** Remove this from Contraindicated Dressings section. The patient's notes already confirm blood flow was assessed — this contraindication does not belong in a section grounded in KB content.

---

### Key Issue 4 — Compression Frequency "1–2 times per week" Not in KB

EWMA_VLU_MOISTURE says bandages "may need to be changed more frequently if soiled" but gives **no specific weekly frequency.** The "1–2 times per week" figure is not in any cited chunk.

**Fix:** Remove the specific frequency. Replace with: *"Compression bandaging: per clinical protocol — change if soiled by excessive exudate or if limb circumference has changed significantly."*

---

### Key Issue 5 — "Biofilm development" Monitor Not in VLU Chunk

The I rationale says: *"Monitor for biofilm development."* Biofilm is mentioned only in `EWMA_TIME_FRAMEWORK` (general TIME framework), not in `EWMA_VLU_INFECTION`. The VLU-specific infection chunk discusses localised infection, cellulitis, antimicrobials, and systemic antibiotics — not biofilm monitoring specifically.

**Fix:** Remove "biofilm development" from I rationale. Replace with what EWMA_VLU_INFECTION actually states: indicators of infection in VLU (increased pain intensity or change in pain character, discoloured/friable granulation tissue, odour, wound breakdown, delayed healing).

---

### Key Issue 6 — WCM_HYDROFIBRE and WCM_FOAM Missing from reference_contexts

Hydrofibre and foam are both recommended dressings with properties and frequencies cited in the reference, but neither chunk is in reference_contexts.

**Fix:** Add `WCM_HYDROFIBRE` and `WCM_FOAM`.

---

### 🛡️ contraindicated_dressings Field

```python
"contraindicated_dressings": ["dry_gauze"]
```

**Issue:** `dry_gauze` is NOT grounded in any cited chunk (see Key Issue 2). Should be replaced with what EWMA_VLU_MOISTURE explicitly states: *"Avoid hydrocolloids and films."*

**Fix:**
```python
"contraindicated_dressings": ["hydrocolloid", "film"]
```

Both are explicitly stated in EWMA_VLU_MOISTURE TEXT: *"Avoid hydrocolloids and films."*

---

### reference_contexts Fix — E1
**Add:** `WCM_HYDROFIBRE`, `WCM_FOAM`
**Updated set:** `EWMA_VLU_TISSUE`, `EWMA_VLU_MOISTURE`, `EWMA_VLU_INFECTION`, `EWMA_VLU_EDGE`, `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

---

---

## CASE E2 — `cat_e_dfu_infected_ewma`
**Current reference_contexts:** `EWMA_DFU_INFECTION`, `EWMA_DFU_TISSUE`, `EWMA_DFU_MOISTURE`,
`EWMA_DFU_EDGE`, `AJGP_DIABFOOT`, `WCM_SILVER`, `GP_TYPE3`

### Patient Notes POV Check

**`time_payload` notes field:**
> "I have Type 2 diabetes. The wound is on the bottom of my right foot and has been there for 5 weeks.
> The skin around it is red and sore. I cannot feel pain in my foot properly. My blood sugar levels
> have been high — my last blood test showed my average blood sugar control has been poor. I am not
> using any special shoe or boot for the wound."

✅ Fully 1st-person patient-facing language. No clinical jargon. Realistic description. **No change needed.**

**`user_input` field:**
> "...No offloading device in use."

⚠️ Minor issue — "No offloading device in use" is clinical phrasing, not patient language. The `time_payload` notes version uses the better phrasing ("I am not using any special shoe or boot"). The `user_input` should match the `time_payload` notes for consistency.

**Fix:** Update `user_input` last line from `"No offloading device in use."` to `"I am not using any special shoe or boot for the wound."`

---

### Hallucination Check — E2

| Reference Statement | Source | Status |
|---|---|---|
| Wound Type 3: NV <25%, infected, dry/minimal — GP algorithm | GP_TYPE3 TEXT ✓ | ✅ |
| GP_TYPE3 dressing list: Tulle, Hydrogel, Hydrocolloid, Silver, Iodine | GP_TYPE3 TEXT ✓ | ✅ |
| "EWMA DFU guidelines provide additional DFU-specific TIME management supplementing standard GP wound type" | EWMA_DFU_TISSUE/INFECTION/MOISTURE/EDGE TEXT ✓ (all 4 EWMA DFU chunks present) | ✅ |
| Silver: bactericidal, locally acting, change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| "Per EWMA DFU infection control guidelines, local wound infection must be treated aggressively with antimicrobial dressings as first-line pending C&S results" | EWMA_DFU_INFECTION TEXT ✓ ("At initial presentation of infection it is important to prescribe wide-spectrum antibiotics and take cultures") | ✅ |
| AJGP: "primary antimicrobial dressing + secondary by exudate level; moderate exudate → silicone foam" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| AJGP: "silicone foams without borders, anchored with tape or bandages" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| "Non-bordered foam or non-adherent pad for moderate exudate — WITHOUT adhesive borders on foot" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| T: "35% non-viable — EWMA DFU recommends debridement before infection can fully resolve. Sharp or mechanical debridement preferred." | EWMA_DFU_TISSUE TEXT ✓ ("sharp debridement is the gold standard"; "diabetic foot does not tolerate sloughy, necrotic tissue, and debridement is therefore an important component") | ✅ |
| I: "wound infection drives chronicity and must be the primary treatment target" | EWMA_DFU_INFECTION TEXT ✓ ("infection is implicated in most cases that result in major amputation"; "clinical infection resulting in failure to heal must be treated aggressively and promptly") | ✅ |
| M: "non-bordered foam or non-adherent pad is appropriate" | EWMA_DFU_MOISTURE TEXT ✓ ("dressing is: easy to remove, absorbent, able to accommodate pressures of walking without disintegrating") + AJGP_DIABFOOT ✓ | ✅ |
| E: "non-advancing at 5 weeks — per EWMA DFU pathway, this triggers specialist review" | EWMA_DFU_EDGE TEXT — source discusses extrinsic/intrinsic factors; does NOT state a specific "5-week trigger" — similar issue as D2 | ⚠️ |
| "Adhesive bordered foam: pressure and skin damage risk when sensation reduced" | AJGP_DIABFOOT TEXT ✓ ("without borders and anchored with tape or bandages") | ✅ |
| "Hydrocolloid: not recommended per AJGP and EWMA DFU guidelines" | AJGP_DIABFOOT: "Not recommended for... diabetic foot ulcers" (via SFP_HYDROCOLLOID — but SFP_HYDROCOLLOID is NOT in reference_contexts for E2) | ⚠️ |
| "Systemic antibiotic required if redness is spreading" | EWMA_DFU_INFECTION TEXT ✓ ("Systemic antibiotic treatment is always indicated in the presence of cellulitis, lymphangitis and osteomyelitis") | ✅ |
| "C&S swab must be taken immediately" | EWMA_DFU_INFECTION TEXT ✓ ("At initial presentation of infection it is important to prescribe wide-spectrum antibiotics and take cultures"; "deep swabs or tissue should be taken from the ulcer after initial debridement") | ✅ |
| Referral: "poor blood sugar control and infected non-healing DFU at 5 weeks both indicate urgent MDT review" | EWMA_DFU_TISSUE TEXT ✓ ("metabolic control" as prerequisite; "wound care is more likely to fail" without the 3 basic elements); EWMA_DFU_ADVANCED TEXT ✓ ("management must involve multidisciplinary approach") | ✅ |
| Application Tips: "OFFLOADING IS MANDATORY — a total contact cast or removable cast walker is essential" | EWMA_DFU_EDGE TEXT — source says "some form of cast, adapted footwear or padding" — NOT "total contact cast or removable cast walker" specifically | ⚠️ |
| "Dressing alone will not heal a pressure wound on the bottom of the foot" | EWMA_TIME_PRACTICE TEXT ✓ ("Diabetic foot ulcers will not heal without pressure offloading and diabetic control") — **EWMA_TIME_PRACTICE is NOT in reference_contexts** | ⚠️ |
| "blood sugar control, blood flow assessment, offloading, debridement, and dressing selection must all be addressed simultaneously" | EWMA_DFU_TISSUE TEXT ✓ (three basic elements: pressure control, blood flow, metabolic control) + EWMA_DFU_ADVANCED ✓ (After TIME: "linked to pressure control and management of blood glucose and perfusion") | ✅ |
| "Specialist multidisciplinary team review is urgently required" | EWMA_DFU_ADVANCED TEXT ✓ ("management requires a multidisciplinary approach") | ✅ |

---

### Key Issue 1 — "5-week trigger" Not in EWMA_DFU_EDGE

The reference states: *"Non-advancing at 5 weeks — per EWMA DFU pathway, this triggers specialist review."* `EWMA_DFU_EDGE` TEXT discusses extrinsic and intrinsic factors causing non-advancement but **does not state a specific number of weeks as a threshold.** This is the same pattern as D2's "4-week" hallucination.

**Fix:** Remove the specific "5 weeks" trigger wording. Replace with: *"E (Edge): Non-advancing — per EWMA DFU edge advancement pathway, failure to progress indicates reassessment of extrinsic factors (repeated undetected trauma due to neuropathy, ischaemia, poor metabolic control) and specialist review is required."*

---

### Key Issue 2 — Hydrocolloid Contraindication Missing SFP_HYDROCOLLOID

The reference states: *"Hydrocolloid on diabetic foot ulcers: not recommended per AJGP and EWMA DFU guidelines."* The AJGP DFU chunk (`AJGP_DIABFOOT`) does not explicitly say hydrocolloid is contraindicated. The explicit "not recommended for... diabetic foot ulcers" language is in `SFP_HYDROCOLLOID` TEXT. **SFP_HYDROCOLLOID is not in reference_contexts for E2.**

**Fix:** Add `SFP_HYDROCOLLOID` to reference_contexts.

---

### Key Issue 3 — "Total contact cast or removable cast walker" Not in KB

Application Tips state: *"OFFLOADING IS MANDATORY — a total contact cast or removable cast walker is essential."* `EWMA_DFU_EDGE` TEXT says: *"Redistribute plantar pressures evenly by applying **some form of cast, adapted footwear or padding**."* The specific named devices are not in the KB — identical issue as D2.

**Fix:** Replace with KB-grounded wording: *"OFFLOADING IS MANDATORY — some form of cast, adapted footwear, or padding must be applied to redistribute plantar pressures evenly. Dressing alone will not heal this wound without offloading."*

---

### Key Issue 4 — "Dressing alone will not heal" Grounded in EWMA_TIME_PRACTICE (Not in Contexts)

Clinical Notes state: *"Offloading is the single most critical intervention for a neuropathic plantar DFU."* This is directly grounded in `EWMA_TIME_PRACTICE` TEXT: *"Diabetic foot ulcers will not heal without pressure offloading and diabetic control."* However, `EWMA_TIME_PRACTICE` is **not in reference_contexts**.

**Fix:** Add `EWMA_TIME_PRACTICE` to reference_contexts.

---

### Key Issue 5 — "Wound infection drives chronicity" — Verify Grounding

The reference states: *"Per EWMA DFU guidelines, wound infection drives chronicity and must be the primary treatment target."* `EWMA_DFU_INFECTION` TEXT states: *"clinical infection resulting in failure to heal must be treated aggressively and promptly"* and infection is *"implicated in most cases that result in major amputation."* This is sufficiently grounded. ✅ No change needed.

---

### 🛡️ contraindicated_dressings Field

```python
"contraindicated_dressings": ["bordered_foam", "hydrocolloid"]
```

Both are grounded:
- `bordered_foam`: AJGP_DIABFOOT TEXT ✓ ("without borders and anchored with tape or bandages")
- `hydrocolloid`: SFP_HYDROCOLLOID TEXT ✓ ("Not recommended for... diabetic foot ulcers") — **but SFP_HYDROCOLLOID is not yet in reference_contexts**

**Fix:** Add `SFP_HYDROCOLLOID` to reference_contexts to ground the hydrocolloid contraindication. The `contraindicated_dressings` field values are correct — just needs the supporting chunk added. ✅

---

### reference_contexts Fix — E2
**Add:** `SFP_HYDROCOLLOID`, `EWMA_TIME_PRACTICE`
**Updated set:** `EWMA_DFU_INFECTION`, `EWMA_DFU_TISSUE`, `EWMA_DFU_MOISTURE`, `EWMA_DFU_EDGE`, `AJGP_DIABFOOT`, `WCM_SILVER`, `GP_TYPE3`, `SFP_HYDROCOLLOID`, `EWMA_TIME_PRACTICE`

---

## Summary Table — All Category E Issues

| Case | Hallucination / Ungrounded | Missing Chunk | Reference Text Fix | POV Issue |
|---|---|---|---|---|
| E1 `vlu_chronic_ewma` | 🔧 Debridement framing contradicts EWMA_VLU_TISSUE; ⚠️ "dry gauze" not in KB; ⚠️ compression contraindication not in KB; ⚠️ "1–2x per week" frequency not in KB; ⚠️ "biofilm" not in VLU chunk | ➕ `WCM_HYDROFIBRE`, `WCM_FOAM` | Remove/reframe debridement priority; remove dry gauze contraindication; remove compression frequency; fix biofilm reference; fix skin barrier to "paraffin-based or zinc paste" | None |
| E2 `dfu_infected_ewma` | ⚠️ "5-week trigger" not in EWMA_DFU_EDGE; ⚠️ "total contact cast/removable cast walker" names not in KB; ⚠️ "dressing alone won't heal" grounded in EWMA_TIME_PRACTICE (missing); hydrocolloid grounding needs SFP_HYDROCOLLOID | ➕ `SFP_HYDROCOLLOID`, `EWMA_TIME_PRACTICE` | Remove "5 weeks" trigger; rephrase offloading device names to match KB | 📝 Minor: `user_input` last line uses clinical phrasing "No offloading device in use" — fix to match `time_payload` notes |

---

## Required Fixes — Category E (Precise)

### E1 — `cat_e_vlu_chronic_ewma`

**reference_contexts — add:**
```python
"reference_contexts": [
    ctx(EWMA_VLU_TISSUE), ctx(EWMA_VLU_MOISTURE), ctx(EWMA_VLU_INFECTION), ctx(EWMA_VLU_EDGE),
    ctx(GP_TYPE6), ctx(GP_REFERRAL), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM),
],
```

**contraindicated_dressings — fix from:**
```python
"contraindicated_dressings": ["dry_gauze"]
```
**to:**
```python
"contraindicated_dressings": ["hydrocolloid", "film"]
```
*(Grounded in EWMA_VLU_MOISTURE: "Avoid hydrocolloids and films")*

**Primary Dressing — change from:**
> "Per EWMA VLU wound bed preparation guidelines, debridement of slough is the first priority."

**Change to:**
> "Alginate sheet for heavy exudate absorption (change every 2–5 days) as the initial primary dressing. Hydrofibre is equally appropriate: manages heavy exuding wounds and reduces periwound maceration (change every 2–5 days)."

**Rationale T — change from:**
> "EWMA recommends mechanical or autolytic debridement first. Slough must be removed before epithelialisation can proceed."

**Change to:**
> "The majority of uncomplicated venous ulcers do not require debridement. Compression therapy combined with appropriate dressings is the primary approach. Debridement may be considered for complex ulcers of long duration with a chronic fibrinous base, but autolytic debridement under compression is not considered effective per EWMA VLU guidelines."

**Rationale I — change from:**
> "Not infected — no antimicrobial is required. Monitor for biofilm development."

**Change to:**
> "Not infected — no antimicrobial is required. Monitor for infection indicators: increased intensity or change in character of pain, discoloured or friable granulation tissue, odour, wound breakdown, or delayed healing."

**Contraindicated Dressings — change from:**
> "- Dry gauze: desiccates the wound bed and causes trauma on removal.
> - Compression bandaging without prior blood flow assessment: compression must not be used if adequate blood flow has not been confirmed — it has been confirmed safe in your case."

**Change to:**
> "Per EWMA VLU moisture balance guidelines, the following dressings should be avoided:
> - Hydrocolloid dressings: should be avoided for venous leg ulcers.
> - Film dressings: should be avoided for venous leg ulcers."

**Dressing Change Frequency — change from:**
> "Compression bandaging: per protocol (usually 1–2 times per week)."

**Change to:**
> "Compression bandaging: per clinical protocol — change if soiled by excessive exudate or if limb circumference has changed significantly."

**Application Tips — change from:**
> "- Apply a skin barrier cream or film to the white, soggy skin around the wound before applying the dressing."

**Change to:**
> "- Apply paraffin-based products or zinc paste to the white, soggy skin around the wound before applying the dressing, to protect against maceration and irritant dermatitis."

---

### E2 — `cat_e_dfu_infected_ewma`

**user_input — fix last line from:**
> `"My blood sugar has been poorly controlled. No offloading device in use."`

**Change to:**
> `"My blood sugar has been poorly controlled. I am not using any special shoe or boot for the wound."`

**reference_contexts — add:**
```python
"reference_contexts": [
    ctx(EWMA_DFU_INFECTION), ctx(EWMA_DFU_TISSUE), ctx(EWMA_DFU_MOISTURE), ctx(EWMA_DFU_EDGE),
    ctx(AJGP_DIABFOOT), ctx(WCM_SILVER), ctx(GP_TYPE3),
    ctx(SFP_HYDROCOLLOID), ctx(EWMA_TIME_PRACTICE),
],
```

**Rationale E — change from:**
> "E (Edge): Non-advancing at 5 weeks — per EWMA DFU pathway, this triggers specialist review."

**Change to:**
> "E (Edge): Non-advancing — per EWMA DFU edge advancement pathway, failure to progress indicates the need to reassess extrinsic factors (repeated undetected trauma due to neuropathy, ischaemia, and poor metabolic control) and intrinsic factors. Specialist review is required."

**Application Tips — change from:**
> "- OFFLOADING IS MANDATORY — a total contact cast or removable cast walker is essential. Dressing alone will not heal a pressure wound on the bottom of the foot."

**Change to:**
> "- OFFLOADING IS MANDATORY — some form of cast, adapted footwear, or padding must be applied to redistribute plantar pressures evenly. Diabetic foot ulcers will not heal without pressure offloading and diabetic control."

---

## No Other Changes Required for Category E

Both patient notes fields are correctly written in 1st-person patient-facing language (with the minor user_input fix for E2 above).
The `contraindicated_dressings` field for E2 is correct once `SFP_HYDROCOLLOID` is added to contexts.
The `contraindicated_dressings` field for E1 requires the substitution from `dry_gauze` to `["hydrocolloid", "film"]` as explicitly grounded in EWMA_VLU_MOISTURE.
All core clinical recommendations across both cases are clinically accurate and appropriate once the above fixes are applied.
