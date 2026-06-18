# Category D — Gold Standard Audit Report
**Based on direct comparison of `reference` text vs actual chunk `text` + `ai_summary` content**
**All 8 source _kept.json files read verbatim. Every finding is chunk-traceable.**

---

## Legend
- ✅ Correct — grounded in cited chunk
- ⚠️ Hallucination / Ungrounded — stated in reference but NOT in any cited chunk
- 🔧 Fix needed — wrong chunk cited, chunk missing, or wording mismatch
- 📝 POV / style issue on patient notes
- ➕ Chunk should be added to reference_contexts
- ➖ Chunk should be removed from reference_contexts
- 🛡️ contraindicated_dressings field check

---

## CASE D1 — `cat_d_notes_infection_override`
**Current reference_contexts:** `GP_TYPE4`, `WCM_SILVER`, `SFP_IODINE`

### Patient Notes POV Check
> "The wound has been getting more painful over the last 3 days. The skin around it is redder and warmer than before. The fluid coming out today looks cloudy and has a bad smell."

✅ Fully 1st-person patient-facing language. No clinical jargon. Realistic patient description. **No change needed.**

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Silver: bactericidal, locally acting | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Silver: "apply directly to wound bed" | WCM_SILVER TEXT ✓ ("Place the dressing with the side with silver facing the wound bed") | ✅ |
| Alginate as secondary for moderate exudate | WCM_ALGINATE TEXT — **WCM_ALGINATE is NOT in reference_contexts** | ⚠️ |
| Foam as secondary for moderate exudate | WCM_FOAM TEXT — **WCM_FOAM is NOT in reference_contexts** | ⚠️ |
| "Cover with secondary alginate or foam" (Application Tips) | WCM_ALGINATE / WCM_FOAM — **neither chunk is in contexts** | ⚠️ |
| Iodine contraindicated if thyroid disorder | SFP_IODINE TEXT ✓ | ✅ |
| Antibiotic required, C&S swab | GP_TYPE4 TEXT ✓ ("Yes based C&S report of infected tissue") | ✅ |
| "Wound Type 4 — wet, infected, <25% NV" mapping | GP_TYPE4 TEXT ✓ | ✅ |
| "Reassess at 48 hours" (Dressing Change section) | NOT in GP_TYPE4, WCM_SILVER, or SFP_IODINE | ⚠️ |
| "Seek same-day clinical review" (Referral section) | NOT in any cited chunk — no GP chunk states this urgency | ⚠️ |
| "spreading redness / fever / feel unwell → refer" | NOT in GP_TYPE4; GP_REFERRAL lists sepsis/cellulitis as referral criteria — **GP_REFERRAL not in contexts** | ⚠️ |

### Key Issue — Secondary Dressings Ungrounded
The reference recommends alginate and foam as secondary dressings but neither `WCM_ALGINATE` nor `WCM_FOAM` are in `reference_contexts`. GP_TYPE4 lists "1.Alginate 2.Foam 3.Silver 4.Hydrofiber 5.Polymeric membrane 6.Iodine" — so the dressing names are grounded in GP_TYPE4, but the properties and change frequency stated in the reference come from WCM chunks that are absent.

### Key Issue — Urgency Instructions Ungrounded
"Reassess at 48 hours", "seek same-day clinical review" — these are clinically sound but **not stated in any of the 3 cited chunks**. They are practitioner-level reasoning, not KB-sourced statements.

### Key Issue — Referral Criteria Ungrounded
The Referral/Escalation section mentions "spreading redness, fever, feel unwell" as escalation triggers. GP_REFERRAL lists sepsis and cellulitis as criteria — but `GP_REFERRAL` is not in `reference_contexts`.

### 🛡️ contraindicated_dressings Field
```python
"contraindicated_dressings": []
```
The reference mentions iodine as conditionally contraindicated (thyroid disorder). Since this is a conditional contraindication (patient-dependent), leaving `[]` is acceptable — the condition is not always true. **No change needed.**

However, the safety pass checker should be aware: if patient notes mention thyroid disorder, `iodine` should be added. For this case the notes do NOT mention thyroid disorder, so `[]` is correct.

### reference_contexts Fix
**Add:** `WCM_ALGINATE`, `WCM_FOAM`, `GP_REFERRAL`
**Updated set:** `GP_TYPE4`, `WCM_SILVER`, `SFP_IODINE`, `WCM_ALGINATE`, `WCM_FOAM`, `GP_REFERRAL`

### Reference Text Fixes Required
1. Remove "Reassess at 48 hours" from Dressing Change Frequency — not in KB.
2. Remove "same-day clinical review" specific urgency wording — rephrase to what GP_REFERRAL states (systemic complications → refer).
3. Remove "fever or feel unwell" escalation trigger — rephrase to KB-grounded criteria: sepsis, severe cellulitis (GP_REFERRAL).

---

## CASE D2 — `cat_d_notes_diabetic_nonhealing`
**Current reference_contexts:** `AJGP_DIABFOOT`, `WCM_SILVER`, `SFP_HYDROCOLLOID`, `EWMA_DFU_EDGE`, `EWMA_DFU_TISSUE`

### Patient Notes POV Check
> "I have diabetes and I cannot feel much sensation in my feet. The wound is on the bottom of my foot and it has not improved at all in 6 weeks even though I have been changing the dressings as told. I have not been using any special shoe or boot for it."

✅ Fully 1st-person patient-facing language. No clinical jargon. Realistic description. **No change needed.**

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Silver: antimicrobial primary dressing | WCM_SILVER TEXT ✓ | ✅ |
| Silver: bactericidal, change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| DFU: primary antimicrobial dressing + secondary by exudate level | AJGP_DIABFOOT TEXT ✓ | ✅ |
| "Silicone foams on feet without borders, anchored with tape/bandage" | AJGP_DIABFOOT TEXT ✓ (exact wording in source) | ✅ |
| "Check pedal pulses and sensation; poor perfusion → refer to diabetic foot clinic or vascular surgeon" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| Hydrocolloid "not recommended for diabetic foot ulcers" | SFP_HYDROCOLLOID TEXT ✓ ("Not recommended for... diabetic foot ulcers") | ✅ |
| EWMA DFU: offloading — "redistribute plantar pressures evenly by applying some form of cast, adapted footwear or padding" | EWMA_DFU_EDGE TEXT ✓ | ✅ |
| EWMA DFU: non-healing edge → extrinsic/intrinsic factors including "repeated trauma not sensed due to neuropathy" | EWMA_DFU_EDGE TEXT ✓ | ✅ |
| EWMA DFU: "pressure control: offloading and weight redistribution" — prerequisite for wound care success | EWMA_DFU_TISSUE TEXT ✓ ("Unless these elements are addressed, wound care is more likely to fail") | ✅ |
| Debridement: sharp debridement is gold standard for DFU | EWMA_DFU_TISSUE TEXT ✓ | ✅ |
| "Non-healing at 4 or more weeks indicates edge advancement failure" | EWMA_DFU_EDGE TEXT — **EWMA_DFU_EDGE does NOT state a 4-week threshold explicitly** | ⚠️ |
| "Occult infection high risk in diabetic" | EWMA_DFU_INFECTION TEXT ✓ ("signs of inflammation and infection are absent or reduced in many diabetic patients") — **but EWMA_DFU_INFECTION is NOT in reference_contexts** | ⚠️ |
| "Silver as precautionary antimicrobial given diabetic status" | WCM_SILVER ✓ (properties), AJGP_DIABFOOT ✓ (antimicrobial primary) | ✅ |
| "Non-bordered silicone foam… WITHOUT adhesive borders" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| "Total contact cast or removable cast walker" (Application Tips) | EWMA_DFU_EDGE TEXT: "some form of cast, adapted footwear or padding" ✓ — "total contact cast or removable cast walker" are specific names | 🔧 Minor — "total contact cast" is a named device not verbatim in EWMA_DFU_EDGE which says "some form of cast" |
| "Bordered adhesive foam on foot: CONTRAINDICATED" | AJGP_DIABFOOT TEXT ✓ ("silicone foams on feet, if applied, should be without borders") | ✅ |
| "Blood sugar control, blood flow assessment" — referral content | EWMA_DFU_TISSUE TEXT ✓ ("restoration or maintenance of pulsatile blood flow", "metabolic control") | ✅ |
| "Dressing alone will not heal a pressure wound" | EWMA_DFU_TISSUE TEXT ✓ ("Unless these elements are addressed, wound care is more likely to fail") — paraphrase is acceptable | ✅ |

### Key Issue — "4 or more weeks" Threshold
`EWMA_DFU_EDGE` TEXT discusses edge advancement failure and intrinsic/extrinsic factors, but does **not state a specific "4-week" threshold** as a trigger. The reference states: *"non-healing at 4 or more weeks indicates edge advancement failure."* This specific threshold is not in any cited chunk — it is added reasoning.

**Fix:** Rephrase to: *"Per EWMA DFU edge advancement pathway, failure to progress indicates reassessment of all TIME components and extrinsic/intrinsic factors."* Remove the specific "4 or more weeks" number.

### Key Issue — "Occult infection" Ungrounded
The reference states diabetic wounds are "high risk for occult infection." This is grounded in `EWMA_DFU_INFECTION` TEXT which states: *"signs of inflammation and infection are absent or reduced in many diabetic patients"* — but `EWMA_DFU_INFECTION` is **not in reference_contexts**.

**Fix:** Add `EWMA_DFU_INFECTION` to reference_contexts.

### Key Issue — "Total contact cast or removable cast walker" Specificity
`EWMA_DFU_EDGE` says "some form of cast, adapted footwear or padding." The specific named devices ("total contact cast", "removable cast walker") are not verbatim in the KB. These are well-established clinical terms but they are not in the chunk. The Application Tips should be rephrased to match what the KB actually says.

**Fix:** Rephrase Application Tips to: *"Offloading is essential — some form of cast, adapted footwear, or padding must be used to redistribute plantar pressures. Dressing alone will not heal this wound without offloading."*

### 🛡️ contraindicated_dressings Field
```python
"contraindicated_dressings": ["bordered_foam", "hydrocolloid"]
```
Both are grounded:
- `bordered_foam`: AJGP_DIABFOOT ✓ ("without borders and anchored with tape or bandages")
- `hydrocolloid`: SFP_HYDROCOLLOID ✓ ("Not recommended for... diabetic foot ulcers")

✅ **Correct. No change needed.**

### reference_contexts Fix
**Add:** `EWMA_DFU_INFECTION` (grounding for "occult infection / absent infection signs in diabetic patients")
**Updated set:** `AJGP_DIABFOOT`, `WCM_SILVER`, `SFP_HYDROCOLLOID`, `EWMA_DFU_EDGE`, `EWMA_DFU_TISSUE`, `EWMA_DFU_INFECTION`

---

## CASE D3 — `cat_d_notes_malodour_clean`
**Current reference_contexts:** `WCM_CHARCOAL`, `WCM_ALGINATE`, `GP_TYPE2`

### Patient Notes POV Check
> "The wound looks clean and is healing, but every time the dressing is changed there is a bad smell. There is no pus, just a lot of fluid."

✅ Fully 1st-person patient-facing language. Realistic description. **No change needed.**

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Alginate for high exudate management | WCM_ALGINATE TEXT ✓ / GP_TYPE2 TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Charcoal: "odour absorbent, reduces odour" | WCM_CHARCOAL TEXT ✓ | ✅ |
| Charcoal: "needs secondary dressing" | WCM_CHARCOAL TEXT ✓ | ✅ |
| Charcoal: change every 2 days | WCM_CHARCOAL TEXT ✓ | ✅ |
| Silver as antimicrobial precaution for suspected colonisation | WCM_SILVER TEXT ✓ (purpose: reduces bacterial bioburden) — **WCM_SILVER is NOT in reference_contexts** | ⚠️ |
| Foam as outer secondary | WCM_FOAM TEXT ✓ ("secondary dressing or cavity fillers", change 2–3 days) — **WCM_FOAM is NOT in reference_contexts** | ⚠️ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ — **WCM_FOAM not in contexts** | ⚠️ |
| "Apply alginate to wound bed; cover with foam, then charcoal as outermost layer" (Application Tips) | WCM_FOAM ✓ (secondary), WCM_CHARCOAL ✓ (needs secondary) — layering order not explicitly stated, but derivable | 🔧 WCM_FOAM not in contexts |
| "Keep charcoal intact for full odour-absorbing effect" | NOT in WCM_CHARCOAL TEXT (charcoal chunk has NO application instructions beyond frequency and needs secondary) | ⚠️ |
| "bacterial colonisation rather than a true infection" | NOT stated in any cited chunk — clinical reasoning added by author | ⚠️ |
| "Change charcoal every 2 days regardless of other dressing schedules" | WCM_CHARCOAL TEXT ✓ ("Frequency of dressing change: 2 days") — derivable | ✅ |
| Wound Type 2 mapping (clean, wet, <25% NV, not infected) | GP_TYPE2 TEXT ✓ | ✅ |
| Antibiotic: not currently indicated | GP_TYPE2 TEXT ✓ ("May or may not") — reference says "not currently indicated" which is an acceptable interpretation for no frank signs | ✅ |
| Referral: not required | GP_TYPE2 TEXT ✓ (no referral flag for Type 2) | ✅ |

### Key Issue — Silver Referenced but WCM_SILVER Absent
The reference proposes silver as an optional antimicrobial precaution for bacterial colonisation. `WCM_SILVER` TEXT grounds the antimicrobial rationale but is **not in reference_contexts**.

**Fix:** Add `WCM_SILVER` to reference_contexts.

### Key Issue — Foam Referenced but WCM_FOAM Absent
Foam is the outer secondary dressing with a 2–3 day change frequency. `WCM_FOAM` is **not in reference_contexts**.

**Fix:** Add `WCM_FOAM` to reference_contexts.

### Key Issue — "Keep charcoal intact for full odour-absorbing effect"
WCM_CHARCOAL has **no application instruction** beyond "Needs secondary dressing. Frequency: 2 days." The phrase "keep intact for full odour-absorbing effect" is not in the KB.

**Fix:** Remove this phrase from Application Tips. Replace with: *"Charcoal dressing requires a secondary dressing; change every 2 days."*

### Key Issue — "Bacterial colonisation rather than true infection"
This clinical distinction is reasonable but is **not stated in any of the 3 cited chunks**. WCM_CHARCOAL, WCM_ALGINATE, and GP_TYPE2 make no mention of bacterial colonisation.

**Fix:** Remove the colonisation distinction from Clinical Notes. Replace with a grounded statement: *"Monitor closely at each change for progression to active infection signs (increasing pain, redness, warmth, or purulent discharge)."*

### 🛡️ contraindicated_dressings Field
```python
"contraindicated_dressings": []
```
No explicit contraindications exist for Wound Type 2 in the cited chunks. Silver and charcoal are not listed for Type 2 by GP_TYPE2, but charcoal is specifically **recommended** here for malodour (not contraindicated). ✅ **Correct. No change needed.**

### reference_contexts Fix
**Add:** `WCM_SILVER`, `WCM_FOAM`
**Updated set:** `WCM_CHARCOAL`, `WCM_ALGINATE`, `GP_TYPE2`, `WCM_SILVER`, `WCM_FOAM`

---

## CASE D4 — `cat_d_notes_npwt_adjunct`
**Current reference_contexts:** `WCM_NPWT`, `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`

### Patient Notes POV Check
> "I had surgery yesterday to clean out my wound. The doctor said the wound is clean now and is thinking about putting on a vacuum dressing machine. Is that okay now that the dead tissue has been removed?"

✅ Fully 1st-person patient-facing language. Realistic patient question. **No change needed.**

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| NPWT: "subatmospheric pressure via open-cell wound dressing" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT: "adjunct to management of chronic, acute and difficult wounds; not a panacea" | WCM_NPWT TEXT ✓ ("NPWT is only an adjunct… it is not a panacea") | ✅ |
| NPWT: "prepares wound bed for greater chance of closure" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT: "does not replace surgical procedures" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT Contraindications: "necrotic wound bed or eschar" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT Contraindications: "clotting disorders" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT Contraindications: "neoplastic tissue in the wound area" | WCM_NPWT TEXT ✓ | ✅ |
| NPWT Contraindications: "untreated infection" | WCM_NPWT TEXT ✓ ("Untreated infection") | ✅ |
| "NPWT dressing change: 3–5 days interval" | WCM_NPWT TEXT ✓ | ✅ |
| "NPWT foam inserts typically changed every 2–3 days when initiated by specialist" | WCM_NPWT TEXT — source says "3–5 days interval or depending on amount or characteristic of fluid withdrawn" | ⚠️ |
| Alginate interim: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre interim: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ — **WCM_HYDROFIBRE is NOT in reference_contexts** | ⚠️ |
| Foam as secondary: change every 2–3 days | WCM_FOAM TEXT ✓ — **WCM_FOAM is NOT in reference_contexts** | ⚠️ |
| Wound Type 6: requires hospital referral | GP_TYPE6 TEXT ✓ / GP_REFERRAL TEXT ✓ | ✅ |
| Wound Type 6: alginate, foam, polymeric membrane, hydrofiber as dressings | GP_TYPE6 TEXT ✓ | ✅ |
| GP_REFERRAL: "vacuum (negative pressure) dressing" as referral criterion | GP_REFERRAL TEXT ✓ ("Vacuum (negative pressure) dressing" listed under extensive care criteria) | ✅ |
| "NPWT is contraindicated on necrotic wound bed — but dead tissue removed, so now may be considered" | WCM_NPWT TEXT ✓ (contraindication: "necrotic wound bed or eschar") | ✅ |

### Key Issue — NPWT Change Frequency Mismatch
Reference states "NPWT foam inserts typically changed every 2–3 days." `WCM_NPWT` TEXT states: **"3–5 days interval or depending on the amount or characteristic of fluid withdrawn."**

**Fix:** Change to: *"NPWT dressing change: every 3–5 days, or depending on the amount and characteristics of fluid withdrawn, as determined by the specialist team."*

### Key Issue — Hydrofibre Referenced but WCM_HYDROFIBRE Absent
Hydrofibre is recommended as an interim dressing alongside alginate, but `WCM_HYDROFIBRE` is **not in reference_contexts**.

**Fix:** Add `WCM_HYDROFIBRE` to reference_contexts.

### Key Issue — Foam Referenced but WCM_FOAM Absent
Foam is the outer secondary dressing with a 2–3 day change frequency, but `WCM_FOAM` is **not in reference_contexts**.

**Fix:** Add `WCM_FOAM` to reference_contexts.

### 🛡️ contraindicated_dressings Field
```python
"contraindicated_dressings": []
```
The Contraindicated Dressings section in the reference correctly identifies NPWT conditions that must be absent before initiation (necrotic tissue, neoplastic tissue, clotting disorder), but these are NPWT eligibility conditions, not wound dressing contraindications in the traditional sense.

**Issue:** The safety pass checker needs to know NPWT is contraindicated if these conditions are present. However since the patient's notes confirm debridement has occurred, there is no active wound dressing contraindication at this point.

**Recommendation:** Add `"npwt_if_necrosis_present"` or keep `[]` but add a note. For consistency with the safety pass checker format, keep `[]` for dressing materials but note that NPWT eligibility must be confirmed by specialist. ✅ **Keep `[]` — no standard dressing contraindication applies here.**

### reference_contexts Fix
**Add:** `WCM_HYDROFIBRE`, `WCM_FOAM`
**Updated set:** `WCM_NPWT`, `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

---

## Summary Table — All Category D Issues

| Case | Hallucination / Ungrounded | Missing Chunk | Reference Text Fix | POV Issue |
|---|---|---|---|---|
| D1 `infection_override` | ⚠️ "Reassess 48hr", "same-day review", escalation criteria | ➕ `WCM_ALGINATE`, `WCM_FOAM`, `GP_REFERRAL` | Rephrase urgency to KB-sourced wording | None |
| D2 `diabetic_nonhealing` | ⚠️ "4 or more weeks" threshold; "total contact cast/removable cast walker"; "occult infection" ungrounded | ➕ `EWMA_DFU_INFECTION` | Remove "4-week" threshold; rephrase offloading device names; add EWMA_DFU_INFECTION grounding | None |
| D3 `malodour_clean` | ⚠️ "bacterial colonisation" distinction; "keep charcoal intact"; silver/foam missing from contexts | ➕ `WCM_SILVER`, `WCM_FOAM` | Remove colonisation claim; remove "keep intact"; rephrase charcoal tip | None |
| D4 `npwt_adjunct` | ⚠️ NPWT change "2–3 days" contradicts KB "3–5 days" | ➕ `WCM_HYDROFIBRE`, `WCM_FOAM` | Fix NPWT change frequency to match WCM_NPWT exactly | None |

**POV Check:** All 4 Category D cases have appropriate 1st-person patient-facing notes. No clinical jargon issues. No changes needed to `user_input` or `notes` fields.

---

## Required Fixes — Category D (Precise)

### D1 — `cat_d_notes_infection_override`

**reference_contexts — add:**
```python
"reference_contexts": [ctx(GP_TYPE4), ctx(WCM_SILVER), ctx(SFP_IODINE),
                       ctx(WCM_ALGINATE), ctx(WCM_FOAM), ctx(GP_REFERRAL)],
```

**Secondary Dressing — keep, now grounded via WCM_ALGINATE and WCM_FOAM.**

**Dressing Change Frequency — change from:**
> "Reassess at 48 hours — if no improvement, seek urgent clinical review."

**Change to:**
> "Silver: every 2–3 days. Alginate: every 2–5 days. Foam: every 2–3 days."

**Referral/Escalation — change from:**
> "Seek same-day clinical review given the multiple signs of active infection. Referral to a specialist may be required if infection is spreading (increasing redness spreading outward, or if you develop fever or feel unwell)."

**Change to:**
> "Seek an urgent clinical review given the multiple signs of active infection. Referral to hospital is required if systemic wound complications develop, including sepsis or severe cellulitis."

**Application Tips — change from:**
> "Seek a clinical review the same day to obtain a C&S swab and antibiotic prescription."

**Change to:**
> "Seek an urgent clinical review to obtain a C&S swab and antibiotic prescription."

---

### D2 — `cat_d_notes_diabetic_nonhealing`

**reference_contexts — add:**
```python
"reference_contexts": [ctx(AJGP_DIABFOOT), ctx(WCM_SILVER), ctx(SFP_HYDROCOLLOID),
                       ctx(EWMA_DFU_EDGE), ctx(EWMA_DFU_TISSUE), ctx(EWMA_DFU_INFECTION)],
```

**Primary Dressing — change from:**
> "Per EWMA DFU TIME framework, non-healing at 4 or more weeks indicates edge advancement failure"

**Change to:**
> "Per the EWMA DFU TIME framework, failure of the wound edge to advance indicates reassessment of all TIME components and extrinsic/intrinsic factors is required."

**Rationale E — change from:**
> "Non-advancing at 6 weeks — per EWMA DFU pathway, this triggers reassessment and escalation to a specialist."

**Change to:**
> "Non-advancing — per EWMA DFU edge advancement pathway, failure to progress indicates reassessment of all TIME components. Extrinsic factors (repeated undetected trauma due to neuropathy, ischaemia, poor metabolic control) must be addressed."

**Rationale I — add grounding for occult infection, now supported by EWMA_DFU_INFECTION:**
> "Not currently infected — however, signs of inflammation and infection may be absent or reduced in diabetic patients due to sensory neuropathy and/or poor blood supply. Silver is used as a precautionary antimicrobial."

**Application Tips — change from:**
> "OFFLOADING IS THE MOST IMPORTANT STEP — a special boot or shoe insert (total contact cast or removable cast walker) is essential."

**Change to:**
> "OFFLOADING IS THE MOST IMPORTANT STEP — some form of cast, adapted footwear, or padding must be applied to redistribute plantar pressures evenly. Dressing alone will not heal this wound without offloading."

---

### D3 — `cat_d_notes_malodour_clean`

**reference_contexts — add:**
```python
"reference_contexts": [ctx(WCM_CHARCOAL), ctx(WCM_ALGINATE), ctx(GP_TYPE2),
                       ctx(WCM_SILVER), ctx(WCM_FOAM)],
```

**Application Tips — change from:**
> "Keep the charcoal intact for full odour-absorbing effect."

**Change to:**
> "Charcoal dressing requires a secondary dressing; change every 2 days."

**Clinical Notes — change from:**
> "A bad smell without pus may indicate bacterial colonisation rather than a true infection. The charcoal dressing addresses the odour directly; silver addresses the colonisation concern. Monitor closely at each change for progression to infection."

**Change to:**
> "The charcoal dressing addresses wound odour directly. Silver addresses bacterial burden if clinically indicated. Monitor closely at each change for signs of active infection (increasing pain, redness, warmth, or purulent discharge)."

---

### D4 — `cat_d_notes_npwt_adjunct`

**reference_contexts — add:**
```python
"reference_contexts": [ctx(WCM_NPWT), ctx(GP_TYPE6), ctx(GP_REFERRAL),
                       ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM)],
```

**Dressing Change Frequency — change from:**
> "NPWT foam inserts are typically changed every 2–3 days when initiated by the specialist."

**Change to:**
> "NPWT dressing change: every 3–5 days, or depending on the amount and characteristics of fluid withdrawn, as determined by the specialist team."

---

## No Other Changes Required for Category D

All 4 patient notes fields are correctly written in 1st-person patient-facing language.
All `contraindicated_dressings` fields are correctly set.
The core clinical recommendations across all 4 cases are clinically accurate and appropriate.
Only the specific issues above require correction.
