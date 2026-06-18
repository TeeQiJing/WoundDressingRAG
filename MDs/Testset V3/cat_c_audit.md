# Category C — Gold Standard Audit Report
**Based on direct comparison of `reference` text vs actual chunk `text`**
**All 6 Cat C cases checked against 8 source _kept.json files**

---

## Legend
- ✅ Correct — grounded in cited chunk
- ⚠️ Hallucination / Ungrounded — stated in reference but NOT in any cited chunk
- 🔧 Fix needed
- 📝 POV issue
- ➕ Chunk to add to reference_contexts
- ➖ Chunk to remove from reference_contexts

---

## 1. `cat_c_dressing_saturation`
**reference_contexts:** `GP_TYPE2`, `WCM_FOAM`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`

### POV Check — user_input notes
> "The foam dressing was put on 3 days ago. It is wet and soaked through, the edges are coming unstuck, and fluid is leaking out."
✅ Excellent 1st-person patient language. Clearly describes all three change criteria.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Change the dressing IMMEDIATELY — soiled, edges lifting, strikethrough" | WCM_FOAM TEXT: "dressing change: 2 to 3 days" but NO explicit rule about soiled/strikethrough/lifting triggers | ⚠️ The three trigger criteria not in any chunk |
| GP_TYPE2 dressing list: Foam, Alginate, Hydrofiber, Polymeric membrane | GP_TYPE2 TEXT ✓ | ✅ |
| Alginate: "forms a gel on contact with wound fluid" | WCM_ALGINATE TEXT: "Absorb wound exudates and maintain moisture" — "forms a gel" NOT stated verbatim | ⚠️ Minor paraphrase — gel formation is from ISTAP_PRODUCTS for alginate but NOT in WCM_ALGINATE |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: requires secondary dressing | WCM_ALGINATE TEXT ✓ ("Need secondary dressing") | ✅ |
| Hydrofibre: "manages heavy exuding wounds, reduces maceration" | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: "highly absorbent, bacterial barrier" | WCM_FOAM TEXT ✓ ("Highly absorbent... Bacterial and waterproof") | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| Antibiotic: No | GP_TYPE2 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE2 TEXT ✓ (no referral flag) | ✅ |
| "saturated dressing is a contamination and infection risk" | Not in any chunk | ⚠️ Ungrounded — clinical reasoning, not sourced |
| "Leaving saturated dressing increases risk of wound infection and maceration" | Not in any chunk | ⚠️ Ungrounded — clinical reasoning, not sourced |
| "find and treat underlying cause of high exudate" | GP_TYPE2 TEXT ✓ ("Find underlying cause / Treat underlying cause") | ✅ |

### Issues Found

**⚠️ Three ungrounded clinical statements:**
1. The three-criteria rule (soiled / edges lifting / strikethrough) — no chunk defines when to change. WCM_FOAM only gives frequency. GP guidelines do not state explicit criteria.
2. "forms a gel on contact with wound fluid" for alginate — WCM_ALGINATE says "absorb wound exudates" but does not say "gel formation." The gel description is accurate but ungrounded in WCM_ALGINATE.
3. "contamination and infection risk" / "maceration of surrounding skin" — not in any cited chunk.

**Assessment:** These are all valid clinical knowledge, but they are elaborations beyond the source chunks. For RAGAS Faithfulness scoring, these will not be penalised heavily as they don't contradict the source — they are additions. However for a strict gold standard, note them.

**Recommendation:** Accept (1) and (3) as clinical elaboration — they strengthen the patient-facing explanation. Fix (2): change "forms a gel on contact with wound fluid" to "absorbs wound exudates" to match WCM_ALGINATE verbatim.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ **Correct.** No source explicitly contraindications any dressing for Wound Type 2 with no infection. Silver and charcoal are "not listed" (same logic as Cat A Type 2 audit). Empty is correct.

### reference_contexts Assessment
All 4 chunks are correct and necessary. No additions or removals needed.

---

## 2. `cat_c_malodour_type8`
**reference_contexts:** `GP_TYPE8`, `WCM_CHARCOAL`, `WCM_SILVER`, `GP_REFERRAL`

### POV Check — user_input notes
> "The wound has a very bad smell. My family cannot be in the same room when the dressing is changed."
✅ Excellent 1st-person patient language. Emotionally resonant and clinically informative.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP_TYPE8 dressing list: Alginate, Silver, Hydrofiber, Foam, Polymeric membrane, Charcoal, Iodine | GP_TYPE8 TEXT ✓ | ✅ |
| "Referral required — Wound Type 8" | GP_TYPE8 + GP_REFERRAL TEXT ✓ | ✅ |
| "Surgical debridement strongly recommended" | GP_TYPE8 TEXT ✓ | ✅ |
| "May need repeated debridement" | GP_TYPE8 TEXT ✓ | ✅ |
| Charcoal: "absorbs wound odour" | WCM_CHARCOAL TEXT ✓ ("Odour absorbent, Reduces odour") | ✅ |
| Charcoal: change every 2 days | WCM_CHARCOAL TEXT ✓ ("Frequency of dressing change: 2 days") | ✅ |
| Charcoal: "needs secondary dressing" | WCM_CHARCOAL TEXT ✓ ("Needs secondary dressing") | ✅ |
| **"must NOT be cut (cutting disrupts the active charcoal layer)"** | WCM_CHARCOAL TEXT: says NOTHING about cutting. Entire charcoal text: "Purpose: Odour absorbent / Advantages: Reduces odour / Disadvantages: Needs secondary dressing / Frequency: 2 days" | ⚠️ **HALLUCINATION** |
| Silver: bactericidal, no known resistance | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ — **but WCM_ALGINATE is NOT in reference_contexts** |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ — **but WCM_HYDROFIBRE is NOT in reference_contexts** |
| Foam: "additional absorbency as outer layer" | WCM_FOAM TEXT ✓ ("secondary dressing") | ✅ — **but WCM_FOAM is NOT in reference_contexts** |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ — **but WCM_FOAM is NOT in reference_contexts** |
| Antibiotic: required — C&S | GP_TYPE8 TEXT ✓ | ✅ |
| "Apply charcoal as outermost layer — do not cut it" | WCM_CHARCOAL TEXT: only says "needs secondary dressing" — no instruction on layering or cutting | ⚠️ **HALLUCINATION** (both the layering and the cutting instruction) |

### Issues Found

**🔧 HALLUCINATION — "Do not cut charcoal"**
This is the same hallucination flagged in Cat A Type 8 audit. `WCM_CHARCOAL` TEXT contains **nothing** about cutting the dressing. The entire charcoal chunk is 4 lines. "Do not cut" appears in the reference twice (once in Secondary Dressing, once in Application Tips) but is not in any source chunk. This must be **removed** from both locations.

**🔧 Missing chunks for alginate, hydrofibre, foam**
All three are named with change frequencies in the reference but none of their WCM chunks are in reference_contexts.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ **Correct.** The reference correctly notes "Iodine should be avoided if a thyroid disorder is present" but since no thyroid disorder is stated in the notes, iodine is not contraindicated here. Empty is correct.

### Reference Text Fixes

**Remove from Secondary Dressing:**
> ~~"must NOT be cut (cutting disrupts the active charcoal layer)"~~

**Remove from Application Tips:**
> ~~"- Add charcoal dressing as the outermost layer — do not cut it."~~
Replace with: `"- Add charcoal dressing as the outermost layer."`
> Keep: `"- Change charcoal every 2 days regardless of other dressing schedules."`

**Fix alginate description:**
"forms a gel on contact" → not in WCM_ALGINATE; remove or replace with "absorbs exudate"

### reference_contexts Fix
**Add:** `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`
**Updated set:** `GP_TYPE8`, `WCM_CHARCOAL`, `WCM_SILVER`, `GP_REFERRAL`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

---

## 3. `cat_c_heavy_exudate_maceration`
**reference_contexts:** `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`, `GP_TYPE2`

### POV Check — user_input notes
> "The foam dressing fills up with fluid within one day. The skin right around the wound has gone white and soft."
✅ Excellent 1st-person patient language. "Gone white and soft" perfectly describes maceration without using the medical term.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Alginate: "highest absorption capacity; forms a gel on contact with wound fluid" | WCM_ALGINATE TEXT: "Absorb wound exudates and maintain moisture" — no mention of "highest absorption" or "gel formation" | ⚠️ "Highest absorption" and "forms a gel" not in WCM_ALGINATE |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: requires secondary dressing | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: "manages heavy exuding wounds, specifically reduces maceration risk" | WCM_HYDROFIBRE TEXT ✓ ("Manage heavy exuding wounds... Reduce risk of maceration") | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: "highly absorbent, bacterial barrier" | WCM_FOAM TEXT ✓ | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| GP_TYPE2: foam alone is insufficient — upgrade to alginate/hydrofibre | GP_TYPE2 TEXT: lists Foam, Alginate, Hydrofiber as options — does NOT say foam is insufficient alone | ⚠️ Clinical reasoning, not directly stated in GP_TYPE2 |
| "Apply skin barrier wipe to white, soft skin around wound" | Not in any cited chunk | ⚠️ Ungrounded — reasonable clinical tip |
| "Do not apply alginate directly on macerated skin" | Not in any cited chunk | ⚠️ Ungrounded |
| "White, soft skin around wound (maceration) can cause skin breakdown" | Not in any cited chunk | ⚠️ Ungrounded |
| "Find and treat underlying cause of high exudate" | GP_TYPE2 TEXT ✓ ("Find underlying cause / Treat underlying cause") | ✅ |
| Antibiotic: No | GP_TYPE2 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE2 TEXT ✓ | ✅ |

### Issues Found

**⚠️ "Highest absorption capacity" and "forms a gel"**
WCM_ALGINATE TEXT does not say alginate has the "highest absorption capacity" — it says "Absorb wound exudates and maintain moisture." "Forms a gel on contact" is also not stated. Same issue as case 1 and 2.

**Fix:** Change "highest absorption capacity; forms a gel on contact with wound fluid" to "absorbs wound exudates and maintains moisture."

**⚠️ Several Application Tips are ungrounded**
The three Application Tips (skin barrier wipe, do not apply on macerated skin, reassess maceration) are valid clinical advice but not in any of the 4 cited chunks. Accept as clinical elaboration — low risk for RAGAS scoring since they don't contradict the source. The maceration-specific advice actually comes from WCM_HYDROFIBRE advantages ("Reduce risk of maceration") which supports the concept.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ **Correct.** No source explicitly contraindications any dressing for this wound. Foam is "inadvisable" per clinical reasoning but not an explicit source contraindication. Empty is correct.

### reference_contexts Assessment
All 4 chunks are correctly cited and contributing. No additions or removals needed.

**Minor text fix only:**
Change "highest absorption capacity; forms a gel on contact with wound fluid" → "absorbs wound exudates and maintains moisture; requires a secondary dressing."

---

## 4. `cat_c_dry_infected_combo`
**reference_contexts:** `WCM_HYDROGEL`, `WCM_SILVER`, `GP_TYPE3`

### POV Check — user_input notes
> "The wound is infected but also looks dry and there is no fluid coming from it. Can one dressing do both jobs?"
✅ Excellent 1st-person patient language. The question maps exactly to the dual-purpose dressing combination answer.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP_TYPE3 dressing list: Tulle, Hydrogel, Hydrocolloid, Silver, Iodine | GP_TYPE3 TEXT ✓ | ✅ |
| Hydrogel: "rehydrates dry infected wound bed, promotes autolytic debridement" | WCM_HYDROGEL TEXT ✓ ("Rehydrate, debride and deslough... promote moist healing") | ✅ |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "must have a secondary dressing" | WCM_HYDROGEL TEXT ✓ ("Need secondary dressing") | ✅ |
| Silver: "bactericidal" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| "Apply hydrogel first, then silver on top as secondary antimicrobial layer" | WCM_HYDROGEL: "Apply on wound bed as primary" — silver on top is a logical layering but WCM_SILVER says "place silver side facing wound bed" | ⚠️ Conflict: WCM_SILVER says silver faces wound bed directly — placing hydrogel first then silver on top means silver's silver side doesn't face the wound bed |
| Antibiotic: required — C&S | GP_TYPE3 TEXT ✓ | ✅ |
| Referral: not required for Type 3 | GP_TYPE3 TEXT ✓ | ✅ |
| "Iodine must be avoided if thyroid disorder" | GP_TYPE3 TEXT: lists iodine as option; SFP_IODINE grounds the contraindication | ✅ — but SFP_IODINE is NOT in reference_contexts |
| "Debridement may be needed" | GP_TYPE3 TEXT ✓ | ✅ |

### Issues Found

**🔧 CLINICAL CONFLICT — Silver application layer order**
This is a meaningful clinical accuracy issue. The reference says:
> "Apply hydrogel first, directly to the wound bed. Place the silver dressing on top of the hydrogel."

But `WCM_SILVER` TEXT says:
> **"Place the dressing with the side with silver facing the wound bed."**

If hydrogel is applied first and silver is placed on top, the silver side faces the hydrogel, NOT the wound bed — contradicting WCM_SILVER's application instruction. The dual-purpose intention is clinically sound, but the layering description conflicts with the silver dressing's own application rule.

**Recommended Fix:** Clarify the layering to make it consistent:
> "Apply hydrogel directly to the wound bed. Then place the silver dressing over the wound with the silver side facing down toward the wound (through the hydrogel layer), per standard silver application."

Or alternatively, describe it as two separate primary options rather than a layered combination, since the chunked sources don't describe this exact combination. This is the safest fix for grounding.

**⚠️ Iodine contraindication mentioned but SFP_IODINE not in reference_contexts**
The Contraindicated Dressings section says "If a thyroid disorder is present, iodine-based dressings must be avoided" — this is grounded in SFP_IODINE but that chunk is absent.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ **Correct.** No unconditional contraindications for this wound (no thyroid disorder stated in notes). The iodine contraindication is conditional only. Empty is correct.

### reference_contexts Fix
**Add:** `SFP_IODINE` (grounds the iodine contraindication mentioned in Contraindicated Dressings)
**Updated set:** `WCM_HYDROGEL`, `WCM_SILVER`, `GP_TYPE3`, `SFP_IODINE`

---

## 5. `cat_c_time_assessment_mixed`
**reference_contexts:** `GP_ALGO`, `GP_TYPE7`, `GP_REFERRAL`, `WCM_SILVER`, `EWMA_TIME_PRACTICE`

### POV Check — user_input notes
> "Half the wound is covered in black and yellow dead tissue. There is pus coming out and it smells bad."
✅ Excellent 1st-person patient language. "Black and yellow dead tissue" = necrosis + slough. "Pus" = infection. "Smells bad" = malodour. All three TIME-relevant cues present.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP_TYPE7 dressing list: Silver, Hydrogel, Hydrocolloid, Iodine, Polymeric membrane | GP_TYPE7 TEXT ✓ | ✅ |
| GP_ALGO: >25% NV + infected + moderate = wound type classification | GP_ALGO TEXT ✓ | ✅ |
| "Referral required — Wound Type 7" | GP_TYPE7 + GP_REFERRAL TEXT ✓ | ✅ |
| "Surgical debridement strongly recommended" | GP_TYPE7 TEXT ✓ | ✅ |
| Silver: bactericidal, change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Hydrogel: "moisture donation to necrotic component" | WCM_HYDROGEL TEXT ✓ ("Rehydrate, debride") | ✅ — **but WCM_HYDROGEL is NOT in reference_contexts** |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ — **but WCM_HYDROGEL is NOT in reference_contexts** |
| Hydrocolloid: "promotes autolysis" | WCM_HYDROCOLLOID TEXT ✓ | ✅ — **but WCM_HYDROCOLLOID is NOT in reference_contexts** |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ — **but WCM_HYDROCOLLOID is NOT in reference_contexts** |
| Charcoal: "for malodour; change every 2 days; do not cut" | WCM_CHARCOAL TEXT: ✓ for change frequency; ❌ "do not cut" NOT in WCM_CHARCOAL | ⚠️ **HALLUCINATION** — same "do not cut" issue as case 2 |
| Charcoal: "requires a secondary under it" | WCM_CHARCOAL TEXT ✓ ("Needs secondary dressing") | ✅ — **but WCM_CHARCOAL is NOT in reference_contexts** |
| "per EWMA TIME Framework" attribution in Rationale | EWMA_TIME_PRACTICE TEXT: discusses TIME framework for VLU, DFU, general WBP — but does NOT provide specific dressing recommendations for wound types | ⚠️ Weak fit — EWMA_TIME_PRACTICE discusses framework conceptually, not Type 7 dressing selection |
| Antibiotic: required — C&S | GP_TYPE7 TEXT ✓ | ✅ |
| "Iodine must be avoided if thyroid disorder" | GP_TYPE7 lists iodine; SFP_IODINE grounds the contraindication | ✅ — but SFP_IODINE NOT in reference_contexts |

### Issues Found

**🔧 HALLUCINATION — "Do not cut charcoal"**
Same issue as case 2. WCM_CHARCOAL TEXT contains nothing about cutting. Remove from Application Tips.

**🔧 Missing chunks — Hydrogel, Hydrocolloid, Charcoal**
All three are named with properties and frequencies but their WCM chunks are absent from reference_contexts.

**🔧 EWMA_TIME_PRACTICE — Wrong fit**
`EWMA_TIME_PRACTICE` discusses the TIME framework as a conceptual pathway and wound bed preparation philosophy. It does NOT contain dressing recommendations for Wound Type 7, charcoal, or malodour management. Including it as a reference context for this case is a mismatch — RAGAS retrieval scoring would penalise this as a non-contributing retrieved chunk.

The "per EWMA TIME Framework" rationale note in the reference text is fine as a contextual attribution, but the chunk itself does not ground the specific dressing choices made. **Remove `EWMA_TIME_PRACTICE`** and replace with the missing dressing chunks.

**⚠️ SFP_IODINE absent despite iodine contraindication referenced**
Same issue as case 4.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ **Correct.** Iodine is conditional (thyroid disorder not stated). Alginate/hydrofibre are absent from GP_TYPE7 list but the wound is moderate moisture, making them less applicable anyway — not explicitly contraindicated. Empty is correct.

### reference_contexts Fix
**Add:** `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`, `WCM_CHARCOAL`, `SFP_IODINE`
**Remove:** `EWMA_TIME_PRACTICE` (does not ground any specific dressing claim in this case)
**Updated set:** `GP_ALGO`, `GP_TYPE7`, `GP_REFERRAL`, `WCM_SILVER`, `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`, `WCM_CHARCOAL`, `SFP_IODINE`

### Reference Text Fix
**Remove from Application Tips:**
> ~~"do not cut it"~~
Change to: `"- Add charcoal dressing as the outermost layer."`

---

## 6. `cat_c_film_vs_hydrocolloid`
**reference_contexts:** `WCM_FILM`, `WCM_HYDROCOLLOID`, `GP_TYPE1`, `SFP_FILM`

### POV Check — user_input notes
> "The wound is clean and healing. The nurse mentioned I could use either a clear see-through dressing or a thicker opaque one. What is the difference and which should I choose?"
✅ Excellent 1st-person patient language. "Clear see-through" = film; "thicker opaque one" = hydrocolloid. Perfect lay descriptions.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Film: "transparent — you can check the wound without removing the dressing" | WCM_FILM TEXT ✓ ("Transparent with measurement grid... Facilitate assessment") | ✅ |
| Film: "waterproof" | WCM_FILM TEXT ✓ ("Waterproof") | ✅ |
| Film: change every 2–5 days | WCM_FILM TEXT ✓ | ✅ |
| Film: "press edges firmly to create waterproof seal; do not overstretch" | WCM_FILM TEXT: "Apply the film over the site making sure there is no air under it. To remove the film, stretch the film and pull slowly from the edges." — "press edges firmly" is an inference; "do not overstretch" not stated | ⚠️ Minor extension |
| Film: "Both can be left in place for showering" | WCM_FILM TEXT ✓ ("Waterproof") | ✅ (waterproof = shower-safe implied) |
| Hydrocolloid: "provides moist healing environment, promotes autolysis, waterproof" | WCM_HYDROCOLLOID TEXT ✓ ("Provide moist environment... Cleans and debrides by autolysis... Waterproof") | ✅ |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| "Warm hydrocolloid between your hands before applying to improve adherence" | WCM_HYDROCOLLOID TEXT: says "Apply the adhesive side onto the wound without touching the wound bed" — NO mention of warming | ⚠️ Ungrounded — common clinical tip but not in WCM_HYDROCOLLOID |
| GP_TYPE1: "All types except silver, charcoal and special advanced dressings" | GP_TYPE1 TEXT ✓ | ✅ |
| Silver/charcoal: CONTRAINDICATED for Type 1 | GP_TYPE1 TEXT ✓ (explicitly excluded) | ✅ |
| SFP_FILM: "skin around wound must be intact for good seal; avoid in draining or infected wounds" | SFP_FILM TEXT ✓ | ✅ |
| "Film: best choice if easy wound monitoring is a priority" | WCM_FILM + GP_TYPE1 — logically derived from transparent property | ✅ (reasonable derivation) |
| "Hydrocolloid: best choice if gentle autolytic cleaning desired" | WCM_HYDROCOLLOID TEXT ✓ ("Cleans and debrides by autolysis") | ✅ |
| Antibiotic: No | GP_TYPE1 TEXT ✓ | ✅ |
| Referral: Not indicated | GP_TYPE1 TEXT ✓ | ✅ |

### Issues Found

**⚠️ Minor — "Warm hydrocolloid between hands"**
WCM_HYDROCOLLOID TEXT does NOT mention warming the dressing before application. This is a known clinical technique but entirely ungrounded in the source chunk. Remove or accept as elaboration.

**⚠️ Minor — "Press edges firmly; do not overstretch"**
WCM_FILM says "make sure there is no air under it" and gives removal instructions. The "press edges firmly" tip is a reasonable paraphrase of this; "do not overstretch" is not in any chunk (WCM_FILM says stretch to REMOVE, not apply). Minor issue — acceptable elaboration.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["silver", "charcoal"]
```
✅ **Correct** — GP_TYPE1 TEXT explicitly excludes silver and charcoal by name. Strongest source-grounded contraindication in the dataset. This is the one wound type where silver/charcoal are truly contraindicated, not just "not listed."

### reference_contexts Assessment
All 4 chunks are correctly cited and all contribute. No additions or removals needed.

**Minor text fix:**
Remove "warm between your hands for a moment before applying to improve adherence" from Application Tips — not grounded in WCM_HYDROCOLLOID.

---

## Summary Table — All Category C Issues

| Case | Hallucination | Missing Chunks | Chunk to Remove | POV Fix | `contraindicated_dressings` Fix |
|---|---|---|---|---|---|
| dressing_saturation | ⚠️ 3 minor (change criteria, gel formation, infection risk) | None | None | ✅ Clean | ✅ Keep `[]` |
| malodour_type8 | 🔧 **"Do not cut charcoal"** ×2 | ➕ WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM | None | ✅ Clean | ✅ Keep `[]` |
| heavy_exudate_maceration | ⚠️ "highest absorption"/"forms gel" | None | None | ✅ Clean | ✅ Keep `[]` |
| dry_infected_combo | 🔧 Silver layering conflicts with WCM_SILVER application instruction | ➕ SFP_IODINE | None | ✅ Clean | ✅ Keep `[]` |
| time_assessment_mixed | 🔧 **"Do not cut charcoal"** | ➕ WCM_HYDROGEL, WCM_HYDROCOLLOID, WCM_CHARCOAL, SFP_IODINE | ➖ EWMA_TIME_PRACTICE | ✅ Clean | ✅ Keep `[]` |
| film_vs_hydrocolloid | ⚠️ "warm hydrocolloid" minor | None | None | ✅ Clean | ✅ Keep `["silver","charcoal"]` |

**POV check:** All 6 Cat C user_input notes are written in excellent 1st-person patient language. No fixes needed.

---

## Required Fixes — Consolidated

### Fix 1 — "Do not cut charcoal" hallucination (cases 2 and 5)
This appears in two cases. **Remove from both.** WCM_CHARCOAL TEXT has no such instruction.

```
# cat_c_malodour_type8 — Secondary Dressing section:
REMOVE: "must NOT be cut (cutting disrupts the active charcoal layer)."
KEEP:   "Charcoal dressing — specifically indicated for wound malodour management;
         absorbs wound odour; change every 2 days; requires a secondary dressing."

# cat_c_malodour_type8 — Application Tips:
CHANGE: "- Apply the charcoal dressing as the outermost layer over the foam — do not cut it."
TO:     "- Apply the charcoal dressing as the outermost layer over the foam."

# cat_c_time_assessment_mixed — Secondary Dressing section:
REMOVE: "do not cut the charcoal layer;"
CHANGE: "Charcoal may be added as outermost layer given the malodour
         (change every 2 days; requires a secondary under it)."

# cat_c_time_assessment_mixed — Application Tips:
CHANGE: "- Add charcoal dressing as the outermost layer — do not cut it."
TO:     "- Add charcoal dressing as the outermost layer."
```

### Fix 2 — reference_contexts additions

```python
# cat_c_malodour_type8 — add:
ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM)

# cat_c_dry_infected_combo — add:
ctx(SFP_IODINE)

# cat_c_time_assessment_mixed — add and remove:
# ADD: ctx(WCM_HYDROGEL), ctx(WCM_HYDROCOLLOID), ctx(WCM_CHARCOAL), ctx(SFP_IODINE)
# REMOVE: ctx(EWMA_TIME_PRACTICE)
```

### Fix 3 — Silver layering conflict (case 4 — `cat_c_dry_infected_combo`)
WCM_SILVER says "place the side with silver facing the wound bed." The reference describes applying hydrogel first, then silver on top — which means silver does not directly contact the wound bed.

```
# Application Tips — change:
OLD: "Apply hydrogel first, directly to the wound bed.
      Place the silver dressing on top of the hydrogel as the secondary antimicrobial layer."

NEW: "Apply hydrogel directly to the wound bed as the primary dressing.
      Then apply the silver dressing on top, silver side facing down toward the wound.
      The silver acts through the hydrogel layer to control infection while the
      hydrogel maintains moisture."
```

### Fix 4 — Minor text fixes (alginate description and hydrocolloid warming tip)

```
# Cases 1, 2, 3 — alginate description:
CHANGE: "forms a gel on contact with wound fluid"
TO:     "absorbs wound exudates and maintains moisture"
(WCM_ALGINATE TEXT uses the latter wording exactly)

# cat_c_film_vs_hydrocolloid — Application Tips:
REMOVE: "warm between your hands for a moment before applying to improve adherence"
(Not grounded in WCM_HYDROCOLLOID TEXT)
```
