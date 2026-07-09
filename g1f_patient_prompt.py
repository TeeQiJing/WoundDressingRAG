"""
g1f_patient_prompt.py  —  G1-F generation prompt (FYP2 patient-friendly schema)
================================================================================
The new Stage-2 generation prompt for VerdaSense FYP2 (Master Plan Part 13).

WHAT THIS IS
  - A drop-in replacement for the G1-C `SYSTEM_PROMPT` + section block in
    `wound_app_unimodal.generate_recommendation()`, producing the **patient-facing
    9-section schema** that `wound_testset_v5.json` references are written in.
  - Framework-agnostic: `build_g1f_messages(...)` returns (system_str, human_str);
    wrap them in SystemMessage/HumanMessage in your harness.

WHAT STAYS THE SAME (do NOT change for v5)
  - Stage-1 retrieval: R1-C multi-axis on db_wound_care_v5_bge (frozen winner).
  - The classifier (`classify_wound`) — rule layer is unchanged.

ABLATION ROLE
  - G1-C (FYP1 clinician 9-section)  = frozen comparator
  - G1-E (G1-C + Part 12 clinical fixes)
  - G1-F (this file — patient schema)  ← the new prompt
  - G4-A/B add the VLM caption on top of G1-F (slot already wired below; inert when
    `vlm_caption=""`).

GUARDRAILS BAKED IN (Master Plan Part 13.3)
  G1  dressing TYPE only from the Binding Algorithm (rule layer) — never invented
  G2  product BRANDS only if quoted in the retrieved Sources — never guessed
  G3  antibiotics: advise clinician review, never tell the patient to self-medicate
  G4  referral: "Yes" + reason when flagged
  G5  the ⚠️ Warning section is mandatory in every output
"""

# ──────────────────────────────────────────────────────────────────────────────
# G1-F SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_G1F = """You are VerdaSense, an evidence-grounded wound-care assistant writing DIRECTLY TO A PATIENT who is caring for their own wound at home. Write in clear, warm, plain language a non-medical person can follow. Define any clinical term in plain words the first time it appears, e.g. "slough (soft dead tissue)". Keep it concise and mobile-friendly: at most 2–3 short sentences (or a few short bullets) per section.

You are given: (a) the patient's T.I.M.E. wound assessment and the rule-based classification; (b) a BINDING CLINICAL ALGORITHM for this wound type; (c) numbered RETRIEVED GUIDELINE SOURCES; and optionally (d) a clinician-style CAPTION of the wound photograph.

GROUNDING & CITATION (always produced; the patient app hides citations, the evaluator keeps them):
1. Every clinical claim must be supported by the Binding Algorithm or a numbered Source. Put the citation after the claim as [S1], [S2], [S1, S3]. Do NOT use outside knowledge or training data.
2. If a section has no supporting evidence, write exactly: "No specific guidance was retrieved for this." Do NOT invent content.

SAFETY GUARDRAILS — non-negotiable:
G1. DRESSING TYPE COMES FROM THE RULE LAYER. The dressing categories in "Dressing You Need" MUST be chosen only from the dressing list in the BINDING CLINICAL ALGORITHM (Source 1). Never invent, add, or substitute a dressing type.
G2. PRODUCTS ARE QUOTED, NEVER INVENTED. In "Example Products" you may name a brand ONLY if that exact product name appears in the retrieved Sources. If no product appears in the Sources, give the dressing type only and write "(no specific product example retrieved)". Never guess a brand.
G3. ANTIBIOTICS. Never tell the patient to take, buy, or stop antibiotics. If antibiotics are flagged, say the wound shows signs of infection and they should see a clinician for assessment (a wound swab / culture) who will decide.
G4. REFERRAL. If referral is flagged, "Do You Need to See a Doctor?" MUST begin with the word "Yes" and give the reason. If not flagged, say it is not urgently required but to keep monitoring.
G5. The "## ⚠️ Warning — Get Help Now" section is MANDATORY in every response and must list the red-flag signs that mean seek urgent care (spreading redness, warmth, swelling, increasing pain, pus, fever, or a bad smell). Never soften or omit it.

USING THE WOUND-IMAGE CAPTION (only if one is provided):
- Use it to make "Your Wound" more specific and to CROSS-CHECK the T.I.M.E. labels.
- If the caption disagrees with the labels (e.g. the photo suggests infection but the label says "not infected"), do NOT change the rule-based dressing type. Instead add a clear advisory in "Do You Need to See a Doctor?" and the Warning that the photo and the automated reading disagree and a clinician should review.
- If no caption is provided, ignore image considerations entirely.

OUTPUT — use EXACTLY these nine headings, in this order, none added, renamed, or omitted:
## Your Wound
## Dressing You Need
## Example Products
## Dressings to Avoid
## How Often to Change
## Antibiotics?
## Do You Need to See a Doctor?
## Step-by-Step Care
## ⚠️ Warning — Get Help Now"""


# ──────────────────────────────────────────────────────────────────────────────
# Human-prompt assembly
# ──────────────────────────────────────────────────────────────────────────────
def _ev_field(ev, key, default=""):
    """Read a field from an evidence item that may be a dict or a LangChain doc."""
    if isinstance(ev, dict):
        if key == "raw_text":
            return ev.get("raw_text") or ev.get("text") or ev.get("page_content", "")
        return ev.get(key, default)
    md = getattr(ev, "metadata", {}) or {}
    if key == "raw_text":
        return md.get("raw_text") or getattr(ev, "page_content", "") or ""
    if key == "source":
        return md.get("source", default)
    return md.get(key, default)


def _split_binding(evidence, wound_type):
    """Pick the Binding Algorithm chunk (Source 1) = first evidence whose wound_type
    metadata matches; fall back to first chunk containing 'Wound Type {wt}'."""
    wt = str(wound_type)
    binding, rest = None, []
    for ev in evidence:
        if binding is None and str(_ev_field(ev, "wound_type")) == wt:
            binding = ev
        else:
            rest.append(ev)
    if binding is None:  # fallback: keyword match
        rest = []
        for ev in evidence:
            if binding is None and f"Wound Type {wt}" in _ev_field(ev, "raw_text"):
                binding = ev
            else:
                rest.append(ev)
    return binding, rest


def _sources_block(binding, rest):
    """Number the sources: S1 = binding algorithm, S2..N = supporting evidence."""
    lines, idx = [], 1
    if binding is not None:
        src = _ev_field(binding, "source", "Unknown")
        lines.append(f"[S{idx}] (BINDING CLINICAL ALGORITHM — dressing TYPE must come from here) "
                     f"— {src}\n{_ev_field(binding, 'raw_text')}")
        idx += 1
    for ev in rest:
        src = _ev_field(ev, "source", "Unknown")
        auth = _ev_field(ev, "authority", "")
        tag = f"{src}{(' [' + auth + ']') if auth else ''}"
        lines.append(f"[S{idx}] — {tag}\n{_ev_field(ev, 'raw_text')}")
        idx += 1
    return "\n\n".join(lines)


def _mandatory_block(classifier):
    wt = classifier.get("wound_type")
    out = []
    if classifier.get("referral_required"):
        reason = classifier.get("escalation_reason") or ""
        out.append(f'REFERRAL IS REQUIRED for Wound Type {wt}. '
                   f'"## Do You Need to See a Doctor?" MUST begin with "Yes" and give the reason'
                   f'{" (" + reason + ")" if reason else ""}.')
    else:
        out.append('Referral is NOT required: say it is not urgently needed but to keep monitoring.')
    if classifier.get("antibiotic_required"):
        sub = (" The structured label may say 'not infected' but the notes/image suggest infection — "
               "address this discrepancy.") if classifier.get("subclinical_infection") else ""
        out.append('ANTIBIOTICS FLAGGED: in "## Antibiotics?" state the wound shows signs of infection '
                   'and the patient should see a clinician for assessment (a wound swab / culture). '
                   'Do NOT tell the patient to take antibiotics.' + sub)
    else:
        out.append('Antibiotics not flagged: say they are not needed (or "may or may not be needed, '
                   'depending on the cause" if the wound type is borderline).')
    return "\n".join(f"- {x}" for x in out)


_SECTION_GUIDE = """Fill EXACTLY these sections (plain language, cite [S#] after claims):
## Your Wound                 — 1–2 sentences describing the wound (use the caption if provided).
## Dressing You Need          — Primary + Secondary dressing TYPE, taken only from the Binding Algorithm.
## Example Products           — brand(s) ONLY if named in the Sources, with their generic class; else type only.
## Dressings to Avoid         — contraindications from the Sources/algorithm (state the trigger).
## How Often to Change        — per dressing, from the Sources.
## Antibiotics?               — per the antibiotic instruction above (never advise self-medication).
## Do You Need to See a Doctor? — per the referral instruction above.
## Step-by-Step Care          — short numbered steps from the protocol Source.
## ⚠️ Warning — Get Help Now  — mandatory red-flag signs to seek urgent care."""


def build_g1f_human_prompt(assessment_text, classifier, evidence, vlm_caption=""):
    """Assemble the G1-F human prompt.

    assessment_text : the T.I.M.E. payload as text (e.g. the testset `user_input`)
    classifier      : dict from classify_wound() (wound_type, referral_required,
                      antibiotic_required, etiology, subclinical_infection, ...)
    evidence        : list of retrieved chunks — dicts (build_evidence_list output)
                      or LangChain docs (duck-typed).
    vlm_caption     : optional VLM wound-image caption (G4); "" = no image.
    """
    wt = classifier.get("wound_type")
    binding, rest = _split_binding(evidence, wt)
    caption_block = (f"WOUND-IMAGE CAPTION (clinician-style; cross-check the labels):\n{vlm_caption.strip()}"
                     if vlm_caption and vlm_caption.strip()
                     else "WOUND-IMAGE CAPTION: [none provided — ignore image considerations]")

    return f"""PATIENT WOUND ASSESSMENT (T.I.M.E.) AND CLASSIFICATION
{assessment_text}
Rule-based classification: Wound Type {wt} | referral_required={classifier.get('referral_required')} | antibiotic_required={classifier.get('antibiotic_required')} | etiology={classifier.get('etiology', 'generic')}

{caption_block}

MANDATORY CLINICAL INSTRUCTIONS (override tone, never safety):
{_mandatory_block(classifier)}

RETRIEVED EVIDENCE (cite as [S#]; [S1] is the binding algorithm — dressing TYPE comes from it):
{_sources_block(binding, rest)}

{_SECTION_GUIDE}
"""


def build_g1f_messages(assessment_text, classifier, evidence, vlm_caption="", no_think=False):
    """Return (system_prompt, human_prompt). `no_think=True` prepends /no_think (Qwen)."""
    sys_p = ("/no_think\n\n" + SYSTEM_PROMPT_G1F) if no_think else SYSTEM_PROMPT_G1F
    return sys_p, build_g1f_human_prompt(assessment_text, classifier, evidence, vlm_caption)


# ──────────────────────────────────────────────────────────────────────────────
# DEMO — builds a real prompt from a v5 testset case (no LLM call)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, os
    case_id = "cat_a_wt4"
    ts = json.load(open(os.path.join("ragas_testset", "wound_testset_v5.json"), encoding="utf-8"))
    case = next(c for c in ts if c["case_id"] == case_id)

    # Build evidence dicts from the case's ranked reference_contexts (real ai_summaries).
    evidence = []
    for text, meta in zip(case["reference_contexts"], case["reference_contexts_meta"]):
        evidence.append({"raw_text": text, "source": meta["abbrev"],
                         "wound_type": case["wound_type_expected"] if meta["grade"] == 3 and meta["role"] == "algorithm_anchor" else None})

    classifier = {"wound_type": case["wound_type_expected"],
                  "referral_required": case["referral_required"],
                  "antibiotic_required": case["antibiotic_required"],
                  "etiology": "generic", "subclinical_infection": False}

    sys_p, human_p = build_g1f_messages(case["user_input"], classifier, evidence,
                                        vlm_caption="")  # set a caption string to test G4 slot
    print("=" * 90, f"\nG1-F PROMPT DEMO — case {case_id}\n", "=" * 90)
    print("\n----- SYSTEM (first 600 chars) -----\n" + sys_p[:600] + " ...")
    print("\n----- HUMAN -----\n" + human_p[:2200] + "\n... [truncated]")
    print("\nSlots verified: binding S1 present:",
          "[S1] (BINDING" in human_p, "| caption slot present:", "WOUND-IMAGE CAPTION" in human_p,
          "| 9 headings in guide:", human_p.count("## "))
