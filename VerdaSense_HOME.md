---
title: VerdaSense — Home
type: dashboard
project: VerdaSense Wound-Dressing RAG (FYP)
updated: 2026-07-20
tags: [fyp2, home, moc]
---

# 🩹 VerdaSense — Project Home

> A hybrid **multimodal RAG** clinical decision-support system for wound-dressing recommendation (T.I.M.E. framework + MOH wound-type algorithm). This note is the map of the whole project — start here.

**Pin this note** (right-click tab → Pin) and set it as the start page (Settings → set as homepage if you install the *Homepage* plugin).

---

## 🚦 Status at a glance

| Area                               | State                 |
| ---------------------------------- | --------------------- |
| Retrieval ablations (R1–R5)        | ✅ done                |
| Generation ablations (G1–G3)       | ✅ done                |
| Multimodal ablations (G4-P/A/B/C)  | ✅ done                |
| Testset v5 (34 cases, curated)     | ✅ frozen-pending-H1   |
| H1 review package built              | ✅ done                |
| **H1 — sent to Ms Saw?**           | ⏳ **NO — the blocker** |
| Report chapters                    | 🚧 in progress        |

---

## 🧪 Ablation tracker

| Stage      | Exp  | Question                   | Winner / result                   | Analysis                                     |
| ---------- | ---- | -------------------------- | --------------------------------- | -------------------------------------------- |
| Retrieval  | R1   | Query strategy             | R1-C multi-axis (+4.4 pp CR)      | [[R1_Query_Strategy_Analysis]]               |
| Retrieval  | R2   | Retrieval method           | R2-A dense (hybrid didn't help)   | [[R2_Retrieval_Strategy_Analysis]]           |
| Retrieval  | R3   | Top-K                      | R3-C k=6 (CR peaks)               | [[R3_Guidance_Analysis]]                     |
| Retrieval  | R4   | Embedding                  | R4-B BGE-large                    | [[R4_Embedding_Model_Analysis]]              |
| Retrieval  | R5   | Caption in retrieval?      | **No** — hurts (−18.75 pp HR@6)   | [[R5_Multimodal_Caption_Retrieval_Analysis]] |
| Generation | G1   | Prompt strategy            | G1-C grounded (FA 0.69→0.81)      | [[G1_Prompt_Strategy_Analysis]]              |
| Generation | G2   | Closed-source LLM          | G2-D Gemini (FA 0.815)            | [[G2_LLM_Comparison_Analysis]]               |
| Generation | G3   | Open-source LLM            | none matched closed               | [[G3_OpenSource_LLM_Analysis]]               |
| Multimodal | G4-P | How to prompt the VLM      | **P4 blind — 100% VLM-DISC**      | [[G4P_VLM_Prompt_Strategy_Analysis]]         |
| Multimodal | G4-A | Caption vs no-caption      | FA/safety-neutral; directionality | [[G4A_Multimodal_Caption_Analysis]]          |
| Multimodal | G4-B | Closed VLM (GPT vs Gemini) | GPT-4o-mini (Gemini 41% refuse)   | [[G4B_VLM_Comparison_Analysis]]              |
| Multimodal | G4-C | Open-source VLM            | Qwen2.5-VL-72B best open          | [[G4C_OpenSource_VLM_Analysis]]              |

📊 **Full consolidated write-up:** [[VerdaSense_Complete_Ablation_Summary]]

---

## 🗺️ Plans & strategy
- [[VerdaSense_FYP2_Master_Plan]] — the living master plan (consult before any FYP2 work)
- [[VerdaSense_FYP2_Ablation_Map_v5]] — the eval plan (what to run, why)
- [[VerdaSense_FYP2_Testset_Construction_and_Review_Plan]] — testset + Ms Saw review
- [[VerdaSense_Testset_v5_Plan]] · [[VerdaSense_Paper_Analysis_and_FYP2_Meeting_Brief]]

## 🧾 Testset & data
- Testset: `ragas_testset/wound_testset_v5.json` (34 cases) · viewer: `ragas_testset/testset_viewer.html`
- Curated images: `ragas_testset/wound_images/` (14 distinct)

## 👩‍⚕️ Human evaluation (H1) — the deliverable
- [[H1_Review_Session_Guide]] — how to run the Ms Saw session + WhatsApp message
- Review sheet: `ragas_testset/h1_review.html` — **built ✅, not yet sent ⏳**. Self-contained offline review: auto-saves to her browser, per-case/per-question ✕ clear, 🗑 Clear all, **Download my answers** → JSON. Part 1 = the 8 MOH↔DyaMed questions, Part 2 = the 34 cases. Rebuild: `python ragas_testset/build_h1_review.py`
- **Delivery:** email the single HTML file — *not* HuggingFace/public hosting (no backend to receive answers; clinical images shouldn't be public)
- ⏳ To build **after** her answers return: fold-back script → concordance % + Cohen's κ + disagreement diff

## 🎓 Report & viva
- [[VerdaSense_Viva_QA_Cheatsheet]] — hard questions + evidence-backed answers
- Figures: `report_figures/` (fig1–fig4)

## ⚙️ System reference
- [[CLAUDE]] — architecture, ablation-best config, current status

---

## 🔖 How to keep this organised (conventions)

**Tags** (type `#` in any note): `#ablation/retrieval` `#ablation/generation` `#ablation/multimodal` `#testset` `#plan` `#viva` `#done` `#pending`

**Frontmatter** — add this block to the top of each ablation note so the live tracker (below) works:
```yaml
---
type: ablation
stage: multimodal        # retrieval | generation | multimodal
exp: G4-C
status: done             # done | running | pending
winner: Qwen2.5-VL-72B
key_metric: "VLM-DISC / accuracy"
tags: [ablation/multimodal, done]
---
```

**Live tracker (Dataview).** With the Dataview plugin enabled and this note in **Reading view**, the table below builds itself from every note tagged `#ablation`:

```dataview
TABLE WITHOUT ID exp AS "Exp", stage AS "Stage", status AS "Status", winner AS "Winner", key_metric AS "Key metric"
FROM #ablation
SORT stage ASC, exp ASC
```

> **Seeing raw text instead of a table?** (1) Settings → Community plugins → Browse → install & **enable "Dataview"**; (2) switch this note to **Reading view** — the 📖 book icon at the top-right (Dataview never renders in *Source* edit mode).

**Graph view** (Ctrl/Cmd+G) will show every `[[link]]` above as a connected map — a nice figure for your report on how the experiments relate.
