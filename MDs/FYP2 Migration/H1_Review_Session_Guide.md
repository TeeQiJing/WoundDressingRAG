# H1 — Clinical Review with Ms Saw: Session Guide

**Goal:** get every clinically-decisive field in the 34-case testset validated by Ms Saw in **one pass**, and record her agreement rate → the **Clinical Concordance Rate** baseline (the H1 result). This is the FYP2 deliverable your supervisor prioritised.

---

## The package (already generated)

| File | What it is |
|---|---|
| **`ragas_testset/h1_review.html`** | The whole review — open in any browser (offline, mobile-friendly). 34 cases, each with the wound photo, T.I.M.E. labels, AI caption, and **pre-filled gold** (dressing + antibiotic + referral), plus 5 tick-boxes. Starts with the **invariants sheet** (WT1–8 dressing map + 6 KB conflicts). |
| `ragas_testset/build_h1_review.py` | Regenerates the HTML from the JSON if the testset changes (`python ragas_testset/build_h1_review.py`). |

**Her surface is tiny by design** (Testset Plan §5): she taps a box and only writes when she *disagrees*. She is **not** asked about citations, prose, retrieval, or any metric.

### The 5 decisions per case
1. **Image suitable** for this wound type? (Yes / No)
2. **AI caption accurate**? (Accurate / Minor errors / Misleading)
3. **Dressing** (primary + secondary, pre-filled)? (Agree / Minor fix / Disagree)
4. **Antibiotic** (pre-filled Yes/No)? (Agree / Disagree)
5. **Referral** (pre-filled Yes/No)? (Agree / Disagree)
+ debridement tick for WT5–8, + one comment line.

### Reviewed once (not per case) — the invariants sheet
- WT1–8 → allowed / avoid dressing map (confirm the algorithm once).
- **6 KB conflicts** (C1–C5 + Q8), each with a working default — she confirms or overrides. These are the items you already flagged to her on WhatsApp.

---

## How to run it (the method that guarantees completion)

She didn't finish the docx last time → **do not send a form and hope.** Run it as a **live 30–45 min call**:

1. **1–2 days before:** send the HTML and ask her to glance at the **Invariants section only** (top of the page) — the WT map + the 6 KB conflicts.
2. **On the call:** screen-share `h1_review.html`. Go case by case; she says her call out loud, **you tick the boxes**. Surgeons talk faster than they type — this is why it completes.
3. **Batched by wound type** (the file is already ordered A→G, and Cat A is WT1→WT8) so her clinical "mode" stays fixed.
4. **Hard-cap one session.** If she's energetic, you get comments too; if not, you still captured every decisive field.

---

## After the session → freeze the gold

1. Fold her ticks into the JSON: where she **disagreed**, update the gold; where she **agreed**, tag `clinician_validated: true` (+ her note).
2. Compute her **Agree-rate per field** (image / caption / dressing / antibiotic / referral) = **Clinical Concordance Rate** (the H1 headline number).
3. Save as `wound_testset_v5_GOLD.json` and freeze — all reported ablation numbers then cite the clinician-validated gold.

I can build the "fold-answers-back + concordance" script when you have her responses.

---

## Ready-to-send WhatsApp message (edit as you like)

> Hi Ms Saw, thank you again for helping with my FYP. I've put everything into **one simple review sheet** — 34 wound cases, each already filled in with the recommended dressing, antibiotic, and referral. For each case you just tick **Agree** or note a correction — no writing from scratch. It also has a short one-page summary (the wound-type dressing table + the 6 dressing questions I sent you) to confirm once at the start.
>
> Could we do a **short 30–40 min video call** where I share my screen and we go through it together (I'll tick as you tell me)? It's much faster than typing. I'll send the sheet ahead so you can glance at the one-page summary first. What day/time next week suits you?

---

*The golden testset is the measuring stick for every FYP2 number — this review is the highest-leverage remaining task. Keep her surface tiny, pre-fill aggressively, run it once.*
