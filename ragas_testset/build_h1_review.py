"""
build_h1_review.py — generates the H1 clinician-review package (Ms Saw) from the testset.

Renders a self-contained `ragas_testset/h1_review.html` directly from
`wound_testset_v5.json` (+ frozen GPT-4o-mini captions), so the form always matches
the data. Implements the one-pass design (Testset Plan §5): 5 tick-decisions per case,
pre-filled gold, an invariants sheet reviewed once, and the KB-conflict rulings.

Run: python ragas_testset/build_h1_review.py   (from project root)
"""
import json, os, re, base64, io, html
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "ragas_testset", "wound_testset_v5.json")
CAPS    = os.path.join(ROOT, "RAGAS_EVAL", "G4A_Multimodal_Caption", "results", "G4A_vlm_captions.json")
OUT     = os.path.join(ROOT, "ragas_testset", "h1_review.html")

cases = json.load(open(TESTSET, encoding="utf-8"))
caps  = json.load(open(CAPS, encoding="utf-8")) if os.path.exists(CAPS) else {}

CAT_NAME = {"A": "Cat A — Canonical wound types 1–8", "B": "Cat B — Comorbidity / contraindication (note-driven)",
            "C": "Cat C — Escalation logic", "D": "Cat D — Depth / cavity", "E": "Cat E — Complex chronic",
            "F": "Cat F — Image robustness", "G": "Cat G — Adversarial (image contradicts the label)"}
ORDER = ["A", "B", "C", "D", "E", "F", "G"]

def img_b64(ref):
    p = os.path.join(ROOT, ref)
    try:
        im = Image.open(p).convert("RGB"); im.thumbnail((380, 380))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""

def section(ref_text, header):
    """Pull one '## <header>' section out of the patient-friendly reference."""
    m = re.search(rf"##\s*{re.escape(header)}\s*\n(.*?)(?:\n##|\Z)", ref_text, re.S | re.I)
    if not m: return ""
    body = m.group(1).strip()
    body = re.sub(r"\[S\d+\]", "", body)                       # drop [S#] cites
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)         # bold
    return body.strip()

def yn(b): return "YES" if b else "No"

# ── invariants: WT1–8 dressing map (from Cat A) + DyaMed examples ──────────────
wt_rows = []
for c in [x for x in cases if x["category"] == "A"]:
    wt = c.get("wound_type_expected")
    allowed = ", ".join(c.get("allowed_dressings", []))
    contra  = ", ".join(c.get("contraindicated_dressings", [])) or "—"
    ex = "; ".join(f"{k}: {v}" for k, v in (c.get("example_products") or {}).items())
    wt_rows.append((wt, allowed, contra, f"abx {yn(c['antibiotic_required'])} · referral {yn(c['referral_required'])}", ex))
wt_rows.sort(key=lambda r: r[0] or 0)

KB_CONFLICTS = [
    ("C1", "Carbon (Zorflex / Zorflex LA): DyaMed lists it for WT1–7, but MOH uses charcoal for WT8 (infected/odour) only.",
     "Default: follow MOH — carbon only for infected/odour wounds."),
    ("C2", "Drawtex (hydroconductive) is not a named MOH dressing category.",
     "Default: allow as a high-exudate absorbent secondary."),
    ("C3", "High-exudate secondary: Drawtex (per WT protocol) vs Gauze & Gamgee (per selection tree).",
     "Default: follow the exudate-selection tree."),
    ("C4", "Alginogel (Flaminal Hydro) on a DRY wound (WT7).",
     "Default: allow — alginogel donates moisture (unlike alginate fibre)."),
    ("C5", "Foam as a secondary on a DRY wound (WT3 / WT7).",
     "Default: low priority; advisory only."),
    ("Q8", "Product scope: DyaMed-only, or also include non-DyaMed brands (Aquacel Ag, Activon honey, Kaltostat, Winner Foam)?",
     "Default: speak in dressing classes; name DyaMed products as examples; honey gated by bee allergy."),
]

def tick(name, opts):
    """A row of checkboxes."""
    return " ".join(f'<label class="tk"><input type="checkbox" name="{name}"> {html.escape(o)}</label>' for o in opts)

def case_card(c):
    tp = c["time_payload"]
    cap = (caps.get(c["case_id"], {}) or {}).get("caption", "").strip() or "(caption shown live in the app)"
    nv = tp["necrotic_pct"] + tp["slough_pct"]
    dressing = section(c["reference"], "Dressing You Need") or "(see gold)"
    ex = "; ".join(f"{k}: {v}" for k, v in (c.get("example_products") or {}).items())
    notes = tp.get("notes", "").strip()
    cond = ", ".join(c.get("conditional_contraindications", []))
    contra = ", ".join(c.get("contraindicated_dressings", []))
    esc = " · ".join(c.get("escalation_flags_expected", []))
    img = img_b64(c["image_ref"])
    time_line = (f"N {tp['necrotic_pct']}% / S {tp['slough_pct']}% / G {tp['granulation_pct']}% "
                 f"(non-viable {nv}%) &nbsp;|&nbsp; <b>{tp['infection']}</b> &nbsp;|&nbsp; "
                 f"moisture {tp['moisture']} &nbsp;|&nbsp; edge {tp['edge']}")
    return f"""
    <div class="card">
      <div class="chdr"><span class="cid">{c['case_id']}</span>
        <span class="pill">{c['category']}{(' · WT' + str(c['wound_type_expected'])) if c.get('wound_type_expected') else ''}</span></div>
      <div class="cbody">
        <div class="cimg">{'<img src="'+img+'">' if img else '(no image)'}</div>
        <div class="cinfo">
          <div class="time">{time_line}</div>
          {f'<div class="notes">📝 Patient notes: “{html.escape(notes)}”</div>' if notes else ''}
          <div class="cap"><b>AI caption:</b> {html.escape(cap)}</div>
          <div class="gold">
            <div><b>GOLD dressing:</b> {dressing}</div>
            {f'<div class="ex"><b>Example product(s):</b> {html.escape(ex)}</div>' if ex else ''}
            {f'<div class="avoid"><b>Avoid:</b> {html.escape(contra)}</div>' if contra else ''}
            {f'<div class="avoid"><b>Conditional avoid:</b> {html.escape(cond)}</div>' if cond else ''}
            <div><b>Antibiotic:</b> {yn(c['antibiotic_required'])} &nbsp;&nbsp; <b>Referral:</b> {yn(c['referral_required'])}</div>
            {f'<div class="esc"><b>Expected action / flag:</b> {html.escape(esc)}</div>' if esc else ''}
          </div>
        </div>
      </div>
      <div class="dec">
        <div class="q"><span class="qn">1. Image suitable for this wound type?</span> {tick(c['case_id']+'_img', ['Yes','No — why:'])}</div>
        <div class="q"><span class="qn">2. AI caption accurate?</span> {tick(c['case_id']+'_cap', ['Accurate','Minor errors','Misleading'])}</div>
        <div class="q"><span class="qn">3. Dressing (primary + secondary above)?</span> {tick(c['case_id']+'_dr', ['Agree','Minor fix','Disagree — instead:'])}</div>
        <div class="q"><span class="qn">4. Antibiotic ({yn(c['antibiotic_required'])})?</span> {tick(c['case_id']+'_ab', ['Agree','Disagree'])}
             &nbsp;&nbsp;<span class="qn">5. Referral ({yn(c['referral_required'])})?</span> {tick(c['case_id']+'_rf', ['Agree','Disagree'])}</div>
        {f'<div class="q"><span class="qn">Debridement appropriate? (WT5–8)</span> {tick(c["case_id"]+"_deb", ["Yes","No"])}</div>' if (c.get('wound_type_expected') or 0) >= 5 else ''}
        <div class="q comment"><span class="qn">Comment:</span> <span class="line"></span></div>
      </div>
    </div>"""

# ── assemble ──────────────────────────────────────────────────────────────────
wt_table = "".join(f"<tr><td>WT{r[0]}</td><td>{html.escape(r[1])}</td><td>{html.escape(r[2])}</td>"
                   f"<td>{r[3]}</td><td>{html.escape(r[4])}</td></tr>" for r in wt_rows)
kb_table = "".join(f'<tr><td><b>{n}</b></td><td>{html.escape(t)}</td><td>{html.escape(d)}</td>'
                   f'<td class="rule">{tick(n+"_rule", ["Confirm default","Change:"])}</td></tr>' for n, t, d in KB_CONFLICTS)

body = []
for cat in ORDER:
    grp = [c for c in cases if c["category"] == cat]
    if not grp: continue
    body.append(f'<h2 class="cat">{CAT_NAME[cat]} <span class="n">({len(grp)} cases)</span></h2>')
    body += [case_card(c) for c in grp]

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VerdaSense — Clinical Review (Ms Saw) · H1</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:0 auto;padding:16px;color:#1a2230;background:#f6f8fb;line-height:1.4}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#5b6b82;font-size:13px;margin-bottom:14px}}
 .box{{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:14px 16px;margin:12px 0}}
 h2.cat{{font-size:17px;margin:26px 0 8px;color:#0b5;border-bottom:2px solid #0b5;padding-bottom:4px}} .cat .n{{color:#889;font-weight:400;font-size:13px}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px}} th,td{{border:1px solid #dce3ec;padding:6px 8px;text-align:left;vertical-align:top}}
 th{{background:#eef3f9}}
 .card{{background:#fff;border:1px solid #dce3ec;border-radius:10px;margin:10px 0;overflow:hidden}}
 .chdr{{background:#eef3f9;padding:6px 12px;display:flex;justify-content:space-between;align-items:center}}
 .cid{{font-weight:700;font-size:14px}} .pill{{background:#0b5;color:#fff;border-radius:10px;padding:1px 9px;font-size:11px}}
 .cbody{{display:flex;gap:12px;padding:12px}} .cimg img{{width:200px;border-radius:6px;border:1px solid #ccd}}
 .cinfo{{flex:1;font-size:12.5px}} .time{{background:#f0f4f9;padding:5px 8px;border-radius:5px;margin-bottom:6px}}
 .notes{{color:#7a4;background:#f4faf0;padding:4px 8px;border-radius:5px;margin-bottom:6px}}
 .cap{{color:#446;font-style:italic;margin-bottom:6px}} .gold{{background:#fffdf3;border:1px solid #ece3bf;border-radius:6px;padding:8px}}
 .gold>div{{margin:2px 0}} .avoid{{color:#a33}} .esc{{color:#b60}}
 .dec{{border-top:1px dashed #cdd;padding:10px 12px;background:#fbfcfe}} .q{{margin:5px 0;font-size:13px}}
 .qn{{font-weight:600;margin-right:6px}} .tk{{margin-right:12px;white-space:nowrap}} input[type=checkbox]{{transform:scale(1.15);margin-right:3px}}
 .comment .line{{display:inline-block;border-bottom:1px solid #99a;min-width:60%;height:16px}}
 .rule{{white-space:nowrap}} @media print{{body{{background:#fff}} .card,.box{{break-inside:avoid}}}}
</style></head><body>
<h1>VerdaSense — Wound Dressing Recommendation: Clinical Review</h1>
<div class="sub">Reviewer: <b>Ms Saw</b> · 34 cases · one-pass · pre-filled — tap a box, only write when you disagree.
Generated from <code>wound_testset_v5.json</code>.</div>

<div class="box">
 <b>How this works (5 taps per case):</b>
 <ol style="margin:6px 0 0;padding-left:20px;font-size:13px">
  <li><b>Image suitable</b> for this wound type? · 2. <b>AI caption</b> accurate? · 3. <b>Dressing</b> (gold shown) agree/fix? ·
      4. <b>Antibiotic</b> agree? · 5. <b>Referral</b> agree?</li>
  <li>You do <b>not</b> write descriptions or dressings from scratch — everything is pre-filled; correct only where you disagree.</li>
  <li>You are <b>not</b> asked about citations, prose wording, or any metric — only the clinical calls above.</li>
 </ol>
</div>

<h2 class="cat">Invariants — review ONCE (not per case)</h2>
<div class="box">
 <b>A) Wound-type → dressing map (confirm the algorithm mapping once):</b>
 <table><tr><th>WT</th><th>Allowed dressings</th><th>Avoid</th><th>Abx / Referral</th><th>Example product(s)</th></tr>{wt_table}</table>
</div>
<div class="box">
 <b>B) KB conflicts pending your ruling</b> (each has a working default — confirm or change):
 <table><tr><th>#</th><th>Conflict</th><th>Working default</th><th>Your ruling</th></tr>{kb_table}</table>
</div>

{''.join(body)}

<div class="box" style="margin-top:24px">
 <b>After the session:</b> agreement rate per field (image / caption / dressing / antibiotic / referral) becomes the
 <b>Clinical Concordance Rate</b> baseline (H1). Where you disagreed, the gold is updated and the case tagged
 <code>clinician_validated</code>; the file is frozen as <code>wound_testset_v5_GOLD.json</code>.
</div>
</body></html>"""

open(OUT, "w", encoding="utf-8").write(HTML)
print(f"[OK] Wrote {os.path.relpath(OUT, ROOT)}  ({len(cases)} cases, {sum(1 for c in cases if img_b64(c['image_ref']))} images embedded)")
print("     Open in a browser (double-click) — self-contained, works offline / on mobile.")
