"""
build_h1_review.py — generates the H1 clinician-review package (Ms Saw) from the testset.

Renders a SELF-CONTAINED, OFFLINE-CAPABLE `ragas_testset/h1_review.html` from
`wound_testset_v5.json` (+ frozen GPT-4o-mini captions). She can review at her own pace
(answers auto-save to her browser across sittings), then click "Download my answers" to
send one JSON file back — no hosting, no backend, keeps patient data off public servers.

Includes: 34 cases (image + T.I.M.E. + AI caption + pre-filled gold + tick decisions),
an invariants sheet, and the 8 KB-conflict questions (merged from the WhatsApp message).

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
    m = re.search(rf"##\s*{re.escape(header)}\s*\n(.*?)(?:\n##|\Z)", ref_text, re.S | re.I)
    if not m: return ""
    body = re.sub(r"\[S\d+\]", "", m.group(1).strip())
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body).strip()

def yn(b): return "YES" if b else "No"

def radio(name, options):
    """A single-choice radio group. `name` is the unique answer key; options are clean values."""
    return " ".join(
        f'<label class="tk"><input type="radio" name="{name}" value="{html.escape(o)}"> {html.escape(o)}</label>'
        for o in options)

def textfield(fid, ph="if you disagree / a correction…"):
    return f'<input type="text" id="{fid}" class="cmt" placeholder="{ph}">'

# ── the 8 KB-conflict questions (verbatim intent from the WhatsApp message) ────
KB_Q = [
    ("Q1", "MASTER RULE — When MOH guidelines and DyaMed charts disagree, which should the AI default to?",
     ["MOH default", "DyaMed (hospital) default", "Case-by-case"]),
    ("Q2", "Carbon (Zorflex LA): MOH limits charcoal to WT8 (infected/odour); DyaMed uses it across most types. Allow Zorflex LA on a CLEAN, non-infected wound (WT1)?",
     ["Allow on WT1", "No — MOH (infected/odour only)"]),
    ("Q3", "Alginogel on a DRY wound (WT7): DyaMed suggests Flaminal Hydro. Is alginogel OK here, or stick to a standard hydrogel?",
     ["Alginogel OK", "Use hydrogel instead"]),
    ("Q4", "Heavy-exudate SECONDARY: DyaMed flowchart says Drawtex; the 'Pemilihan Material' sheet says Gauze & Gamgee. Which first?",
     ["Drawtex first", "Gauze & Gamgee first"]),
    ("Q5", "Is Drawtex the go-to absorbent secondary for heavy exudate?", ["Yes", "No"]),
    ("Q6", "Exudate matching — Flaminal Forte (heavy) · Flaminal Hydro (light–mod) · Foam (mod). Correct?",
     ["Yes", "No"]),
    ("Q7", "WT5 (dry / sloughy / non-infected): is home softening with hydrogel + routine review OK, or must it always be referred?",
     ["Home hydrogel + review OK", "Must refer"]),
    ("Q8", "'Pemilihan Material' names non-DyaMed brands (Aquacel Ag, Activon honey, Kaltostat, Winner Foam, Anscare/Cavidagel). Stick to DyaMed as product examples, or OK to name others?",
     ["DyaMed only", "OK to name others"]),
]

def case_card(c):
    cid = c["case_id"]; tp = c["time_payload"]
    cap = (caps.get(cid, {}) or {}).get("caption", "").strip() or "(caption shown live in the app)"
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
    deb = (f'<div class="q"><span class="qn">Debridement appropriate? (WT5–8)</span> {radio(cid+"_deb", ["Yes","No"])}</div>'
           if (c.get("wound_type_expected") or 0) >= 5 else "")
    return f"""
    <div class="card" data-case="{cid}">
      <div class="chdr"><span class="cid">{cid}</span>
        <span class="chr"><span class="pill">{c['category']}{(' · WT' + str(c['wound_type_expected'])) if c.get('wound_type_expected') else ''}</span>
        <button class="clr" onclick="clearScope('{cid}_')">✕ clear this case</button></span></div>
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
        <div class="q"><span class="qn">1. Image suitable for this wound type?</span> {radio(cid+'_img', ['Yes','No'])}</div>
        <div class="q"><span class="qn">2. AI caption accurate?</span> {radio(cid+'_cap', ['Accurate','Minor errors','Misleading'])}</div>
        <div class="q"><span class="qn">3. Dressing (primary + secondary above)?</span> {radio(cid+'_dr', ['Agree','Minor fix','Disagree'])}</div>
        <div class="q"><span class="qn">4. Antibiotic ({yn(c['antibiotic_required'])})?</span> {radio(cid+'_ab', ['Agree','Disagree'])}
             &nbsp;&nbsp;<span class="qn">5. Referral ({yn(c['referral_required'])})?</span> {radio(cid+'_rf', ['Agree','Disagree'])}</div>
        {deb}
        <div class="q comment"><span class="qn">Comment / correction:</span> {textfield(cid+'_note')}</div>
      </div>
    </div>"""

# ── invariants: WT1–8 dressing map (from Cat A) ───────────────────────────────
wt_rows = []
for c in [x for x in cases if x["category"] == "A"]:
    ex = "; ".join(f"{k}: {v}" for k, v in (c.get("example_products") or {}).items())
    wt_rows.append((c.get("wound_type_expected"), ", ".join(c.get("allowed_dressings", [])),
                    ", ".join(c.get("contraindicated_dressings", [])) or "—",
                    f"abx {yn(c['antibiotic_required'])} · referral {yn(c['referral_required'])}", ex))
wt_rows.sort(key=lambda r: r[0] or 0)
wt_table = "".join(f"<tr><td>WT{r[0]}</td><td>{html.escape(r[1])}</td><td>{html.escape(r[2])}</td>"
                   f"<td>{r[3]}</td><td>{html.escape(r[4])}</td></tr>" for r in wt_rows)
kb_rows = "".join(
    f'<div class="q kbq"><div class="kbtxt"><b>{qid}.</b> {html.escape(txt)}'
    f' <button class="clr" onclick="clearScope(\'KB_{qid}\')">✕ clear</button></div>'
    f'<div>{radio("KB_"+qid, opts)}</div>{textfield("KB_"+qid+"_note", "notes (optional)…")}</div>'
    for qid, txt, opts in KB_Q)

body = []
for cat in ORDER:
    grp = [c for c in cases if c["category"] == cat]
    if not grp: continue
    body.append(f'<h2 class="cat">{CAT_NAME[cat]} <span class="n">({len(grp)} cases)</span></h2>')
    body += [case_card(c) for c in grp]

N = len(cases)
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VerdaSense — Clinical Review (Ms Saw) · H1</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:0 auto;padding:0 16px 60px;color:#1a2230;background:#f6f8fb;line-height:1.4}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#5b6b82;font-size:13px}}
 .box{{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:14px 16px;margin:12px 0}}
 h2.cat{{font-size:17px;margin:26px 0 8px;color:#0b7;border-bottom:2px solid #0b7;padding-bottom:4px}} .cat .n{{color:#889;font-weight:400;font-size:13px}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px}} th,td{{border:1px solid #dce3ec;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#eef3f9}}
 .card{{background:#fff;border:1px solid #dce3ec;border-radius:10px;margin:10px 0;overflow:hidden}}
 .chdr{{background:#eef3f9;padding:6px 12px;display:flex;justify-content:space-between;align-items:center}}
 .chr{{display:flex;gap:8px;align-items:center}}
 .clr{{border:1px solid #d9b3b3;background:#fff;color:#b02a2a;border-radius:5px;padding:1px 8px;font-size:11px;cursor:pointer}}
 .clr:hover{{background:#fdeaea}}
 .cid{{font-weight:700;font-size:14px}} .pill{{background:#0b7;color:#fff;border-radius:10px;padding:1px 9px;font-size:11px}}
 .cbody{{display:flex;gap:12px;padding:12px;flex-wrap:wrap}} .cimg img{{width:200px;border-radius:6px;border:1px solid #ccd}}
 .cinfo{{flex:1;min-width:240px;font-size:12.5px}} .time{{background:#f0f4f9;padding:5px 8px;border-radius:5px;margin-bottom:6px}}
 .notes{{color:#5a7;background:#f4faf0;padding:4px 8px;border-radius:5px;margin-bottom:6px}}
 .cap{{color:#446;font-style:italic;margin-bottom:6px}} .gold{{background:#fffdf3;border:1px solid #ece3bf;border-radius:6px;padding:8px}}
 .gold>div{{margin:2px 0}} .avoid{{color:#a33}} .esc{{color:#b60}}
 .dec{{border-top:1px dashed #cdd;padding:10px 12px;background:#fbfcfe}} .q{{margin:7px 0;font-size:13px}}
 .qn{{font-weight:600;margin-right:6px}} .tk{{margin-right:12px;white-space:nowrap;cursor:pointer}}
 input[type=radio]{{transform:scale(1.15);margin-right:3px;cursor:pointer}}
 .cmt{{width:70%;padding:4px 6px;border:1px solid #bcc6d2;border-radius:5px;font-size:12.5px}}
 .kbq{{border-bottom:1px solid #eef;padding-bottom:8px;margin-bottom:8px}} .kbtxt{{margin-bottom:4px}}
 /* sticky toolbar */
 #bar{{position:sticky;top:0;z-index:20;background:#0b7;color:#fff;padding:8px 12px;border-radius:0 0 10px 10px;
   display:flex;gap:10px;align-items:center;flex-wrap:wrap;box-shadow:0 2px 8px rgba(0,0,0,.15)}}
 #bar b{{font-size:14px}} #bar .prog{{background:rgba(255,255,255,.2);padding:2px 8px;border-radius:10px;font-size:12px}}
 #bar input{{border:none;border-radius:5px;padding:4px 8px;font-size:12px}}
 #bar button{{border:none;border-radius:6px;padding:6px 12px;font-weight:600;cursor:pointer;font-size:12.5px;background:#fff;color:#0a6}}
 #saved{{font-size:11px;opacity:.9}}
 @media print{{#bar{{display:none}} body{{background:#fff}} .card,.box{{break-inside:avoid}}}}
</style></head><body>

<div id="bar">
  <b>VerdaSense Review</b>
  <span>Reviewer:</span> <input id="reviewer" placeholder="your name" style="width:110px">
  <span class="prog" id="prog">0 / {N} cases</span>
  <span style="flex:1"></span>
  <button onclick="downloadJSON()">💾 Download my answers</button>
  <button onclick="downloadCSV()" style="color:#358">Download CSV</button>
  <button onclick="clearAll()" style="color:#b02a2a">🗑 Clear all</button>
  <span id="saved"></span>
</div>

<div style="padding-top:10px">
<h1>VerdaSense — Wound Dressing Recommendation: Clinical Review</h1>
<div class="sub">Ms Saw · {N} cases + 8 guideline questions · answers <b>save automatically in this browser</b> — review over as many sittings as you like, then press <b>“Download my answers”</b> and send me the file. 🙏</div>

<div class="box">
 <b>How this works — for each case, 5 quick taps:</b> 1. Image suitable? · 2. AI caption accurate? ·
 3. Dressing (pre-filled) agree/fix? · 4. Antibiotic agree? · 5. Referral agree? &nbsp;— only type in the
 comment box if you disagree. You are <b>not</b> asked about citations, wording, or any metric.
</div>

<h2 class="cat">Part 1 — 8 guideline questions (MOH ↔ DyaMed conflicts)</h2>
<div class="box">{kb_rows}</div>

<h2 class="cat">Reference — wound-type dressing map (for context, no answer needed)</h2>
<div class="box"><table><tr><th>WT</th><th>Allowed</th><th>Avoid</th><th>Abx / Referral</th><th>Example product(s)</th></tr>{wt_table}</table></div>

<h2 class="cat">Part 2 — the {N} cases</h2>
{''.join(body)}

<div class="box" style="margin-top:20px;text-align:center">
 ✅ Finished? Press <b>“Download my answers”</b> at the top and send me the file. Thank you so much!
</div>
</div>

<script>
const KEY = "verdasense_h1_review_v1";
function collect() {{
  const d = {{}};
  document.querySelectorAll('input,textarea').forEach(el => {{
    if (el.type === 'radio') {{ if (el.checked) d[el.name] = el.value; }}
    else if (el.value) {{ d[el.id] = el.value; }}
  }});
  return d;
}}
function save() {{
  localStorage.setItem(KEY, JSON.stringify(collect()));
  const s = document.getElementById('saved'); s.textContent = '✓ saved';
  setTimeout(() => s.textContent = '', 1200);
  progress();
}}
function restore() {{
  const d = JSON.parse(localStorage.getItem(KEY) || '{{}}');
  for (const [k, v] of Object.entries(d)) {{
    const r = document.querySelector('input[type=radio][name="' + k + '"][value="' + (window.CSS ? CSS.escape(v) : v) + '"]');
    if (r) {{ r.checked = true; continue; }}
    const t = document.getElementById(k);
    if (t) t.value = v;
  }}
  progress();
}}
function progress() {{
  const d = collect();
  const answered = new Set();
  document.querySelectorAll('.card').forEach(card => {{
    const id = card.getAttribute('data-case');
    if (Object.keys(d).some(k => k.startsWith(id + '_'))) answered.add(id);
  }});
  document.getElementById('prog').textContent = answered.size + ' / {N} cases';
}}
function clearScope(prefix) {{
  document.querySelectorAll('input,textarea').forEach(el => {{
    const key = el.type === 'radio' ? el.name : el.id;
    if (key && key.startsWith(prefix)) {{
      if (el.type === 'radio') el.checked = false; else el.value = '';
    }}
  }});
  save();
}}
function clearAll() {{
  if (!confirm('Clear ALL your answers on this page? This cannot be undone.')) return;
  document.querySelectorAll('input,textarea').forEach(el => {{
    if (el.id === 'reviewer') return;
    if (el.type === 'radio') el.checked = false; else el.value = '';
  }});
  localStorage.removeItem(KEY);
  save();
}}
function triggerDownload(blob, name) {{
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}}
function downloadJSON() {{
  const out = {{ reviewer: document.getElementById('reviewer').value, date: new Date().toISOString(), answers: collect() }};
  triggerDownload(new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}}),
                  'MsSaw_review_' + (document.getElementById('reviewer').value || 'answers').replace(/\\s+/g,'_') + '.json');
}}
function downloadCSV() {{
  const d = collect(); let csv = 'key,answer\\n';
  csv += 'reviewer,"' + (document.getElementById('reviewer').value||'') + '"\\n';
  for (const [k, v] of Object.entries(d)) csv += k + ',"' + String(v).replace(/"/g,'""') + '"\\n';
  triggerDownload(new Blob([csv], {{type:'text/csv'}}), 'MsSaw_review_answers.csv');
}}
document.addEventListener('change', save);
document.addEventListener('input', e => {{ if (e.target.classList.contains('cmt') || e.target.id === 'reviewer') save(); }});
window.addEventListener('load', restore);
</script>
</body></html>"""

open(OUT, "w", encoding="utf-8").write(HTML)
print(f"[OK] Wrote {os.path.relpath(OUT, ROOT)}  ({N} cases + 8 KB questions, self-save + download enabled)")
print("     Send her the file (or host the single .html on a static site) — she reviews offline, presses")
print("     'Download my answers', and sends you back one JSON file.")
