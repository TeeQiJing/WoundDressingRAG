"""
chunk_visualiser_unstructured.py  (coordinate-fixed edition)
═════════════════════════════════════════════════════════════
Visualises unstructured.io chunks overlaid on PDF pages.
Uses the EXACT same partition_pdf + chunk_by_title as your ingestion pipeline.

Key fix over previous version:
  - unstructured hi_res renders at 300 DPI internally, NOT 200 DPI.
  - Bounding box coordinates from unstructured are in pixels at that DPI.
  - We derive the scale factor from the actual page image dimensions
    reported by unstructured vs the PDF point dimensions from PyMuPDF,
    so the overlay is always calibrated to the real render scale.

Usage:
    python chunk_visualiser_unstructured.py --pdf path/to/doc.pdf --out chunks.html
    python chunk_visualiser_unstructured.py --dir ./clinical_pdfs_v2/ --out all.html

Controls in the browser:
  • Click a chunk card  → jumps to that page, highlights its blocks
  • Click a highlight   → selects that chunk in the left panel
  • Toggle KEEP / SKIP  → mark chunks to include/exclude
  • "Export JSON"       → downloads kept_chunks.json ready for ingestion
  • Filter by T.I.M.E. tag, PDF source, language warning, search text
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Hard dependencies ─────────────────────────────────────────────
try:
    import fitz
except ImportError:
    sys.exit("❌  PyMuPDF not found.\n    Run: pip install pymupdf --break-system-packages")

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
except ImportError:
    sys.exit(
        "❌  unstructured not found.\n"
        "    Run: pip install 'unstructured[pdf]' --break-system-packages"
    )

# ── Configuration — keep in sync with ingestion pipeline ─────────
CHUNK_MAX_CHARACTERS        = 3000
CHUNK_NEW_AFTER_N_CHARS     = 2400
CHUNK_COMBINE_UNDER_N_CHARS = 500
RENDER_SCALE                = 2.0   # PyMuPDF render scale (higher = sharper image)
MAX_CHUNK_PREVIEW           = 350

COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF9F45",
    "#C77DFF", "#48CAE4", "#F72585", "#3A86FF", "#06D6A0",
    "#FB8500", "#8338EC", "#FF006E", "#FFBE0B", "#43AA8B",
    "#E63946", "#2EC4B6", "#FF9F1C", "#CBFF8C", "#9B5DE5",
]

# ── T.I.M.E. tag detection ────────────────────────────────────────
TIME_KEYWORDS = {
    "T": ["necrotic tissue", "slough", "granulat", "eschar", "fibrinous",
          "necrosis", "tissue viability", "black tissue", "yellow tissue",
          "wound bed tissue", "tissue type"],
    "I": ["infected wound", "wound infection", "biofilm", "antimicrobial dressing",
          "silver dressing", "cadexomer", "erythema", "cellulitis", "purulent",
          "critically colonised", "bacterial", "sepsis", "iodine dressing",
          "signs of infection"],
    "M": ["exudate", "maceration", "desicat", "heavily exud", "copious drainage",
          "moisture balance", "exuding wound", "wet wound", "dry wound bed",
          "absorbent dressing", "hydrofiber", "alginate", "moisture imbalance"],
    "E": ["wound edge", "epidermal margin", "epithelial migration", "wound margin",
          "periwound skin", "undermining", "rolled edge", "epibole",
          "advancing edge", "non-advancing", "epithelialis", "wound border"],
}

def extract_time_tags(text: str) -> List[str]:
    t = text.lower()
    return [tag for tag, kws in TIME_KEYWORDS.items() if any(kw in t for kw in kws)]

# ── Clinical / English content detection ─────────────────────────
CLINICAL_KEYWORDS = [
    "wound", "dressing", "exudate", "tissue", "infection", "necrotic",
    "slough", "granulation", "foam", "alginate", "hydrogel", "silver",
    "debridement", "T.I.M.E", "moisture", "epithelial", "bandage",
    "ulcer", "healing", "antimicrobial",
]

def is_english_clinical(text: str) -> bool:
    if len(text.strip()) < 30:
        return True
    if any(k.lower() in text.lower() for k in CLINICAL_KEYWORDS):
        return True
    malay_markers = [
        "adalah", "untuk", "dengan", "yang", "kepada", "dalam", "ini",
        "tidak", "atau", "boleh", "perlu", "telah", "akan", "oleh",
        "pesakit", "luka", "penjagaan", "rawatan", "klinik", "kesihatan",
    ]
    text_lower = text.lower()
    malay_hits = sum(1 for w in malay_markers if f" {w} " in f" {text_lower} ")
    if malay_hits >= 3:
        return False
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return (ascii_alpha / total_alpha) >= 0.60


# ── Element metadata extraction ───────────────────────────────────
def get_element_page(element) -> Optional[int]:
    try:
        pn = element.metadata.page_number
        if pn is not None:
            return int(pn) - 1
    except AttributeError:
        pass
    return None


def get_element_bbox(element) -> Optional[List[float]]:
    """
    Returns bbox in unstructured's native pixel coordinates.
    These are pixels at whatever DPI unstructured used internally.
    We do NOT convert here — conversion happens in JS using the
    empirically derived scale factor.
    """
    try:
        coords = element.metadata.coordinates
        if coords is None:
            return None
        pts = coords.points
        if pts is None or len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    except AttributeError:
        return None


def get_element_image_size(element) -> Optional[tuple]:
    """
    Extract the image dimensions unstructured used when it extracted coordinates.
    This is stored in coordinates.system as (width, height) in pixels.
    """
    try:
        coords = element.metadata.coordinates
        if coords is None:
            return None
        system = coords.system
        if system is None:
            return None
        # PixelSpace or similar — has width and height attrs
        w = getattr(system, 'width', None)
        h = getattr(system, 'height', None)
        if w and h:
            return (float(w), float(h))
    except AttributeError:
        pass
    return None


# ── Core: partition + chunk ───────────────────────────────────────
def partition_and_chunk(pdf_path: str):
    print(f"   Partitioning with unstructured (hi_res)... ", end="", flush=True)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    print(f"{len(elements)} elements")

    print(f"   Chunking by title... ", end="", flush=True)
    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        new_after_n_chars=CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER_N_CHARS,
    )
    print(f"{len(chunks)} chunks")
    return elements, chunks


# ── Derive scale factor per page ──────────────────────────────────
def build_page_scale_map(elements, page_sizes_pts: List[tuple]) -> Dict[int, Dict]:
    """
    For each page, find the image dimensions unstructured used (from element
    coordinate metadata), then compute the scale factor needed to map those
    pixel coords onto the PyMuPDF-rendered image.

    unstructured stores coordinates in pixels relative to the image it rendered
    internally. PyMuPDF renders at RENDER_SCALE × 72 DPI. We need:
        overlay_x = bbox_px * (pymupdf_image_width / unstructured_image_width)
        overlay_y = bbox_px * (pymupdf_image_height / unstructured_image_height)

    page_sizes_pts: list of (width_pts, height_pts) in PDF coordinate space
    """
    # Collect unstructured image dimensions per page
    unstructured_img_size = {}   # page_0idx -> (w_px, h_px)
    for elem in elements:
        page = get_element_page(elem)
        if page is None:
            continue
        if page in unstructured_img_size:
            continue
        sz = get_element_image_size(elem)
        if sz:
            unstructured_img_size[page] = sz

    # Build scale map
    scale_map = {}
    for page_idx, (pdf_w_pts, pdf_h_pts) in enumerate(page_sizes_pts):
        # PyMuPDF renders at RENDER_SCALE × 72 dpi
        pymupdf_w = pdf_w_pts * RENDER_SCALE
        pymupdf_h = pdf_h_pts * RENDER_SCALE

        if page_idx in unstructured_img_size:
            us_w, us_h = unstructured_img_size[page_idx]
            sx = pymupdf_w / us_w if us_w else 1.0
            sy = pymupdf_h / us_h if us_h else 1.0
        else:
            # Fallback: assume unstructured used 300 DPI (hi_res default)
            # PDF pts × (300/72) = px at 300 DPI
            us_w_est = pdf_w_pts * (300.0 / 72.0)
            us_h_est = pdf_h_pts * (300.0 / 72.0)
            sx = pymupdf_w / us_w_est
            sy = pymupdf_h / us_h_est
            print(f"   [WARN] Page {page_idx+1}: no unstructured img dims found, "
                  f"using 300 DPI fallback (sx={sx:.4f}, sy={sy:.4f})")

        scale_map[page_idx] = {
            "sx": sx,
            "sy": sy,
            "pymupdf_w": pymupdf_w,
            "pymupdf_h": pymupdf_h,
        }

    return scale_map


# ── Map chunks to pages and bboxes ───────────────────────────────
def extract_chunk_location(chunk) -> Dict:
    orig_elements = []
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        orig_elements = chunk.metadata.orig_elements or []
    if not orig_elements:
        orig_elements = [chunk]

    blocks = []
    pages  = set()

    for elem in orig_elements:
        page = get_element_page(elem)
        bbox = get_element_bbox(elem)   # raw unstructured px coords
        elem_text = getattr(elem, "text", "") or ""
        elem_type = type(elem).__name__

        if page is not None:
            pages.add(page)

        blocks.append({
            "page":      page,
            "bbox":      bbox,
            "text":      elem_text[:120],
            "elem_type": elem_type,
        })

    return {
        "pages":        sorted(pages),
        "primary_page": min(pages) if pages else 0,
        "blocks":       blocks,
    }


# ── Build chunk objects ───────────────────────────────────────────
def build_chunk_objects(chunks, pdf_path: str) -> List[Dict]:
    result = []
    for idx, chunk in enumerate(chunks):
        full_text = chunk.text or ""
        location  = extract_chunk_location(chunk)

        has_table = has_image = False
        if hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
            for elem in chunk.metadata.orig_elements:
                etype = type(elem).__name__
                if etype == "Table":
                    has_table = True
                if etype == "Image":
                    has_image = True

        time_tags = extract_time_tags(full_text)
        english   = is_english_clinical(full_text)

        result.append({
            "chunk_id":     idx,
            "source":       os.path.basename(pdf_path),
            "pages":        location["pages"],
            "primary_page": location["primary_page"],
            "blocks":       location["blocks"],
            "text":         full_text,
            "preview":      full_text[:MAX_CHUNK_PREVIEW].replace("<", "&lt;").replace(">", "&gt;"),
            "char_count":   len(full_text),
            "has_table":    has_table,
            "has_image":    has_image,
            "time_tags":    time_tags,
            "is_english":   english,
            "keep":         english,
            "color_idx":    idx % len(COLORS),
        })

    return result


# ── PDF page rendering ────────────────────────────────────────────
def render_pages(pdf_path: str, scale: float = RENDER_SCALE):
    doc        = fitz.open(pdf_path)
    pages_b64  = []
    page_sizes = []

    for i in range(len(doc)):
        page = doc[i]
        mat  = fitz.Matrix(scale, scale)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        buf  = io.BytesIO(pix.tobytes("png"))
        pages_b64.append(base64.b64encode(buf.getvalue()).decode())
        page_sizes.append((page.rect.width, page.rect.height))
        print(f"   Rendering pages: {i+1}/{len(doc)}", end="\r", flush=True)

    print(f"   Rendered {len(doc)} pages             ")
    doc.close()
    return pages_b64, page_sizes


# ── HTML generation ───────────────────────────────────────────────
def build_html(pdf_data: List[Dict]) -> str:
    all_chunks_serialisable = []
    for pd_item in pdf_data:
        for c in pd_item["chunks"]:
            all_chunks_serialisable.append({k: v for k, v in c.items() if k != "blocks"})

    pages_data = []
    for pd_item in pdf_data:
        scale_map = pd_item["scale_map"]
        for pnum, (img_b64, (pw, ph)) in enumerate(
            zip(pd_item["pages_b64"], pd_item["page_sizes"])
        ):
            sm = scale_map.get(pnum, {"sx": 1.0, "sy": 1.0})

            page_chunks = []
            for c in pd_item["chunks"]:
                if pnum in c["pages"]:
                    page_blocks = [
                        b for b in c["blocks"]
                        if b["page"] == pnum and b["bbox"] is not None
                    ]
                    if page_blocks or pnum == c["primary_page"]:
                        page_chunks.append({
                            "chunk_id":  c["chunk_id"],
                            "color_idx": c["color_idx"],
                            "blocks":    page_blocks,
                        })

            pages_data.append({
                "pdf":        pd_item["pdf_name"],
                "page_num":   pnum,
                "pdf_width":  pw,
                "pdf_height": ph,
                "sx":         sm["sx"],   # ← scale factors baked per page
                "sy":         sm["sy"],
                "chunks":     page_chunks,
            })

    img_map = {}
    for pd_item in pdf_data:
        for pnum, img_b64 in enumerate(pd_item["pages_b64"]):
            img_map[f"{pd_item['pdf_name']}::{pnum}"] = img_b64

    sources = [pd_item["pdf_name"] for pd_item in pdf_data]

    all_chunks_json = json.dumps(all_chunks_serialisable, ensure_ascii=False)
    pages_json      = json.dumps(
        [{k: v for k, v in p.items()} for p in pages_data],
        ensure_ascii=False
    )
    img_map_json = json.dumps(img_map)
    sources_json = json.dumps(sources)
    colors_json  = json.dumps(COLORS)
    render_scale = RENDER_SCALE

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VerdaSense — Chunk Selector</title>
<style>
:root {{
  --bg:#090e18;--surface:#101828;--surface2:#19243a;--border:#1d3048;
  --accent:#00c8ff;--accent2:#007fa8;--text:#cce4f0;--muted:#476070;
  --keep:#22c55e;--skip:#ef4444;--warn:#f59e0b;
  --table-c:#a78bfa;--image-c:#fb923c;
  --font-ui:'DM Mono',monospace;--font-body:'IBM Plex Sans',sans-serif;
}}
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--font-body);
  display:grid;grid-template-columns:400px 1fr;height:100vh;overflow:hidden}}

#left{{display:flex;flex-direction:column;border-right:1px solid var(--border);
  background:var(--surface);overflow:hidden}}
#toolbar{{padding:14px 16px 12px;border-bottom:1px solid var(--border);flex-shrink:0}}
#toolbar h1{{font-family:var(--font-ui);font-size:12px;color:var(--accent);
  letter-spacing:.12em;margin-bottom:10px}}
.stat-row{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}}
.stat{{background:var(--surface2);border:1px solid var(--border);border-radius:5px;
  padding:3px 9px;font-family:var(--font-ui);font-size:10px;color:var(--muted)}}
.stat b{{color:var(--text)}}
.controls{{display:flex;flex-direction:column;gap:6px}}
input[type=text],select{{width:100%;background:var(--bg);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-family:var(--font-body);font-size:12px;
  padding:7px 10px;outline:none;transition:border-color .2s}}
input[type=text]:focus,select:focus{{border-color:var(--accent)}}
.btn-row{{display:flex;gap:5px;margin-top:2px}}
button.action{{flex:1;padding:7px 6px;border:none;border-radius:6px;cursor:pointer;
  font-family:var(--font-ui);font-size:10px;font-weight:500;letter-spacing:.04em;
  transition:opacity .15s,transform .1s}}
button.action:hover{{opacity:.85}}
button.action:active{{transform:scale(.97)}}
.btn-keep-all{{background:var(--keep);color:#fff}}
.btn-skip-all{{background:var(--skip);color:#fff}}
.btn-export{{background:var(--accent2);color:#fff;flex:1.4}}

#chunk-list{{flex:1;overflow-y:auto;padding:8px 8px 16px}}
#chunk-list::-webkit-scrollbar{{width:3px}}
#chunk-list::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
#no-results{{display:none;padding:40px 20px;text-align:center;font-size:12px;color:var(--muted)}}

.chunk-card{{border:1px solid var(--border);border-radius:10px;padding:11px 13px;
  margin-bottom:6px;cursor:pointer;transition:border-color .15s,background .15s,opacity .15s;
  position:relative;overflow:hidden}}
.chunk-card:hover{{border-color:var(--accent2);background:var(--surface2)}}
.chunk-card.selected{{border-color:var(--accent)!important;background:var(--surface2);
  box-shadow:0 0 0 1px var(--accent)}}
.chunk-card.skipped{{opacity:.38}}
.chunk-card::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  border-radius:3px 0 0 3px}}
.chunk-card.kept::before{{background:var(--keep)}}
.chunk-card.skipped::before{{background:var(--skip)}}
.card-header{{display:flex;align-items:center;gap:5px;margin-bottom:6px;flex-wrap:wrap}}
.chunk-id{{font-family:var(--font-ui);font-size:10px;color:var(--muted);flex-shrink:0}}
.page-badge{{font-family:var(--font-ui);font-size:10px;background:var(--surface2);
  border:1px solid var(--border);border-radius:4px;padding:1px 6px;color:var(--accent2);flex-shrink:0}}
.source-badge{{font-family:var(--font-ui);font-size:9px;background:#0f2010;
  border:1px solid #1a4020;border-radius:4px;padding:1px 7px;color:#6bcb77;
  max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.content-badges{{display:flex;gap:3px}}
.badge-table{{font-family:var(--font-ui);font-size:9px;padding:1px 5px;border-radius:3px;
  background:rgba(167,139,250,.12);color:var(--table-c);border:1px solid rgba(167,139,250,.25)}}
.badge-image{{font-family:var(--font-ui);font-size:9px;padding:1px 5px;border-radius:3px;
  background:rgba(251,146,60,.12);color:var(--image-c);border:1px solid rgba(251,146,60,.25)}}
.time-tags{{display:flex;gap:3px;margin-left:auto}}
.time-tag{{font-family:var(--font-ui);font-size:10px;font-weight:500;padding:1px 5px;border-radius:3px}}
.tag-T{{background:rgba(255,112,67,.12);color:#ff7043;border:1px solid rgba(255,112,67,.25)}}
.tag-I{{background:rgba(239,83,80,.12);color:#ef5350;border:1px solid rgba(239,83,80,.25)}}
.tag-M{{background:rgba(66,165,245,.12);color:#42a5f5;border:1px solid rgba(66,165,245,.25)}}
.tag-E{{background:rgba(102,187,106,.12);color:#66bb6a;border:1px solid rgba(102,187,106,.25)}}
.tag-none{{background:rgba(255,255,255,.04);color:var(--muted);border:1px solid var(--border)}}
.chunk-preview{{font-size:11.5px;line-height:1.6;color:var(--muted);
  margin-bottom:7px;word-break:break-word}}
.chunk-preview em{{color:var(--accent);font-style:normal;background:rgba(0,200,255,.08);
  border-radius:2px;padding:0 2px}}
.card-footer{{display:flex;align-items:center;gap:5px}}
.char-count{{font-family:var(--font-ui);font-size:10px;color:var(--muted)}}
.lang-warn{{font-family:var(--font-ui);font-size:10px;color:var(--warn);
  background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);
  border-radius:3px;padding:1px 6px}}
.toggle-btn{{margin-left:auto;padding:3px 10px;border:none;border-radius:4px;
  font-family:var(--font-ui);font-size:10px;font-weight:500;cursor:pointer;transition:all .15s}}
.toggle-skip{{background:rgba(239,68,68,.12);color:var(--skip);border:1px solid rgba(239,68,68,.25)}}
.toggle-keep{{background:rgba(34,197,94,.12);color:var(--keep);border:1px solid rgba(34,197,94,.25)}}

#right{{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}}
#viewer-header{{padding:11px 20px;border-bottom:1px solid var(--border);
  font-family:var(--font-ui);font-size:11px;color:var(--muted);
  flex-shrink:0;display:flex;align-items:center;gap:12px}}
#viewer-header strong{{color:var(--text);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
#viewer-chunk-info{{color:var(--accent)}}
#page-nav{{display:flex;align-items:center;gap:6px;margin-left:auto}}
#page-nav button{{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:5px;padding:4px 10px;cursor:pointer;font-family:var(--font-ui);
  font-size:10px;transition:border-color .15s}}
#page-nav button:hover{{border-color:var(--accent)}}
#page-label{{font-family:var(--font-ui);font-size:11px;color:var(--muted)}}

/* debug info bar */
#debug-bar{{padding:4px 20px;background:rgba(0,200,255,.04);border-bottom:1px solid var(--border);
  font-family:var(--font-ui);font-size:10px;color:var(--muted);flex-shrink:0;
  display:flex;gap:16px}}
#debug-bar span{{color:var(--accent2)}}

#canvas-container{{flex:1;overflow:auto;display:flex;justify-content:center;
  align-items:flex-start;padding:24px}}
#canvas-wrapper{{position:relative;display:inline-block;line-height:0}}
#pdf-img{{display:block;border-radius:5px;box-shadow:0 8px 40px rgba(0,0,0,.6);max-width:none}}

.h-box{{position:absolute;border-radius:2px;opacity:.28;cursor:pointer;transition:opacity .15s}}
.h-box:hover{{opacity:.55}}
.h-box.selected{{opacity:.6;outline:2px solid rgba(255,255,255,.65);z-index:10}}
.h-box.skipped{{opacity:.07}}
.h-label{{position:absolute;font-family:var(--font-ui);font-size:9px;font-weight:600;
  color:#fff;padding:0px 4px;border-radius:2px;pointer-events:none;
  white-space:nowrap;z-index:11;text-shadow:0 1px 3px rgba(0,0,0,.9);line-height:1.6}}
.no-bbox-note{{position:absolute;right:8px;bottom:8px;font-family:var(--font-ui);
  font-size:10px;color:var(--muted);background:rgba(0,0,0,.5);padding:3px 8px;border-radius:4px}}
</style>
</head>
<body>

<div id="left">
  <div id="toolbar">
    <h1>▸ VERDASENSE / CHUNK SELECTOR</h1>
    <div class="stat-row">
      <div class="stat">Total <b id="st-total">0</b></div>
      <div class="stat" style="color:var(--keep)">Kept <b id="st-kept">0</b></div>
      <div class="stat" style="color:var(--skip)">Skipped <b id="st-skip">0</b></div>
      <div class="stat" style="color:var(--warn)">Non-EN <b id="st-noen">0</b></div>
      <div class="stat" style="color:var(--table-c)">Tables <b id="st-tbl">0</b></div>
    </div>
    <div class="controls">
      <input type="text" id="q-search" placeholder="Search chunk text…" />
      <select id="f-source"><option value="">All PDFs</option></select>
      <select id="f-time">
        <option value="">All T.I.M.E. tags</option>
        <option value="T">T — Tissue</option>
        <option value="I">I — Infection</option>
        <option value="M">M — Moisture</option>
        <option value="E">E — Edge</option>
        <option value="none">No tags</option>
      </select>
      <select id="f-status">
        <option value="">All chunks</option>
        <option value="kept">Kept only</option>
        <option value="skipped">Skipped only</option>
        <option value="noen">Non-English only</option>
        <option value="table">Contains table</option>
        <option value="notag">No T.I.M.E. tags</option>
      </select>
      <div class="btn-row">
        <button class="action btn-keep-all" onclick="bulkVisible('keep')">Keep Visible</button>
        <button class="action btn-skip-all" onclick="bulkVisible('skip')">Skip Visible</button>
        <button class="action btn-export"   onclick="exportKept()">Export JSON</button>
      </div>
    </div>
  </div>
  <div id="chunk-list"></div>
  <div id="no-results">No chunks match this filter.</div>
</div>

<div id="right">
  <div id="viewer-header">
    <span>PDF VIEWER</span>
    <strong id="v-pdf">—</strong>
    <span id="viewer-chunk-info" id="v-chunk"></span>
    <div id="page-nav">
      <button onclick="navPage(-1)">◀</button>
      <span id="page-label">— / —</span>
      <button onclick="navPage(+1)">▶</button>
    </div>
  </div>
  <div id="debug-bar">
    <span id="dbg-scale">scale: —</span>
    <span id="dbg-imgsize">img: —</span>
    <span id="dbg-pdfsize">pdf pts: —</span>
    <span id="dbg-blocks">blocks on page: —</span>
  </div>
  <div id="canvas-container">
    <div id="canvas-wrapper">
      <img id="pdf-img" src="" alt="" />
      <div id="overlays"></div>
    </div>
  </div>
</div>

<script>
const CHUNKS      = {all_chunks_json};
const PAGES       = {pages_json};
const IMG_MAP     = {img_map_json};
const SOURCES     = {sources_json};
const COLORS      = {colors_json};
const RENDER_SCALE = {render_scale};

let keepStatus = {{}};
let selectedId = null;
let curPageIdx = 0;

CHUNKS.forEach(c => {{ keepStatus[c.chunk_id] = c.keep; }});

const selSrc = document.getElementById('f-source');
SOURCES.forEach(s => {{
  const o = document.createElement('option');
  o.value = s; o.textContent = s; selSrc.appendChild(o);
}});

// ── Filter ────────────────────────────────────────────────────────
function filteredChunks() {{
  const q    = document.getElementById('q-search').value.toLowerCase().trim();
  const src  = document.getElementById('f-source').value;
  const time = document.getElementById('f-time').value;
  const stat = document.getElementById('f-status').value;
  return CHUNKS.filter(c => {{
    if (q    && !c.text.toLowerCase().includes(q)) return false;
    if (src  && c.source !== src) return false;
    if (time) {{
      if (time === 'none') {{ if (c.time_tags.length > 0) return false; }}
      else if (!c.time_tags.includes(time)) return false;
    }}
    if (stat === 'kept'    && !keepStatus[c.chunk_id]) return false;
    if (stat === 'skipped' &&  keepStatus[c.chunk_id]) return false;
    if (stat === 'noen'    &&  c.is_english)            return false;
    if (stat === 'table'   && !c.has_table)             return false;
    if (stat === 'notag'   &&  c.time_tags.length > 0)  return false;
    return true;
  }});
}}

// ── Render list ───────────────────────────────────────────────────
function renderList() {{
  const list  = document.getElementById('chunk-list');
  const noRes = document.getElementById('no-results');
  const fc    = filteredChunks();
  const q     = document.getElementById('q-search').value.trim();

  list.innerHTML = '';
  noRes.style.display = fc.length === 0 ? 'block' : 'none';

  fc.forEach(c => {{
    const kept  = keepStatus[c.chunk_id];
    const color = COLORS[c.color_idx];
    const pages = c.pages.length ? c.pages.map(p => p+1).join(',') : '?';
    const tags  = c.time_tags.length ? c.time_tags : ['none'];

    let preview = c.preview || '';
    if (q) {{
      const esc = q.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
      preview = preview.replace(new RegExp(`(${{esc}})`, 'gi'), '<em>$1</em>');
    }}

    const tagsHTML = tags.map(t => `<span class="time-tag tag-${{t}}">${{t}}</span>`).join('');
    const contentBadges = [
      c.has_table ? '<span class="badge-table">⊞ table</span>' : '',
      c.has_image ? '<span class="badge-image">⊡ image</span>' : '',
    ].filter(Boolean).join('');

    const card = document.createElement('div');
    card.className = `chunk-card ${{kept ? 'kept' : 'skipped'}} ${{selectedId === c.chunk_id ? 'selected' : ''}}`;
    card.dataset.id = c.chunk_id;
    card.style.borderLeftColor = color;

    card.innerHTML = `
      <div class="card-header">
        <span class="chunk-id">#${{c.chunk_id}}</span>
        <span class="page-badge">p.${{pages}}</span>
        <span class="source-badge" title="${{c.source}}">${{c.source}}</span>
        <div class="content-badges">${{contentBadges}}</div>
        <div class="time-tags">${{tagsHTML}}</div>
      </div>
      <div class="chunk-preview">${{preview}}</div>
      <div class="card-footer">
        <span class="char-count">${{c.char_count}} chars</span>
        ${{!c.is_english ? '<span class="lang-warn">non-EN</span>' : ''}}
        <button class="toggle-btn ${{kept ? 'toggle-skip' : 'toggle-keep'}}"
                onclick="toggleKeep(event,${{c.chunk_id}})">
          ${{kept ? 'SKIP' : 'KEEP'}}
        </button>
      </div>`;

    card.addEventListener('click', e => {{
      if (e.target.closest('.toggle-btn')) return;
      selectChunk(c.chunk_id);
    }});
    list.appendChild(card);
  }});

  updateStats();
}}

function updateStats() {{
  document.getElementById('st-total').textContent = CHUNKS.length;
  document.getElementById('st-kept').textContent  = CHUNKS.filter(c =>  keepStatus[c.chunk_id]).length;
  document.getElementById('st-skip').textContent  = CHUNKS.filter(c => !keepStatus[c.chunk_id]).length;
  document.getElementById('st-noen').textContent  = CHUNKS.filter(c => !c.is_english).length;
  document.getElementById('st-tbl').textContent   = CHUNKS.filter(c =>  c.has_table).length;
}}

// ── Toggle / bulk ─────────────────────────────────────────────────
function toggleKeep(evt, id) {{
  evt.stopPropagation();
  keepStatus[id] = !keepStatus[id];
  renderList();
  renderOverlays();
}}

function bulkVisible(action) {{
  filteredChunks().forEach(c => {{ keepStatus[c.chunk_id] = (action === 'keep'); }});
  renderList();
  renderOverlays();
}}

// ── Select chunk → jump to page ───────────────────────────────────
function selectChunk(id) {{
  selectedId = id;
  const c = CHUNKS.find(x => x.chunk_id === id);
  if (!c) return;
  const idx = PAGES.findIndex(p => p.pdf === c.source && p.page_num === c.primary_page);
  if (idx >= 0) renderPage(idx);
  const vi = document.getElementById('viewer-chunk-info');
  if (vi) vi.textContent =
    `Chunk #${{id}} — ${{c.time_tags.length ? c.time_tags.join('+') : 'no tags'}} — ${{c.char_count}} chars`;
  renderList();
}}

// ── Page rendering ────────────────────────────────────────────────
function renderPage(idx) {{
  if (idx < 0 || idx >= PAGES.length) return;
  curPageIdx = idx;
  const page  = PAGES[idx];
  const key   = `${{page.pdf}}::${{page.page_num}}`;
  const imgEl = document.getElementById('pdf-img');
  imgEl.src = `data:image/png;base64,${{IMG_MAP[key]}}`;
  imgEl.onload = renderOverlays;

  document.getElementById('v-pdf').textContent = page.pdf;
  const pdfPages = PAGES.filter(p => p.pdf === page.pdf);
  const posInPdf = pdfPages.findIndex(p => p.page_num === page.page_num) + 1;
  document.getElementById('page-label').textContent = `${{page.page_num+1}} / ${{pdfPages.length}}`;
}}

// ── Overlay rendering — KEY FIX ───────────────────────────────────
// Each page in PAGES carries its own sx/sy scale factors derived from
// the actual unstructured image dimensions vs the PyMuPDF render size.
// We multiply bbox coordinates by (sx * display_scale) to get final pixels.
function renderOverlays() {{
  const overlays = document.getElementById('overlays');
  overlays.innerHTML = '';

  const page  = PAGES[curPageIdx];
  const imgEl = document.getElementById('pdf-img');
  if (!imgEl.naturalWidth) return;

  const dispW  = imgEl.offsetWidth  || imgEl.naturalWidth;
  const dispH  = imgEl.offsetHeight || imgEl.naturalHeight;
  const natW   = imgEl.naturalWidth;
  const natH   = imgEl.naturalHeight;

  // sx/sy map unstructured px coords → PyMuPDF natural image pixels
  // Then we scale from natural to displayed (in case CSS resizes the image)
  const sx = page.sx * (dispW / natW);
  const sy = page.sy * (dispH / natH);

  // Debug bar
  document.getElementById('dbg-scale').textContent =
    `sx=${{page.sx.toFixed(4)}} sy=${{page.sy.toFixed(4)}}`;
  document.getElementById('dbg-imgsize').textContent =
    `disp ${{dispW}}×${{dispH}} nat ${{natW}}×${{natH}}`;
  document.getElementById('dbg-pdfsize').textContent =
    `pdf ${{page.pdf_width.toFixed(0)}}×${{page.pdf_height.toFixed(0)}}pts`;

  let hasBbox = false, noBboxCnt = 0, blockCount = 0;

  page.chunks.forEach(pc => {{
    const kept  = keepStatus[pc.chunk_id];
    const color = COLORS[pc.color_idx % COLORS.length];
    let firstBox = null;

    pc.blocks.forEach(block => {{
      if (!block.bbox) {{ noBboxCnt++; return; }}
      hasBbox = true;
      blockCount++;

      // block.bbox is in raw unstructured pixels
      const [bx0, by0, bx1, by1] = block.bbox;
      const left   = bx0 * sx;
      const top    = by0 * sy;
      const width  = (bx1 - bx0) * sx;
      const height = (by1 - by0) * sy;

      if (width < 2 || height < 2) return;

      const box = document.createElement('div');
      box.className = `h-box ${{selectedId === pc.chunk_id ? 'selected' : ''}} ${{!kept ? 'skipped' : ''}}`;
      box.style.cssText =
        `left:${{left.toFixed(1)}}px;top:${{top.toFixed(1)}}px;` +
        `width:${{width.toFixed(1)}}px;height:${{height.toFixed(1)}}px;` +
        `background:${{color}};`;
      box.title = `Chunk #${{pc.chunk_id}}`;
      box.addEventListener('click', () => selectChunk(pc.chunk_id));
      overlays.appendChild(box);

      if (!firstBox) firstBox = {{ left, top }};
    }});

    if (firstBox) {{
      const lbl = document.createElement('div');
      lbl.className = 'h-label';
      lbl.style.cssText =
        `left:${{(firstBox.left+2).toFixed(1)}}px;` +
        `top:${{Math.max(0,firstBox.top-14).toFixed(1)}}px;` +
        `background:${{color}};`;
      lbl.textContent = `#${{pc.chunk_id}}`;
      overlays.appendChild(lbl);
    }}
  }});

  document.getElementById('dbg-blocks').textContent = `blocks on page: ${{blockCount}}`;

  const wrapper = document.getElementById('canvas-wrapper');
  wrapper.style.width  = dispW + 'px';
  wrapper.style.height = dispH + 'px';

  if (!hasBbox && noBboxCnt > 0) {{
    const note = document.createElement('div');
    note.className   = 'no-bbox-note';
    note.textContent = `${{noBboxCnt}} block(s) have no bbox`;
    overlays.appendChild(note);
  }}
}}

window.addEventListener('resize', () => {{ if (curPageIdx >= 0) renderOverlays(); }});

// ── Page navigation ───────────────────────────────────────────────
function navPage(delta) {{
  const page    = PAGES[curPageIdx];
  const samePdf = PAGES.filter(p => p.pdf === page.pdf);
  const pos     = samePdf.findIndex(p => p.page_num === page.page_num);
  const nextPos = pos + delta;
  if (nextPos < 0 || nextPos >= samePdf.length) return;
  const nextPage  = samePdf[nextPos];
  const globalIdx = PAGES.findIndex(
    p => p.pdf === nextPage.pdf && p.page_num === nextPage.page_num
  );
  renderPage(globalIdx);
}}

// ── Export ────────────────────────────────────────────────────────
function exportKept() {{
  const kept = CHUNKS.filter(c => keepStatus[c.chunk_id]);
  const bySource = {{}};
  kept.forEach(c => {{
    if (!bySource[c.source]) bySource[c.source] = [];
    bySource[c.source].push(c.chunk_id);
  }});

  const payload = {{
    meta: {{
      total_chunks:  CHUNKS.length,
      kept_count:    kept.length,
      skipped_count: CHUNKS.length - kept.length,
      chunk_params: {{
        max_characters:        {CHUNK_MAX_CHARACTERS},
        new_after_n_chars:     {CHUNK_NEW_AFTER_N_CHARS},
        combine_under_n_chars: {CHUNK_COMBINE_UNDER_N_CHARS},
      }},
    }},
    kept_ids_by_source: bySource,
    kept_chunks: kept.map(c => ({{
      chunk_id:   c.chunk_id,
      source:     c.source,
      pages:      c.pages,
      char_count: c.char_count,
      time_tags:  c.time_tags,
      has_table:  c.has_table,
      has_image:  c.has_image,
      is_english: c.is_english,
      text:       c.text,
    }})),
  }};

  const fname = (SOURCES[0] || 'chunks').replace(/\\.pdf$/i,'') + '_kept.json';
  const blob  = new Blob([JSON.stringify(payload, null, 2)], {{type:'application/json'}});
  const a     = document.createElement('a');
  a.href      = URL.createObjectURL(blob);
  a.download  = fname;
  a.click();
}}

// ── Init ──────────────────────────────────────────────────────────
['q-search','f-source','f-time','f-status'].forEach(id =>
  document.getElementById(id).addEventListener('input', renderList)
);

renderList();
if (PAGES.length > 0) renderPage(0);
</script>
</body>
</html>"""


# ── Process one PDF ───────────────────────────────────────────────
def process_pdf(pdf_path: str) -> Dict[str, Any]:
    print(f"\n📄 {os.path.basename(pdf_path)}")

    elements, chunks = partition_and_chunk(pdf_path)

    # Render pages first to get PDF point dimensions
    pages_b64, page_sizes = render_pages(pdf_path)

    # Build scale map using actual unstructured coordinate metadata
    print(f"   Computing coordinate scale factors...", end=" ", flush=True)
    scale_map = build_page_scale_map(elements, page_sizes)
    print(f"done ({len(scale_map)} pages)")

    # Log scale factors for debugging
    for pg, sm in sorted(scale_map.items()):
        print(f"   Page {pg+1}: sx={sm['sx']:.4f}  sy={sm['sy']:.4f}  "
              f"(pymupdf {sm['pymupdf_w']:.0f}×{sm['pymupdf_h']:.0f}px)")

    chunk_objs = build_chunk_objects(chunks, pdf_path)

    kept   = sum(1 for c in chunk_objs if c["keep"])
    noen   = sum(1 for c in chunk_objs if not c["is_english"])
    tables = sum(1 for c in chunk_objs if c["has_table"])
    print(f"   Summary: {len(chunk_objs)} chunks | {kept} auto-kept | "
          f"{noen} non-English | {tables} with tables")

    return {
        "pdf_name":   os.path.basename(pdf_path),
        "pdf_path":   pdf_path,
        "pages_b64":  pages_b64,
        "page_sizes": page_sizes,
        "scale_map":  scale_map,
        "chunks":     chunk_objs,
    }


def run(pdf_paths: List[str], output_path: str):
    print("🚀 VerdaSense Chunk Selector (coordinate-fixed edition)")
    print(f"   PDFs    : {len(pdf_paths)}")
    print(f"   Output  : {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf_data = [process_pdf(p) for p in pdf_paths]

    total = sum(len(d["chunks"]) for d in pdf_data)
    print(f"\n🔮 Building HTML ({total} total chunks)...", end=" ", flush=True)
    html = build_html(pdf_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"done")
    print(f"✅ {output_path}  ({size_mb:.1f} MB)")
    print(f"   Open: file://{os.path.abspath(output_path)}")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise unstructured.io chunks on PDF pages for RAG curation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Single PDF file")
    group.add_argument("--dir", help="Directory of PDFs")
    parser.add_argument("--out", default="chunk_selector.html", help="Output HTML path")
    args = parser.parse_args()

    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(".pdf")
        ])
        if not pdfs:
            sys.exit(f"❌  No PDFs found in: {args.dir}")

    run(pdfs, args.out)
