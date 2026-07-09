"""
build_testset_viewer.py
=======================
Generates a single self-contained HTML viewer for wound_testset_v5.json.

  python ragas_testset/build_testset_viewer.py
    → writes ragas_testset/testset_viewer.html   (open in any browser, no server needed)

Re-run this whenever you update the testset (builder → json → this). Images are downscaled
+ base64-embedded so the HTML works fully offline (no CORS / no file:// fetch issues).
"""
import json, base64, io, html as _html
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
TESTSET  = ROOT / "ragas_testset" / "wound_testset_v5.json"
OUT      = ROOT / "ragas_testset" / "testset_viewer.html"
MAX_DIM  = 760     # downscale longest edge for embedding

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠ Pillow not found — images embedded at full size (HTML will be larger).")

cases = json.load(open(TESTSET, encoding="utf-8"))

embedded = 0
for c in cases:
    ref = c.get("image_ref")
    p = (ROOT / ref) if ref else None
    if p and p.exists():
        try:
            if HAS_PIL:
                im = Image.open(p).convert("RGB")
                im.thumbnail((MAX_DIM, MAX_DIM))
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
                c["_image_data"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            else:
                c["_image_data"] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            embedded += 1
        except Exception as e:
            c["_image_data"] = None; print(f"  image error {ref}: {e}")
    else:
        c["_image_data"] = None

data_json = json.dumps(cases, ensure_ascii=False).replace("</", "<\\/")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VerdaSense Testset v5 Viewer</title>
<style>
:root{--bg:#070d14;--surface:#0d1b2a;--surface2:#112236;--surface3:#0a1520;--border:#1e3a52;
--border2:#254d6b;--accent:#00c8ff;--accent2:#0084a8;--text:#e2eff8;--muted:#5b7b95;
--t:#ff7043;--i:#ef5350;--m:#42a5f5;--e:#66bb6a;--vlm:#b388ff;--g3:#ef4444;--g2:#f59e0b;--g1:#42a5f5;
--ok:#22c55e;--no:#5b7b95;--font:'Segoe UI',system-ui,sans-serif;--mono:'Consolas','DM Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);height:100vh;overflow:hidden;font-size:14px}
.topbar{display:flex;align-items:center;gap:14px;padding:10px 18px;background:var(--surface);border-bottom:1px solid var(--border);flex-wrap:wrap}
.logo{font-family:var(--mono);color:var(--accent);font-size:16px;letter-spacing:.06em;font-weight:600}
.logo b{color:var(--vlm)}
.count{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--border);border-radius:20px;padding:3px 10px}
.search{background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:7px 12px;font-size:13px;outline:none;min-width:200px}
.search:focus{border-color:var(--accent)}
.viewtog{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.viewtog button{background:var(--bg);color:var(--muted);border:none;padding:7px 14px;cursor:pointer;font-family:var(--mono);font-size:12px}
.viewtog button.on{background:rgba(0,200,255,.12);color:var(--accent)}
.catfilter{display:flex;gap:5px;flex-wrap:wrap}
.catchip{font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--border);color:var(--muted);cursor:pointer;user-select:none}
.catchip.on{border-color:var(--accent);color:var(--accent);background:rgba(0,200,255,.08)}
.shell{display:flex;height:calc(100vh - 53px)}
.sidebar{width:300px;flex-shrink:0;background:var(--surface3);border-right:1px solid var(--border);overflow-y:auto}
.sb-group{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding:12px 16px 6px;position:sticky;top:0;background:var(--surface3)}
.sb-item{padding:9px 16px;cursor:pointer;border-left:3px solid transparent;display:flex;align-items:center;gap:8px}
.sb-item:hover{background:var(--surface2)}
.sb-item.active{background:var(--surface2);border-left-color:var(--accent)}
.sb-item .cid{font-family:var(--mono);font-size:12px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sb-item.active .cid{color:var(--accent)}
.wtbadge{font-family:var(--mono);font-size:10px;width:24px;height:20px;display:flex;align-items:center;justify-content:center;border-radius:5px;background:var(--surface);border:1px solid var(--border2);color:var(--muted);flex-shrink:0}
.sb-item .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.main{flex:1;overflow-y:auto;padding:22px 28px}
.tableview{padding:18px 24px;overflow:auto;height:100%}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border:1px solid var(--border);padding:6px 9px;text-align:left;white-space:nowrap}
th{background:var(--surface);font-family:var(--mono);font-size:11px;color:var(--accent);position:sticky;top:0;cursor:pointer}
tr:hover td{background:var(--surface2)}
td.cid{font-family:var(--mono);color:var(--accent);cursor:pointer}
.yes{color:var(--ok);font-weight:600}.dim{color:var(--muted)}
/* detail */
.dh{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px}
.dh h1{font-family:var(--mono);font-size:20px;color:var(--accent)}
.chip{font-family:var(--mono);font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid var(--border2);color:var(--muted)}
.chip.cat{color:var(--vlm);border-color:var(--vlm)}
.chip.wt{color:var(--accent);border-color:var(--accent2)}
.chip.ab{color:var(--i);border-color:var(--i);background:rgba(239,83,80,.08)}
.chip.rf{color:var(--g2);border-color:var(--g2);background:rgba(245,158,11,.08)}
.chip.dep{color:var(--vlm);border-color:var(--vlm2)}
.sub{color:var(--muted);font-size:12px;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:340px 1fr;gap:20px;margin-bottom:20px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin-bottom:18px}
.card h3{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.woundimg{width:100%;border-radius:10px;border:1px solid var(--border);display:block}
.noimg{height:200px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;border:1px dashed var(--border);border-radius:10px}
.imgcap{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px;word-break:break-all}
.tbar{display:flex;height:12px;border-radius:5px;overflow:hidden;margin:8px 0;gap:2px}
.tseg{height:100%}
.kv{display:flex;gap:8px;padding:4px 0;font-size:13px;border-bottom:1px solid var(--surface2)}
.kv .k{color:var(--muted);width:120px;flex-shrink:0;font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.kv .v{color:var(--text)}
.notes{background:var(--surface2);border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.6;margin-top:8px;border-left:2px solid var(--m)}
/* reference markdown */
.md{font-size:14px;line-height:1.7}
.md h2{font-size:14px;color:var(--accent);margin:16px 0 6px;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:3px}
.md h2:first-child{margin-top:0}
.md ul,.md ol{padding-left:20px;margin:6px 0}.md li{margin-bottom:4px}
.md p{margin-bottom:8px}.md strong{color:#fff}
.cite{display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 4px;border-radius:9px;background:rgba(0,200,255,.14);border:1px solid var(--accent2);font-family:var(--mono);font-size:10px;color:var(--accent);vertical-align:super;cursor:pointer;margin:0 1px}
.cite:hover{background:rgba(0,200,255,.3)}
/* contexts */
.ctx{border:1px solid var(--border);border-radius:10px;margin-bottom:10px;overflow:hidden;background:var(--surface2)}
.ctx.hl{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.ctxh{display:flex;align-items:center;gap:8px;padding:9px 12px;cursor:pointer;flex-wrap:wrap}
.ctxh:hover{background:var(--surface)}
.rank{font-family:var(--mono);font-size:11px;width:34px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:6px;background:rgba(0,200,255,.12);border:1px solid var(--accent2);color:var(--accent);flex-shrink:0}
.grade{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:5px;font-weight:600}
.g3{background:rgba(239,68,68,.15);color:var(--g3);border:1px solid var(--g3)}
.g2{background:rgba(245,158,11,.15);color:var(--g2);border:1px solid var(--g2)}
.g1{background:rgba(66,165,245,.15);color:var(--g1);border:1px solid var(--g1)}
.abbr{font-family:var(--mono);font-size:10px;color:var(--text);border:1px solid var(--border2);border-radius:5px;padding:2px 7px}
.role{font-family:var(--mono);font-size:11px;color:var(--vlm)}
.why{font-size:12px;color:var(--muted);flex:1;min-width:140px}
.ctxbody{display:none;padding:0 14px 14px;font-size:12.5px;line-height:1.65;color:var(--muted);white-space:pre-wrap;border-top:1px solid var(--border)}
.ctxbody.open{display:block;padding-top:12px}
.cid-small{font-family:var(--mono);font-size:9px;color:var(--muted)}
.taglist{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:6px;background:var(--surface2);border:1px solid var(--border);color:var(--text)}
.tag.avoid{border-color:var(--i);color:#ffb4b4;background:rgba(239,83,80,.08)}
.tag.cond{border-color:var(--g2);color:#ffd591;background:rgba(245,158,11,.06)}
.prodtab{width:100%;font-size:12.5px;border-collapse:collapse}
.prodtab td{border:1px solid var(--border);padding:5px 9px}
.prodtab td:first-child{font-family:var(--mono);color:var(--accent);width:160px}
.freq{font-family:var(--mono);font-size:12px}
.empty{color:var(--muted);font-style:italic;font-size:12px}
</style></head><body>
<div class="topbar">
  <span class="logo">VerdaSense <b>Testset v5</b> Viewer</span>
  <span class="count" id="count"></span>
  <input class="search" id="search" placeholder="Search case_id / notes…">
  <div class="catfilter" id="catfilter"></div>
  <div class="viewtog"><button id="vCases" class="on" onclick="setView('cases')">Cases</button><button id="vTable" onclick="setView('table')">Table</button></div>
</div>
<div class="shell">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main"></div>
</div>
<div class="tableview" id="tableview" style="display:none"></div>

<script>
const CASES = __DATA__;
const CAT_COLORS={A:'#00c8ff',B:'#ef5350',C:'#f59e0b',D:'#b388ff',E:'#66bb6a',F:'#42a5f5',G:'#ff7043'};
let activeCat=null, activeId=null, view='cases', q='';

function esc(s){return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* minimal markdown + [S#] citations */
function md(text){
  const lines=(text||'').replace(/\r/g,'').split('\n');let h='',lt=null;
  const cl=()=>{if(lt){h+='</'+lt+'>';lt=null;}};
  const inl=t=>{t=esc(t).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
     .replace(/\[S(\d+)\](\[S\d+\])*/g,m=>m.replace(/\[S(\d+)\]/g,(x,n)=>'<span class="cite" onclick="hlCtx('+n+')">S'+n+'</span>'));
     return t;};
  for(let ln of lines){const t=ln.trim();
    if(!t){cl();continue;}
    let m;
    if(m=t.match(/^#{1,6}\s+(.*)$/)){cl();h+='<h2>'+inl(m[1])+'</h2>';continue;}
    if(m=t.match(/^[-*]\s+(.*)$/)){if(lt!=='ul'){cl();h+='<ul>';lt='ul';}h+='<li>'+inl(m[1])+'</li>';continue;}
    if(m=t.match(/^\d+\.\s+(.*)$/)){if(lt!=='ol'){cl();h+='<ol>';lt='ol';}h+='<li>'+inl(m[1])+'</li>';continue;}
    cl();h+='<p>'+inl(t)+'</p>';}
  cl();return h;
}
function hlCtx(n){document.querySelectorAll('.ctx').forEach(c=>c.classList.remove('hl'));
  const el=document.getElementById('ctx-'+n);if(el){el.classList.add('hl');el.scrollIntoView({behavior:'smooth',block:'center'});
  const b=el.querySelector('.ctxbody');if(b&&!b.classList.contains('open'))b.classList.add('open');}}

function filtered(){return CASES.filter(c=>{
  if(activeCat&&c.category!==activeCat)return false;
  if(q){const s=(c.case_id+' '+(c.time_payload&&c.time_payload.notes||'')).toLowerCase();if(!s.includes(q))return false;}
  return true;});}

function buildCatFilter(){
  const cats=[...new Set(CASES.map(c=>c.category))].sort();
  const f=document.getElementById('catfilter');f.innerHTML='';
  const all=document.createElement('span');all.className='catchip'+(activeCat?'':' on');all.textContent='All';
  all.onclick=()=>{activeCat=null;render();};f.appendChild(all);
  cats.forEach(cat=>{const n=CASES.filter(c=>c.category===cat).length;
    const ch=document.createElement('span');ch.className='catchip'+(activeCat===cat?' on':'');
    ch.textContent=cat+' ('+n+')';ch.style.borderColor=activeCat===cat?CAT_COLORS[cat]:'';
    ch.onclick=()=>{activeCat=activeCat===cat?null:cat;render();};f.appendChild(ch);});}

function buildSidebar(){
  const sb=document.getElementById('sidebar');sb.innerHTML='';
  const list=filtered();const byCat={};list.forEach(c=>{(byCat[c.category]=byCat[c.category]||[]).push(c);});
  Object.keys(byCat).sort().forEach(cat=>{
    const g=document.createElement('div');g.className='sb-group';g.textContent='Category '+cat;sb.appendChild(g);
    byCat[cat].forEach(c=>{const it=document.createElement('div');it.className='sb-item'+(c.case_id===activeId?' active':'');
      const inf=(c.time_payload&&c.time_payload.infection||'').toLowerCase().includes('infect')&&!(c.time_payload.infection||'').toLowerCase().includes('not');
      it.innerHTML='<span class="wtbadge">'+(c.wound_type_expected==null?'–':c.wound_type_expected)+'</span>'+
        '<span class="cid">'+esc(c.case_id)+'</span>'+
        '<span class="dot" style="background:'+(inf?'var(--i)':'var(--ok)')+'"></span>';
      it.onclick=()=>{activeId=c.case_id;render();};sb.appendChild(it);});});
  if(!list.length)sb.innerHTML='<div class="empty" style="padding:16px">No cases match.</div>';}

function tbar(tp){const n=tp.necrotic_pct||0,s=tp.slough_pct||0,g=tp.granulation_pct||0;
  return '<div class="tbar">'+(n>0?'<div class="tseg" style="flex:'+n+';background:#555"></div>':'')+
    (s>0?'<div class="tseg" style="flex:'+s+';background:#ffd93d"></div>':'')+
    (g>0?'<div class="tseg" style="flex:'+g+';background:#6bcb77"></div>':'')+'</div>'+
    '<div class="cid-small">Necrotic '+n+'% · Slough '+s+'% · Granulation '+g+'%</div>';}

function renderDetail(c){
  const tp=c.time_payload||{};const main=document.getElementById('main');
  const meta=c.reference_contexts_meta||[];const ctxs=c.reference_contexts||[];
  let ctxHtml='';meta.forEach((m,i)=>{const gr=m.grade;
    ctxHtml+='<div class="ctx" id="ctx-'+m.rank+'"><div class="ctxh" onclick="this.parentNode.querySelector(\'.ctxbody\').classList.toggle(\'open\')">'+
      '<span class="rank">S'+m.rank+'</span>'+
      '<span class="grade g'+gr+'">grade '+gr+'</span>'+
      '<span class="abbr">'+esc(m.abbrev)+'</span>'+
      '<span class="role">'+esc(m.role||'')+'</span>'+
      '<span class="why">'+esc(m.why||'')+'</span>'+
      '<span class="cid-small">'+esc(m.chunk_id||'')+'</span></div>'+
      '<div class="ctxbody">'+esc(ctxs[i]||'(text not stored)')+'</div></div>';});

  const ep=c.example_products||{};let prodRows=Object.keys(ep).map(k=>'<tr><td>'+esc(k)+'</td><td>'+esc(ep[k])+'</td></tr>').join('');
  const cf=c.expected_change_frequency||{};let freqRows=Object.keys(cf).map(k=>'<tr><td>'+esc(k)+'</td><td class="freq">'+esc(cf[k])+'</td></tr>').join('');
  const tags=(arr,cls)=>(arr&&arr.length)?'<div class="taglist">'+arr.map(x=>'<span class="tag '+cls+'">'+esc(x)+'</span>').join('')+'</div>':'<span class="empty">none</span>';

  main.innerHTML=
  '<div class="dh"><h1>'+esc(c.case_id)+'</h1>'+
    '<span class="chip cat">Cat '+esc(c.category)+'</span>'+
    '<span class="chip wt">WT '+(c.wound_type_expected==null?'–':c.wound_type_expected)+'</span>'+
    (c.antibiotic_required?'<span class="chip ab">antibiotic</span>':'')+
    (c.referral_required?'<span class="chip rf">referral</span>':'')+
    '<span class="chip dep">depth: '+esc(c.wound_depth||'superficial')+'</span></div>'+
  '<div class="sub">'+esc(c.user_input||'').split('\n').join(' · ')+'</div>'+

  '<div class="grid2">'+
    '<div class="card"><h3>🩹 Wound image</h3>'+
      (c._image_data?'<img class="woundimg" src="'+c._image_data+'">':'<div class="noimg">no image_ref</div>')+
      '<div class="imgcap">'+esc(c.image_ref||'—')+'</div></div>'+
    '<div class="card"><h3>📋 T.I.M.E. payload + context</h3>'+
      tbar(tp)+
      '<div class="kv"><span class="k">Infection</span><span class="v">'+esc(tp.infection)+'</span></div>'+
      '<div class="kv"><span class="k">Moisture</span><span class="v">'+esc(tp.moisture)+'</span></div>'+
      '<div class="kv"><span class="k">Edge</span><span class="v">'+esc(tp.edge)+'</span></div>'+
      '<div class="kv"><span class="k">Diabetic</span><span class="v">'+((c.demographics||{}).diabetic?'Yes':'No')+'</span></div>'+
      '<div class="kv"><span class="k">Wound type</span><span class="v">'+esc(c.wound_type_expected)+'</span></div>'+
      '<div class="kv"><span class="k">Antibiotic</span><span class="v">'+(c.antibiotic_required?'<span class=yes>Required</span>':'<span class=dim>No</span>')+'</span></div>'+
      '<div class="kv"><span class="k">Referral</span><span class="v">'+(c.referral_required?'<span class=yes>Required</span>':'<span class=dim>No</span>')+'</span></div>'+
      (tp.notes?'<div class="notes">📝 '+esc(tp.notes)+'</div>':'')+
    '</div>'+
  '</div>'+

  '<div class="card"><h3>✅ Gold reference (patient-friendly answer)</h3><div class="md">'+md(c.reference)+'</div></div>'+

  '<div class="card"><h3>📚 Reference contexts — ranked &amp; graded ('+meta.length+')  ·  click [S#] above to highlight</h3>'+
    (ctxHtml||'<span class="empty">none</span>')+'</div>'+

  '<div class="card"><h3>💊 Dressings</h3>'+
    '<div class="kv"><span class="k">Allowed</span><span class="v">'+tags(c.allowed_dressings,'')+'</span></div>'+
    '<div class="kv"><span class="k">Contraindicated</span><span class="v">'+tags(c.contraindicated_dressings,'avoid')+'</span></div>'+
    '<div class="kv"><span class="k">Conditional</span><span class="v">'+tags(c.conditional_contraindications,'cond')+'</span></div>'+
    (prodRows?'<h3 style="margin-top:14px">🛍 Example products</h3><table class="prodtab">'+prodRows+'</table>':'')+
    (freqRows?'<h3 style="margin-top:14px">⏱ Change frequency</h3><table class="prodtab">'+freqRows+'</table>':'')+
    '<div class="kv" style="margin-top:12px"><span class="k">Escalation</span><span class="v">'+tags(c.escalation_flags_expected,'')+'</span></div>'+
  '</div>';
}

function renderTable(){
  const tv=document.getElementById('tableview');const list=filtered();
  let h='<table><thead><tr>'+['case_id','cat','WT','N/S/G','infection','moisture','edge','abx','referral','depth','#ctx','img'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
  list.forEach(c=>{const tp=c.time_payload||{};
    h+='<tr><td class="cid" onclick="activeId=\''+c.case_id+'\';setView(\'cases\')">'+esc(c.case_id)+'</td>'+
      '<td>'+esc(c.category)+'</td><td>'+(c.wound_type_expected==null?'–':c.wound_type_expected)+'</td>'+
      '<td>'+(tp.necrotic_pct||0)+'/'+(tp.slough_pct||0)+'/'+(tp.granulation_pct||0)+'</td>'+
      '<td>'+esc(tp.infection)+'</td><td>'+esc(tp.moisture)+'</td><td>'+esc(tp.edge)+'</td>'+
      '<td>'+(c.antibiotic_required?'<span class=yes>Y</span>':'<span class=dim>·</span>')+'</td>'+
      '<td>'+(c.referral_required?'<span class=yes>Y</span>':'<span class=dim>·</span>')+'</td>'+
      '<td>'+esc(c.wound_depth||'superficial')+'</td>'+
      '<td>'+((c.reference_contexts_meta||[]).length)+'</td>'+
      '<td>'+(c._image_data?'<span class=yes>✓</span>':'<span class=dim>·</span>')+'</td></tr>';});
  h+='</tbody></table>';tv.innerHTML=h;}

function setView(v){view=v;
  document.getElementById('vCases').classList.toggle('on',v==='cases');
  document.getElementById('vTable').classList.toggle('on',v==='table');
  document.querySelector('.shell').style.display=v==='cases'?'flex':'none';
  document.getElementById('tableview').style.display=v==='table'?'block':'none';
  render();}

function render(){
  document.getElementById('count').textContent=filtered().length+' / '+CASES.length+' cases';
  buildCatFilter();
  if(view==='cases'){
    buildSidebar();
    const list=filtered();
    if(!list.find(c=>c.case_id===activeId))activeId=list.length?list[0].case_id:null;
    const c=CASES.find(x=>x.case_id===activeId);
    if(c)renderDetail(c);else document.getElementById('main').innerHTML='<div class="empty" style="padding:30px">No case selected.</div>';
  }else renderTable();}

document.getElementById('search').addEventListener('input',e=>{q=e.target.value.toLowerCase().trim();render();});
render();
</script></body></html>"""

OUT.write_text(TEMPLATE.replace("__DATA__", data_json), encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"[OK] Wrote {OUT.relative_to(ROOT)}  ({len(cases)} cases, {embedded} images embedded, {size_mb:.1f} MB)")
print(f"     Open it in any browser (double-click) - fully offline, no server needed.")
