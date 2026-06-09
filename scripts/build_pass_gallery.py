#!/usr/bin/env python3
"""Build a self-contained HTML gallery of gate-E PASSED edits.

For every edit whose stages.gate_e.status == "pass", collect:
  condition = edits_2d/<eid>_edited.png   (FLUX 2D edit that drives the 3D edit)
  before    = gate_views/before_view_<view>.png
  after     = edits_3d/<eid>/after_view_<view>.png
  prompt    = refined_prompt.json (improved/original) + gate_e_judge.json details

Image paths are stored RELATIVE to the prod root, and the HTML is written into
that same root, so it opens straight from file:// with working <img src>.

Usage:
  python scripts/build_pass_gallery.py \
      --root data/Pxform_v2/prod_posthoc_no2dqc \
      --shards 00 01 02 03 04 05 \
      --per-shard 200 --view front \
      --out data/Pxform_v2/prod_posthoc_no2dqc/pass_gallery.html
"""
import argparse, json, glob, os, html, base64, io

VIEWS = ["front", "left", "right", "back", "down"]
# Canonical column→name order used across the pipeline (partcraft/render/ovox_views.py).
# gate_a's best_view is an absolute index into THIS list; the FLUX condition image is
# rendered from that view, so before/after must use the same name to stay aligned.
VIEW_ORDER = ["front", "right", "back", "left", "down"]

_ENC_CACHE = {}


def to_data_uri(path, thumb):
    """Encode an image to a base64 data URI, optionally downscaled to `thumb` px."""
    if not path or not os.path.exists(path):
        return ""
    key = (path, thumb)
    if key in _ENC_CACHE:
        return _ENC_CACHE[key]
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if thumb:
            im.thumbnail((thumb, thumb), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as fh:
            ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
            uri = f"data:image/{ext};base64," + base64.b64encode(fh.read()).decode()
    _ENC_CACHE[key] = uri
    return uri


def first_existing(objdir, names):
    for n in names:
        if os.path.exists(os.path.join(objdir, n)):
            return n
    return None


def collect(root, shards, per_shard, view, embed=False, thumb=384):
    cards, stats = [], {}
    view_order = [view] + [v for v in VIEWS if v != view]
    for s in shards:
        n_shard = 0
        for st in sorted(glob.glob(f"{root}/objects/{s}/*/edit_status.json")):
            try:
                d = json.load(open(st))
            except Exception:
                continue
            objdir = os.path.dirname(st)
            obj = d.get("obj_id", os.path.basename(objdir))
            for eid, it in d.get("edits", {}).items():
                stg = it.get("stages") or {}
                ge = stg.get("gate_e") or stg.get("gate_quality") or {}
                if ge.get("status") != "pass":
                    continue
                edir = os.path.join(objdir, "edits_3d", eid)

                # condition viewpoint = the view FLUX edited = VIEW_ORDER[gate_a.best_view].
                # before/after must use the SAME named view so all three columns align.
                cview = None
                try:
                    bv = stg.get("gate_a", {}).get("verdict", {}).get("vlm", {}).get("best_view")
                    if bv is not None and 0 <= int(bv) < len(VIEW_ORDER):
                        cview = VIEW_ORDER[int(bv)]
                except Exception:
                    pass
                # preference: matched condition view first, then requested view, then any
                pref = [v for v in [cview, view] if v] + [v for v in view_order if v not in (cview, view)]

                cond_rel = first_existing(
                    objdir, [f"edits_2d/{eid}_edited.png", f"edits_2d/{eid}_input.png"])
                # pick ONE view that has BOTH a before and an after render, in preference order
                bview = aview = picked = None
                for v in pref:
                    bp = f"gate_views/before_view_{v}.png"
                    ap = f"after_view_{v}.png"
                    if os.path.exists(os.path.join(objdir, bp)) and os.path.exists(os.path.join(edir, ap)):
                        bview, aview, picked = bp, ap, v
                        break
                if not (bview and aview):
                    continue

                prompt = orig = desc = ""
                vq = None
                rp = os.path.join(edir, "refined_prompt.json")
                if os.path.exists(rp):
                    try:
                        p = json.load(open(rp))
                        prompt = p.get("improved_prompt", "")
                        orig = p.get("original_prompt", "")
                    except Exception:
                        pass
                jp = os.path.join(edir, "gate_e_judge.json")
                if os.path.exists(jp):
                    try:
                        j = json.load(open(jp)).get("judge", {})
                        vq = j.get("visual_quality")
                        desc = j.get("reason", "")
                    except Exception:
                        pass

                relbase = os.path.relpath(objdir, root)
                cond_p = os.path.join(objdir, cond_rel) if cond_rel else ""
                before_p = os.path.join(objdir, bview)
                after_p = os.path.join(edir, aview)
                if embed:
                    cond = to_data_uri(cond_p, thumb)
                    before = to_data_uri(before_p, thumb)
                    after = to_data_uri(after_p, thumb)
                else:
                    cond = (relbase + "/" + cond_rel) if cond_rel else ""
                    before = relbase + "/" + bview
                    after = os.path.relpath(after_p, root)
                cards.append({
                    "shard": s, "obj": obj, "eid": eid,
                    "type": it.get("edit_type", "?"),
                    "vq": vq, "view": picked, "matched": (picked == cview),
                    "prompt": prompt or orig, "orig": orig, "desc": desc,
                    "cond": cond, "before": before, "after": after,
                })
                n_shard += 1
                if per_shard and n_shard >= per_shard:
                    break
            if per_shard and n_shard >= per_shard:
                break
        stats[s] = n_shard
    return cards, stats


HTML_TMPL = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Gate-E Pass Gallery — {title}</title>
<style>
 :root{{--bg:#11131a;--card:#1b1e27;--mut:#8b93a7;--fg:#e8ebf2;--ac:#5b9dff}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif}}
 header{{position:sticky;top:0;z-index:9;background:#0d0f15ee;backdrop-filter:blur(6px);
   padding:12px 18px;border-bottom:1px solid #262a36;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
 h1{{font-size:16px;margin:0 8px 0 0}} .muted{{color:var(--mut)}}
 .chip{{cursor:pointer;border:1px solid #313747;border-radius:14px;padding:4px 11px;color:var(--fg);
   background:#171a22;font-size:13px}} .chip.on{{background:var(--ac);border-color:var(--ac);color:#06101f;font-weight:600}}
 input{{background:#171a22;border:1px solid #313747;border-radius:8px;color:var(--fg);padding:6px 10px;min-width:240px}}
 #grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px;padding:18px}}
 .card{{background:var(--card);border:1px solid #262a36;border-radius:12px;overflow:hidden}}
 .row3{{display:grid;grid-template-columns:1fr 1fr 1fr}}
 .cell{{position:relative;aspect-ratio:1/1;background:#0c0e13;border-right:1px solid #0c0e13}}
 .cell:last-child{{border-right:0}} .cell img{{width:100%;height:100%;object-fit:contain;display:block}}
 .cell .lbl{{position:absolute;top:5px;left:5px;font-size:10px;letter-spacing:.04em;background:#0009;
   padding:1px 6px;border-radius:6px;color:#cfd6e6}}
 .meta{{padding:10px 12px}} .tags{{display:flex;gap:6px;align-items:center;margin-bottom:6px;flex-wrap:wrap}}
 .tag{{font-size:11px;padding:1px 7px;border-radius:6px;background:#222736;color:#aeb7cc}}
 .tag.t{{background:#1d2f23;color:#7fdca0}} .tag.vq{{background:#2c2540;color:#c3a6ff}}
 .prompt{{font-size:13.5px;color:#eef1f8}} .orig{{font-size:12px;color:var(--mut);margin-top:3px}}
 .desc{{font-size:11.5px;color:#7e879b;margin-top:6px;display:none}} .card.exp .desc{{display:block}}
 .id{{font-size:10.5px;color:#5a6379;margin-top:6px;font-family:ui-monospace,Menlo,monospace;word-break:break-all}}
 #more{{display:block;margin:20px auto 40px;padding:10px 22px;border-radius:10px;border:1px solid #313747;
   background:#171a22;color:var(--fg);cursor:pointer;font-size:14px}}
</style></head><body>
<header>
 <h1>Gate-E Pass Gallery</h1>
 <span class="muted" id="count"></span>
 <span id="shards"></span>
 <span id="types"></span>
 <input id="q" placeholder="搜索 prompt / id …">
 <label class="muted" style="cursor:pointer"><input type="checkbox" id="exp" style="min-width:auto"> 显示判定说明</label>
</header>
<div id="grid"></div>
<button id="more">加载更多</button>
<script>
const DATA = {data};
const PAGE = 60; let shown = 0, fShard = "all", fType = "all", q = "";
const grid = document.getElementById("grid");
const shardSet = [...new Set(DATA.map(d=>d.shard))].sort();
const typeSet  = [...new Set(DATA.map(d=>d.type))].sort();
function chips(host, vals, get, set){{
  const all = document.createElement("span"); all.className="chip on"; all.textContent="全部";
  host.appendChild(all);
  const els=[all];
  vals.forEach(v=>{{const c=document.createElement("span");c.className="chip";c.textContent=v;host.appendChild(c);els.push(c);
    c.onclick=()=>{{set(v);els.forEach(e=>e.classList.remove("on"));c.classList.add("on");render(true);}};}});
  all.onclick=()=>{{set("all");els.forEach(e=>e.classList.remove("on"));all.classList.add("on");render(true);}};
}}
chips(document.getElementById("shards"), shardSet, ()=>fShard, v=>fShard=v);
chips(document.getElementById("types"),  typeSet,  ()=>fType,  v=>fType=v);
document.getElementById("q").oninput=e=>{{q=e.target.value.toLowerCase();render(true);}};
document.getElementById("exp").onchange=e=>document.querySelectorAll(".card").forEach(c=>c.classList.toggle("exp",e.target.checked));
function match(d){{return (fShard==="all"||d.shard===fShard)&&(fType==="all"||d.type===fType)&&
  (!q|| (d.prompt+" "+d.eid+" "+d.obj).toLowerCase().includes(q));}}
function cell(src,lbl){{return src?`<div class="cell"><span class="lbl">${{lbl}}</span><img loading="lazy" src="${{src}}"></div>`
  :`<div class="cell"><span class="lbl">${{lbl}}</span></div>`;}}
let filtered=[];
function render(reset){{
  if(reset){{filtered=DATA.filter(match);shown=0;grid.innerHTML="";}}
  const next=filtered.slice(shown,shown+PAGE);const exp=document.getElementById("exp").checked;
  next.forEach(d=>{{const el=document.createElement("div");el.className="card"+(exp?" exp":"");
    el.innerHTML=`<div class="row3">${{cell(d.cond,"CONDITION")}}${{cell(d.before,"BEFORE")}}${{cell(d.after,"AFTER")}}</div>
      <div class="meta"><div class="tags"><span class="tag">shard ${{d.shard}}</span>
        <span class="tag t">${{d.type}}</span>${{d.vq!=null?`<span class="tag vq">VQ ${{d.vq}}</span>`:""}}
        ${{d.view?`<span class="tag" title="${{d.matched?'condition 视角':'回退视角（无 best_view）'}}">${{d.view}}${{d.matched?"":"*"}}</span>`:""}}</div>
      <div class="prompt">${{d.prompt||"<i class=muted>无 prompt</i>"}}</div>
      ${{d.orig&&d.orig!==d.prompt?`<div class="orig">orig: ${{d.orig}}</div>`:""}}
      <div class="desc">${{d.desc||""}}</div>
      <div class="id">${{d.eid}}</div></div>`;
    grid.appendChild(el);}});
  shown+=next.length;
  document.getElementById("count").textContent=`${{filtered.length}} passed · 已显示 ${{shown}}`;
  document.getElementById("more").style.display=shown<filtered.length?"block":"none";
}}
document.getElementById("more").onclick=()=>render(false);
render(true);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2/prod_posthoc_no2dqc")
    ap.add_argument("--shards", nargs="+", default=["00", "01", "02", "03", "04", "05"])
    ap.add_argument("--per-shard", type=int, default=200, help="0 = no cap")
    ap.add_argument("--view", default="front", choices=VIEWS)
    ap.add_argument("--embed", action="store_true",
                    help="inline images as base64 data URIs → single portable file")
    ap.add_argument("--thumb", type=int, default=384,
                    help="downscale embedded images to this many px (0 = full size)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "pass_gallery.html")

    cards, stats = collect(a.root, a.shards, a.per_shard, a.view,
                           embed=a.embed, thumb=a.thumb)
    print("per-shard collected (cap=%s):" % (a.per_shard or "none"))
    for s in a.shards:
        print(f"  shard {s}: {stats.get(s,0)}")
    print(f"  TOTAL cards: {len(cards)}")

    page = HTML_TMPL.format(
        title=os.path.basename(a.root),
        data=json.dumps(cards, ensure_ascii=False))
    with open(out, "w") as f:
        f.write(page)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
