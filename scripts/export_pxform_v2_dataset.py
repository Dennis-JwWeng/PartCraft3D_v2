#!/usr/bin/env python3
"""Export an H3D_v1-style dataset ("Pxform_v2", export build) from the prod tree.

All 4 edit types now live in the prod object tree as TRELLIS.2 latents @512
(32³ slat / 16³ ss): mod/scale from the FLUX 3D-edit stage, del/add from the
``del_add`` pipeline stage (partcraft/pipeline_v3/del_add_reencode.py).  This
script is therefore a **pure CPU copy** — it reads
``edits_3d/<eid>/latents/{shape_slat,tex_slat,ss}.npz`` verbatim, derives the
mask sidecar from ss.npz, and lays everything out under ``data/Pxform_v2/export/``:

  <edit_type>/<shard>/<obj>/<edit_id>/
      shape_slat.npz  tex_slat.npz  ss.npz   (copied from prod, 32³)
      mask.npz        mask_keep_ss[16,16,16] + mask_keep_slat[N_after]
                      + mask_keep_slat_before[N_before] + selected_part_ids
      before.png / after.png / meta.json / view.meta.json
  _assets/<shard>/<obj>/   shape_slat.npz(e512) + tex_slat.npz(e512) + ss.npz(1,8,16,16,16)
  manifests/all.<tag>.jsonl

Edit selection + provenance:
  mod/scale : gate_e == pass; best_view from gate_a (VLM); before = gate_views.
  del/add   : del_add stage status == done; best_view from stages.del_add.verdict
              (overview-pixel argmax); deletion before = gate_views/before_view,
              addition before = the PAIRED deletion's after_view (the deleted
              render) and after = its own after_view (the restored original).

Requires the ``del_add`` stage to have run for the target shard (otherwise the
del/add candidate lists are empty).  No GPU, no model load.
"""
import argparse, json, shutil, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("export_pxform_v2")

VIEW_ORDER = ["front", "right", "back", "left", "down"]
_LIMIT_KEY = {"deletion": "del", "addition": "add",
              "modification": "modification", "scale": "scale"}


# ───────────────────────── mask (derived from ss.npz) ─────────────────────────
def mask_from_ss(ss):
    """v1-style keep masks from an ss dict (coords0, coords_new, edit_grid, keep16, parts).
    shape and texture share one coordinate mask (verified)."""
    eg = {tuple(int(x) for x in c) for c in np.asarray(ss["edit_grid"]).tolist()}
    def keep(coords):
        return np.fromiter((tuple(int(x) for x in c) not in eg
                            for c in np.asarray(coords).tolist()), dtype=np.uint8,
                           count=len(coords))
    return {
        "mask_keep_ss": np.asarray(ss["keep16"]).astype(np.uint8),
        "mask_keep_slat": keep(ss["coords_new"]),
        "mask_keep_slat_before": keep(ss["coords0"]),
        "selected_part_ids": np.asarray(ss["parts"], dtype=np.int32),
    }


# ───────────────────────── io helpers ─────────────────────────
def _save(path, d):
    np.savez(path, **d)


def _put_png(src, dst):
    if src is None:
        return
    if Path(src).is_file():
        shutil.copy2(src, dst)


def assets_before(out_root, shard, obj, p1_dir):
    """Shared 'before' (original) at 32³: e512 shape/tex + p1 ss latent. Atomic, deduped."""
    import os
    ad = out_root / "_assets" / shard / obj
    ad.mkdir(parents=True, exist_ok=True)
    for name, src in (("shape_slat.npz", "shape_slat_e512.npz"),
                      ("tex_slat.npz", "tex_slat_e512.npz"),
                      ("ss.npz", "ss.npz")):
        dst = ad / name
        if not dst.exists():
            tmp = ad / f".{name}.{os.getpid()}.tmp"
            shutil.copy2(p1_dir / src, tmp)
            os.replace(tmp, dst)
    return ad


def manifest_line(edit_id, edit_type, shard, obj, instruction, quality, bv):
    return json.dumps({
        "edit_id": edit_id, "edit_type": edit_type, "instruction": instruction,
        "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                    "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
        "obj_id": obj, "quality": quality, "schema_version": 3, "shard": shard,
        "views": {"best_view_index": bv}}, ensure_ascii=False)


# ───────────────────────── status / parsed helpers ─────────────────────────
def _seq(eid): return int(eid.rsplit("_", 1)[-1])
def _gate_e(st): return (st.get("gate_e") or st.get("gate_quality") or {}).get("status")
def _gate_a_score(st):
    try: return float(st["gate_a"]["verdict"]["vlm"]["score"])
    except Exception: return None
def _best_view(st):
    try: return int(st["gate_a"]["verdict"]["vlm"]["best_view"])
    except Exception: return 0
def _del_add_verdict(st):
    return ((st.get("del_add") or {}).get("verdict") or {})


def load_parsed(prod_obj_dir):
    d = json.loads((prod_obj_dir / "phase1" / "parsed.json").read_text())
    p = d.get("parsed") or {}
    edits = p.get("edits") or []
    obj_desc = (p.get("object") or {}).get("full_desc", "")
    dels = [e for e in edits if e.get("edit_type") == "deletion"]
    non_dels = [e for e in edits if e.get("edit_type") != "deletion"]
    return obj_desc, dels, non_dels


def _instruction(edit, obj_desc):
    out = {"object_desc": obj_desc}
    for k in ("prompt", "target_part_desc", "after_desc", "edit_params",
              "new_parts_desc", "new_part_desc"):
        if edit.get(k) not in (None, "", {}):
            out[k] = edit[k]
    return out


def _addition_instruction(prod_obj_dir, add_id, dels, obj_desc):
    """Addition prompt = the inverse of its paired deletion.  Prefer the add
    meta.json the del_add stage wrote; fall back to inverting the parsed del."""
    mp = prod_obj_dir / "edits_3d" / add_id / "meta.json"
    if mp.is_file():
        m = json.loads(mp.read_text())
        out = {"object_desc": m.get("object_desc") or obj_desc, "synthesized": True}
        if m.get("prompt"): out["prompt"] = m["prompt"]
        if m.get("target_part_desc"): out["target_part_desc"] = m["target_part_desc"]
        return out
    from partcraft.pipeline_v3.addition_utils import invert_delete_prompt
    spec = dels[_seq(add_id)] if _seq(add_id) < len(dels) else {}
    return {"object_desc": obj_desc, "synthesized": True,
            "prompt": invert_delete_prompt(spec.get("prompt", "")),
            "target_part_desc": spec.get("target_part_desc", "")}


# ───────────────────────── exporter (all 4 types, CPU copy from prod) ─────────────────────────
def export_type_from_prod(out_root, prod_obj_dir, p1_dir, shard, obj, edit_type, limit):
    lk = _LIMIT_KEY[edit_type]
    is_da = edit_type in ("deletion", "addition")
    status = json.loads((prod_obj_dir / "edit_status.json").read_text()).get("edits", {})
    obj_desc, dels, non_dels = load_parsed(prod_obj_dir)
    assets_before(out_root, shard, obj, p1_dir)

    def _ok(st):
        return ((st.get("del_add") or {}).get("status") == "done") if is_da else (_gate_e(st) == "pass")
    cands = sorted([k for k, v in status.items()
                    if v.get("edit_type") == edit_type and _ok(v.get("stages") or {})], key=_seq)

    lines = []
    for eid in cands:
        if limit[lk] <= 0:
            break
        lat = prod_obj_dir / "edits_3d" / eid / "latents"
        if not all((lat / f).is_file() for f in ("shape_slat.npz", "tex_slat.npz", "ss.npz")):
            continue
        st = status[eid].get("stages") or {}
        ss = dict(np.load(lat / "ss.npz", allow_pickle=True))
        mask = mask_from_ss(ss)

        # best_view + provenance
        if is_da:
            bv = int(_del_add_verdict(st).get("best_view") or 0)
            view_source = "overview_part_pixel_argmax"
        else:
            bv = _best_view(st); view_source = "pipeline_gate_a"
        vname = VIEW_ORDER[bv] if 0 <= bv < len(VIEW_ORDER) else "front"

        # after = own after_view; before depends on type
        ap = prod_obj_dir / "edits_3d" / eid / f"after_view_{vname}.png"
        if edit_type == "addition":
            paired = _del_add_verdict(st).get("paired_deletion_edit_id")
            bp = (prod_obj_dir / "edits_3d" / paired / f"after_view_{vname}.png") if paired else None
        else:
            bp = prod_obj_dir / "gate_views" / f"before_view_{vname}.png"

        # instruction
        if edit_type == "deletion":
            instr = _instruction(dels[_seq(eid)] if _seq(eid) < len(dels) else {}, obj_desc)
        elif edit_type == "addition":
            instr = _addition_instruction(prod_obj_dir, eid, dels, obj_desc)
        else:
            instr = _instruction(non_dels[_seq(eid)] if _seq(eid) < len(non_dels) else {}, obj_desc)

        # quality
        if is_da:
            quality = {"alignment_score": 1.0, "final_pass": False, "quality_score": 0.0}
        else:
            ga = _gate_a_score(st); qscore = 0.0
            try:
                vq = json.loads((prod_obj_dir / "edits_3d" / eid / "gate_e_judge.json").read_text()
                                ).get("judge", {}).get("visual_quality")
                qscore = round(vq / 5.0, 3) if vq is not None else 0.0
            except Exception:
                pass
            quality = {"alignment_score": ga if ga is not None else 1.0,
                       "final_pass": True, "quality_score": qscore}

        # write
        ed = out_root / edit_type / shard / obj / eid
        ed.mkdir(parents=True, exist_ok=True)
        for f in ("shape_slat.npz", "tex_slat.npz", "ss.npz"):
            shutil.copy2(lat / f, ed / f)
        _save(ed / "mask.npz", mask)
        _put_png(bp, ed / "before.png")
        _put_png(ap, ed / "after.png")
        meta = {"edit_id": eid, "edit_type": edit_type, "instruction": instr,
                "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                            "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
                "obj_id": obj, "quality": quality, "schema_version": 3, "shard": shard,
                "views": {"best_view_index": bv}}
        vmeta = {"edit_id": eid, "edit_type": edit_type, "shard": shard, "obj_id": obj,
                 "best_view_index": bv, "view_name": vname, "view_source": view_source}
        (ed / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (ed / "view.meta.json").write_text(json.dumps(vmeta, ensure_ascii=False, indent=2))
        lines.append(manifest_line(eid, edit_type, shard, obj, instr, quality, bv))
        limit[lk] -= 1
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-root", default="data/Pxform_v2/prod_posthoc_no2dqc/objects")
    ap.add_argument("--shard", default="00")
    ap.add_argument("--out", default="data/Pxform_v2/export")
    ap.add_argument("--n-del", type=int, default=250)
    ap.add_argument("--n-add", type=int, default=250)
    ap.add_argument("--n-mod", type=int, default=250)
    ap.add_argument("--n-scale", type=int, default=250)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--types", default="del,add,mod,scale")
    ap.add_argument("--obj-shard", "--gpu-shard", dest="obj_shard", default="0/1",
                    help="object partition i/n for parallel CPU runs")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    types = {t.strip() for t in a.types.split(",") if t.strip()}
    gk, gn = (int(x) for x in a.obj_shard.split("/"))
    out_root = Path(a.out); (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    objs = sorted([p.name for p in (Path(a.prod_root) / a.shard).iterdir() if p.is_dir()])
    objs = [o for i, o in enumerate(objs) if i % gn == gk]
    if a.max_objects:
        objs = objs[:a.max_objects]

    limit = {"del": a.n_del if "del" in types else 0, "add": a.n_add if "add" in types else 0,
             "modification": a.n_mod if "mod" in types else 0, "scale": a.n_scale if "scale" in types else 0}
    init = dict(limit)
    manifest = []
    for obj in objs:
        if all(v <= 0 for v in limit.values()):
            break
        pdir = Path(a.prod_root) / a.shard / obj
        p1 = pdir / "p1_encode"
        if not (p1 / "shape_slat_e512.npz").is_file():
            continue
        try:
            for et in ("deletion", "addition", "modification", "scale"):
                if limit[_LIMIT_KEY[et]] > 0:
                    manifest += export_type_from_prod(out_root, pdir, p1, a.shard, obj, et, limit)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] failed: %s", obj, e)
        log.info("after %s: del=%d add=%d mod=%d scale=%d", obj[:8],
                 init["del"] - limit["del"], init["add"] - limit["add"],
                 init["modification"] - limit["modification"], init["scale"] - limit["scale"])

    tag = a.tag or f"{a.shard}_{gk}of{gn}_{'-'.join(sorted(types))}"
    with open(out_root / "manifests" / f"all.{tag}.jsonl", "w") as f:
        for ln in manifest:
            f.write(ln + "\n")
    log.info("DONE: %d examples → %s (del=%d add=%d mod=%d scale=%d)", len(manifest), out_root,
             init["del"] - limit["del"], init["add"] - limit["add"],
             init["modification"] - limit["modification"], init["scale"] - limit["scale"])


if __name__ == "__main__":
    main()
