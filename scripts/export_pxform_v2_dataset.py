#!/usr/bin/env python3
"""Export an H3D_v1-style dataset ("Pxform_v2", export build) in the v2
(TRELLIS.2) latent space, at the 512 edit resolution (32³ slat / 16³ ss).
Lives on the data mount, separate from the raw pipeline output:
  data/Pxform_v2/export/

Per-edit dir mirrors the prod edit-latent format (3 files) + a mask sidecar:
  <edit_type>/<shard>/<obj>/<edit_id>/
      shape_slat.npz  after shape  (feats[N,32], coords[N,3] @32³)
      tex_slat.npz    after tex    (feats[N,32], coords[N,3]; shares coords w/ shape)
      ss.npz          after struct (coords0, coords_new, edit_grid, keep16, parts, ...)
      mask.npz        mask_keep_ss[16,16,16] + mask_keep_slat[N_after]
                      + mask_keep_slat_before[N_before] + selected_part_ids
      before.png / after.png / meta.json / view.meta.json
  _assets/<shard>/<obj>/   shape_slat.npz(e512 32³) + tex_slat.npz(e512) + ss.npz(1,8,16,16,16)
  manifests/all.<tag>.jsonl

Sourcing (all v2-native; v1 NOT reused — v1 prompts differ):
  mod/scale : COPY prod edits_3d/<eid>/latents/{shape,tex}_slat.npz + ss.npz verbatim
              (already 32³, ss.npz already carries edit_grid/keep16/parts) → derive mask.
  del       : s5b _merge_surviving_parts_from_npz → after_new.glb → T2 encode @grid512
              (original-frame M); ss.npz built from part_edit_grid_64(pad=4)+downsample
              (reproduces prod keep16/edit_grid exactly).
  add       : inverse of del (after = original e512; coords0/coords_new swapped).
"""
import argparse, json, os, shutil, tempfile, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("export_pxform_v2")

VIEW_ORDER = ["front", "right", "back", "left", "down"]
S1_PAD = 4          # matches config trellis2_s1_pad
S1_THRESH = 0.1     # matches config s1_thresh


# ───────────────────────── models ─────────────────────────
def load_models(codebase, ckpt_root):
    from partcraft.pipeline_v3.trellis2_encode import _ensure_encoders
    from partcraft.pipeline_v3 import trellis2_3d as T3
    p25 = {"trellis2_codebase": codebase, "trellis2_ckpt": ckpt_root, "ckpt_root": ckpt_root}
    return _ensure_encoders(p25, log), T3._ensure_pipeline(p25, log), T3, p25


# ───────────────── encode after_new.glb @512 (32³), original frame ─────────────────
def encode_after_512(encoders, orig_mesh_npz, after_glb, canonical, grid_size=512):
    """shape+tex SLat for after_new.glb, normalized with the ORIGINAL mesh's M so
    the deleted-result coords share the same 32³ grid as the e512 'before'. No ss_enc
    (the dataset's after-ss is structural metadata, not a re-encoded latent)."""
    import trimesh, torch, o_voxel
    import trellis2.modules.sparse as sp
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    _, M = OVR._normalized_groups(OVR.load_full_scene(Path(orig_mesh_npz)), canonical=canonical)
    scene = trimesh.load(str(after_glb), file_type="glb", process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    groups, _ = OVR._normalized_groups(scene, canonical=canonical, M=M)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=grid_size, aabb=aabb,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual_local = (dv * grid_size - vi).clamp(0., 1.).float()
    if inter.dim() == 2 and inter.shape[1] == 3:
        inter3 = inter.float()
    else:
        b = inter.view(-1).to(torch.uint8)
        inter3 = torch.stack([(b & 1).bool(), ((b >> 1) & 1).bool(), ((b >> 2) & 1).bool()], -1).float()
    sh_coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], -1).int()
    vsp = sp.SparseTensor(feats=dual_local, coords=sh_coords).cuda()
    isp = vsp.replace(inter3.bool().float().cuda())
    with torch.no_grad():
        shape_slat = encoders["shape"](vsp, isp)
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=grid_size, aabb=aabb)
    def _f(x): return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([_f(attr["base_color"]), _f(attr["metallic"]),
                        _f(attr["roughness"]), _f(attr["alpha"])], -1).float()
    tx_coords = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1).int()
    tsp = sp.SparseTensor(feats=feats6, coords=tx_coords).cuda()
    with torch.no_grad():
        tex_slat = encoders["tex"](tsp)
    return {
        "shape_feats": shape_slat.feats.cpu().numpy().astype(np.float32),
        "shape_coords": shape_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "tex_feats": tex_slat.feats.cpu().numpy().astype(np.float32),
        "tex_coords": tex_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
    }


# ───────────────────────── structure (ss) + mask ─────────────────────────
def part_struct_grids(mesh_npz, pids, canonical):
    """Pipeline-exact part region: 64³ edit grid (pad=4) → 16³ keep mask +
    32³ edit_grid coords. Reproduces prod ss.npz keep16/edit_grid (validated IoU=1)."""
    import torch
    from partcraft.pipeline_v3.trellis2_part_mask import (
        part_edit_grid_64, edit_grid_64_to_keep16, downsample_edit_grid)
    g64 = part_edit_grid_64(mesh_npz, pids, pad=S1_PAD, canonical=canonical)
    keep16 = edit_grid_64_to_keep16(g64, thresh=S1_THRESH).cpu().numpy().astype(bool)
    g32 = downsample_edit_grid(g64, 2)
    edit_grid = torch.nonzero(g32).to(torch.int16).cpu().numpy()
    return keep16, edit_grid


def mask_from_ss(ss):
    """v1-style keep masks from an ss dict (coords0, coords_new, edit_grid, keep16, parts)."""
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


# ───────────────────────── decode + render ─────────────────────────
def render_all_views(pipeline, T3, p25, shape_feats, shape_coords, tex_feats, tex_coords, res=512):
    try:
        import torch
        import trellis2.modules.sparse as sp
        from partcraft.render import ovox_views as _ov
        from PIL import Image
        def _slat(feats, coords):
            f = torch.from_numpy(feats).float().cuda()
            c = torch.from_numpy(coords).int()
            c = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), c], 1).cuda()
            return sp.SparseTensor(feats=f, coords=c)
        mesh = pipeline.decode_latent(_slat(shape_feats, shape_coords),
                                      _slat(tex_feats, tex_coords), 512)[0]
        env = T3._get_envmap(p25, log)
        imgs = _ov.render_sample(mesh, envmap=env, resolution=res, bg=(1, 1, 1))
        return {k: Image.fromarray(np.asarray(v)) for k, v in imgs.items()}
    except Exception as e:
        log.warning("render failed: %s", e)
        return {}


def compute_best_view(overview_path, selected_part_ids):
    try:
        import cv2
        from partcraft.pipeline_v3.qc_rules import count_part_pixels_in_overview, _N_VIEWS
        if not overview_path.is_file() or not selected_part_ids:
            return 0, VIEW_ORDER[0]
        ov = cv2.imdecode(np.frombuffer(overview_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        px = [count_part_pixels_in_overview(ov, v, selected_part_ids) for v in range(_N_VIEWS)]
        bv = int(np.argmax(px)) if any(p > 0 for p in px) else 0
        return bv, VIEW_ORDER[bv]
    except Exception as e:
        log.warning("best_view failed: %s", e)
        return 0, VIEW_ORDER[0]


# ───────────────────────── io helpers ─────────────────────────
def _save(path, d):
    np.savez(path, **d)


def _hardlink(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assets_before(out_root, shard, obj, p1_dir):
    """Shared 'before' (original) at 32³: e512 shape/tex + p1 ss latent. Atomic, deduped."""
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


def _put_png(src, dst):
    if src is None:
        return
    if isinstance(src, (str, Path)):
        if Path(src).is_file():
            shutil.copy2(src, dst)
    else:
        src.save(dst)


def write_edit(out_root, edit_type, shard, obj, edit_id, shape_slat, tex_slat, ss, mask,
               before_png, after_png, meta, vmeta):
    ed = out_root / edit_type / shard / obj / edit_id
    ed.mkdir(parents=True, exist_ok=True)
    _save(ed / "shape_slat.npz", shape_slat)
    _save(ed / "tex_slat.npz", tex_slat)
    _save(ed / "ss.npz", ss)
    _save(ed / "mask.npz", mask)
    _put_png(before_png, ed / "before.png")
    _put_png(after_png, ed / "after.png")
    (ed / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (ed / "view.meta.json").write_text(json.dumps(vmeta, ensure_ascii=False, indent=2))


def manifest_line(edit_id, edit_type, shard, obj, instruction, quality, bv):
    return json.dumps({
        "edit_id": edit_id, "edit_type": edit_type, "instruction": instruction,
        "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                    "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
        "obj_id": obj, "quality": quality, "schema_version": 3, "shard": shard,
        "views": {"best_view_index": bv}}, ensure_ascii=False)


# ───────────────────────── parsed.json helpers ─────────────────────────
def load_parsed(prod_obj_dir):
    d = json.loads((prod_obj_dir / "phase1" / "parsed.json").read_text())
    p = d.get("parsed") or {}
    edits = p.get("edits") or []
    obj_desc = (p.get("object") or {}).get("full_desc", "")
    dels = [e for e in edits if e.get("edit_type") == "deletion"]
    non_dels = [e for e in edits if e.get("edit_type") != "deletion"]
    return obj_desc, dels, non_dels


def _seq(eid): return int(eid.rsplit("_", 1)[-1])
def _gate_e(st): return (st.get("gate_e") or st.get("gate_quality") or {}).get("status")
def _gate_a_score(st):
    try: return float(st["gate_a"]["verdict"]["vlm"]["score"])
    except Exception: return None
def _best_view(st):
    try: return int(st["gate_a"]["verdict"]["vlm"]["best_view"])
    except Exception: return 0


def _instruction(edit, obj_desc):
    out = {"object_desc": obj_desc}
    for k in ("prompt", "target_part_desc", "after_desc", "edit_params",
              "new_parts_desc", "new_part_desc"):
        if edit.get(k) not in (None, "", {}):
            out[k] = edit[k]
    return out


# ───────────────────────── exporters ─────────────────────────
def export_del_add(M, out_root, prod_obj_dir, mesh_npz, p1_dir, shard, obj,
                   canonical, do_render, limit):
    from partcraft.pipeline_v3.mesh_deletion import _merge_surviving_parts_from_npz
    from partcraft.pipeline_v3.addition_utils import invert_delete_prompt
    encoders, pipeline, T3, p25 = M
    status = json.loads((prod_obj_dir / "edit_status.json").read_text()).get("edits", {})
    obj_desc, dels_parsed, _ = load_parsed(prod_obj_dir)
    del_ids = sorted([k for k, v in status.items() if v.get("edit_type") == "deletion"], key=_seq)
    overview = prod_obj_dir / "phase1" / "overview.png"
    ad = assets_before(out_root, shard, obj, p1_dir)
    # original (e512) latents = del.before / add.after
    orig_sh = dict(np.load(ad / "shape_slat.npz"))
    orig_tx = dict(np.load(ad / "tex_slat.npz"))
    orig_views = render_all_views(pipeline, T3, p25, orig_sh["feats"], orig_sh["coords"],
                                  orig_tx["feats"], orig_tx["coords"]) if do_render else {}
    lines = []
    for i, del_id in enumerate(del_ids):
        if i >= len(dels_parsed) or (limit["del"] <= 0 and limit["add"] <= 0):
            break
        spec = dels_parsed[i]
        pids = [int(x) for x in (spec.get("selected_part_ids") or [])]
        with tempfile.TemporaryDirectory() as td:
            if not _merge_surviving_parts_from_npz(mesh_npz, pids, Path(td), force=True, logger=log):
                log.warning("[%s] s5b merge failed %s", obj, del_id); continue
            enc = encode_after_512(encoders, mesh_npz, Path(td) / "after_new.glb", canonical)
        keep16, edit_grid = part_struct_grids(mesh_npz, pids, canonical)
        bv, vname = compute_best_view(overview, pids)
        del_after_views = render_all_views(pipeline, T3, p25, enc["shape_feats"], enc["shape_coords"],
                                           enc["tex_feats"], enc["tex_coords"]) if do_render else {}
        seq = f"{_seq(del_id):03d}"
        sh_after = {"feats": enc["shape_feats"], "coords": enc["shape_coords"].astype(np.int16)}
        tx_after = {"feats": enc["tex_feats"], "coords": enc["tex_coords"].astype(np.int16)}
        parts = np.asarray(pids, dtype=np.int32)

        # ── deletion: before=original, after=deleted ──
        if limit["del"] > 0:
            ss = {"coords0": orig_sh["coords"].astype(np.int16),
                  "coords_new": enc["shape_coords"].astype(np.int16),
                  "edit_grid": edit_grid, "keep16": keep16, "parts": parts,
                  "edit_type": np.asarray("deletion"),
                  "s1_pad": np.int32(S1_PAD), "s1_thresh": np.float32(S1_THRESH)}
            mask = mask_from_ss(ss)
            instr = _instruction(spec, obj_desc)
            meta = {"edit_id": del_id, "edit_type": "deletion", "instruction": instr,
                    "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                                "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
                    "obj_id": obj, "quality": {"alignment_score": 1.0, "final_pass": False, "quality_score": 0.0},
                    "schema_version": 3, "shard": shard, "views": {"best_view_index": bv}}
            vmeta = {"edit_id": del_id, "edit_type": "deletion", "shard": shard, "obj_id": obj,
                     "best_view_index": bv, "view_name": vname, "selected_part_ids": pids,
                     "view_source": "overview_part_pixel_argmax"}
            write_edit(out_root, "deletion", shard, obj, del_id, sh_after, tx_after, ss, mask,
                       orig_views.get(vname), del_after_views.get(vname), meta, vmeta)
            lines.append(manifest_line(del_id, "deletion", shard, obj, instr, meta["quality"], bv))
            limit["del"] -= 1
        # ── addition: before=deleted, after=original (inverse) ──
        if limit["add"] > 0:
            add_id = f"add_{obj}_{seq}"
            ss = {"coords0": enc["shape_coords"].astype(np.int16),
                  "coords_new": orig_sh["coords"].astype(np.int16),
                  "edit_grid": edit_grid, "keep16": keep16, "parts": parts,
                  "edit_type": np.asarray("addition"),
                  "s1_pad": np.int32(S1_PAD), "s1_thresh": np.float32(S1_THRESH)}
            mask = mask_from_ss(ss)
            ap = invert_delete_prompt(spec.get("prompt", "")) if spec.get("prompt") else ""
            instr = {"object_desc": obj_desc, "prompt": ap,
                     "target_part_desc": spec.get("target_part_desc", ""), "synthesized": True}
            meta = {"edit_id": add_id, "edit_type": "addition", "instruction": instr,
                    "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                                "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
                    "obj_id": obj, "quality": {"alignment_score": 1.0, "final_pass": False, "quality_score": 0.0},
                    "schema_version": 3, "shard": shard, "views": {"best_view_index": bv}}
            vmeta = {"edit_id": add_id, "edit_type": "addition", "shard": shard, "obj_id": obj,
                     "best_view_index": bv, "view_name": vname, "paired_deletion_edit_id": del_id,
                     "selected_part_ids": pids, "view_source": "overview_part_pixel_argmax"}
            # after = original e512
            sh_o = {"feats": orig_sh["feats"], "coords": orig_sh["coords"].astype(np.int16)}
            tx_o = {"feats": orig_tx["feats"], "coords": orig_tx["coords"].astype(np.int16)}
            write_edit(out_root, "addition", shard, obj, add_id, sh_o, tx_o, ss, mask,
                       del_after_views.get(vname), orig_views.get(vname), meta, vmeta)
            lines.append(manifest_line(add_id, "addition", shard, obj, instr, meta["quality"], bv))
            limit["add"] -= 1
    return lines


def export_mod_scale(out_root, prod_obj_dir, p1_dir, shard, obj, edit_type, limit):
    # pure CPU copy: prod edit latents are already 32³ and ss.npz carries edit_grid/keep16/parts.
    status = json.loads((prod_obj_dir / "edit_status.json").read_text()).get("edits", {})
    obj_desc, _, non_dels = load_parsed(prod_obj_dir)
    assets_before(out_root, shard, obj, p1_dir)
    cands = sorted([k for k, v in status.items()
                    if v.get("edit_type") == edit_type and _gate_e(v.get("stages") or {}) == "pass"], key=_seq)
    lines = []
    for eid in cands:
        if limit[edit_type] <= 0:
            break
        lat = prod_obj_dir / "edits_3d" / eid / "latents"
        if not all((lat / f).is_file() for f in ("shape_slat.npz", "tex_slat.npz", "ss.npz")):
            continue
        st = status[eid].get("stages") or {}
        spec = non_dels[_seq(eid)] if _seq(eid) < len(non_dels) else {}
        ss = dict(np.load(lat / "ss.npz", allow_pickle=True))
        mask = mask_from_ss(ss)
        bv = _best_view(st); vname = VIEW_ORDER[bv] if 0 <= bv < len(VIEW_ORDER) else "front"
        bp = prod_obj_dir / "gate_views" / f"before_view_{vname}.png"
        ap = prod_obj_dir / "edits_3d" / eid / f"after_view_{vname}.png"
        instr = _instruction(spec, obj_desc)
        ga = _gate_a_score(st); qscore = 0.0
        try:
            vq = json.loads((prod_obj_dir / "edits_3d" / eid / "gate_e_judge.json").read_text()).get("judge", {}).get("visual_quality")
            qscore = round(vq / 5.0, 3) if vq is not None else 0.0
        except Exception:
            pass
        ed = out_root / edit_type / shard / obj / eid
        ed.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lat / "shape_slat.npz", ed / "shape_slat.npz")
        shutil.copy2(lat / "tex_slat.npz", ed / "tex_slat.npz")
        shutil.copy2(lat / "ss.npz", ed / "ss.npz")
        _save(ed / "mask.npz", mask)
        _put_png(bp, ed / "before.png")
        _put_png(ap, ed / "after.png")
        meta = {"edit_id": eid, "edit_type": edit_type, "instruction": instr,
                "lineage": {"pipeline_version": "v3", "source_dataset": "partverse",
                            "latent_space": "trellis2", "edit_res": 512, "build": "pxform_v2_export"},
                "obj_id": obj, "quality": {"alignment_score": ga if ga is not None else 1.0,
                                           "final_pass": True, "quality_score": qscore},
                "schema_version": 3, "shard": shard, "views": {"best_view_index": bv}}
        vmeta = {"edit_id": eid, "edit_type": edit_type, "shard": shard, "obj_id": obj,
                 "best_view_index": bv, "view_name": vname, "view_source": "pipeline_gate_a"}
        (ed / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (ed / "view.meta.json").write_text(json.dumps(vmeta, ensure_ascii=False, indent=2))
        lines.append(manifest_line(eid, edit_type, shard, obj, instr, meta["quality"], bv))
        limit[edit_type] -= 1
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-root", default="data/Pxform_v2/prod_posthoc_no2dqc/objects")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--shard", default="00")
    ap.add_argument("--out", default="data/Pxform_v2/export")
    ap.add_argument("--codebase", default="/mnt/zsn/3dobject/TRELLIS.2")
    ap.add_argument("--ckpt-root", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--canonical", type=int, default=1)
    ap.add_argument("--n-del", type=int, default=250)
    ap.add_argument("--n-add", type=int, default=250)
    ap.add_argument("--n-mod", type=int, default=250)
    ap.add_argument("--n-scale", type=int, default=250)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--types", default="del,add,mod,scale")
    ap.add_argument("--gpu-shard", default="0/1")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    types = {t.strip() for t in a.types.split(",") if t.strip()}
    gk, gn = (int(x) for x in a.gpu_shard.split("/"))
    need_gpu = bool(types & {"del", "add"})
    out_root = Path(a.out); (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    M = load_models(a.codebase, a.ckpt_root) if need_gpu else None
    canonical = bool(a.canonical)

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
        mesh = Path(a.mesh_root) / a.shard / f"{obj}.npz"
        if not (p1 / "shape_slat_e512.npz").is_file() or not mesh.is_file():
            continue
        try:
            if limit["del"] > 0 or limit["add"] > 0:
                manifest += export_del_add(M, out_root, pdir, mesh, p1, a.shard, obj,
                                           canonical, not a.no_render, limit)
            for et in ("modification", "scale"):
                if limit[et] > 0:
                    manifest += export_mod_scale(out_root, pdir, p1, a.shard, obj, et, limit)
        except Exception as e:
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
