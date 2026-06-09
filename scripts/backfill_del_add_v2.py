#!/usr/bin/env python3
"""Backfill a SEPARATE del/add tree in the v2 (TRELLIS.2) latent space by
REUSING v1's deletion geometry (after_new.glb) + prompts — but RE-ENCODING
with the v2 shape/tex VAE (v1's DINOv2 SLAT latents are a different space and
are NOT reused).

Per v1 deletion edit we have: after_new.glb (part removed) + prompt.
We produce, in the v2 latent format (feats[N,32] + coords[N,3] + ss):

  del:  before = original p1_encode (already T2, reused)   after = encode(after_new.glb)
  add:  before = encode(after_new.glb)                     after = original p1_encode   (swap)

Frame alignment: after_new.glb is normalized with the ORIGINAL mesh's transform
M (not its own bounds) so the removed-part "after" lands on the SAME 64³ grid as
the original "before".  --smoke prints the coord-overlap diagnostic that proves it.

Output tree (kept separate from prod_posthoc_no2dqc so it can't collide with
v2's own del/add):
  <out>/objects/<shard>/<obj>/edits_3d/del_<obj>_<n>/{latents/{shape_slat,tex_slat,ss}.npz, after_new.glb, meta.json}
  <out>/objects/<shard>/<obj>/edits_3d/add_<obj>_<n>/meta.json   (latents referenced, not duplicated)
  <out>/objects/<shard>/<obj>/edit_status.json
"""
import argparse, json, io, os, sys, shutil, logging, glob
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("backfill_del_add")


# ───────────────────────── encode helpers ─────────────────────────
def load_encoders(p25_cfg):
    from partcraft.pipeline_v3.trellis2_encode import _ensure_encoders
    return _ensure_encoders(p25_cfg, log)


def _norm_M_from_original(orig_mesh_npz, canonical):
    """Normalization transform M computed from the ORIGINAL full mesh."""
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    scene = OVR.load_full_scene(Path(orig_mesh_npz))
    _groups, M = OVR._normalized_groups(scene, canonical=canonical)
    return M


def _encode_glb_in_frame(encoders, after_glb, M, grid_size, canonical):
    """Encode after_new.glb → {shape,tex,ss} latents, normalized with the
    given M (the original mesh's frame) so coords share the original grid."""
    import trimesh, torch, o_voxel
    import trellis2.modules.sparse as sp
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR

    scene = trimesh.load(str(after_glb), file_type="glb", process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    groups, _ = OVR._normalized_groups(scene, canonical=canonical, M=M)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    # shape (dual grid) — mirrors trellis2_encode.encode_shape_tex_ss
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

    # tex (volumetric PBR attr)
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=grid_size, aabb=aabb)
    def _f(x):
        return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([_f(attr["base_color"]), _f(attr["metallic"]),
                        _f(attr["roughness"]), _f(attr["alpha"])], -1).float()
    tx_coords = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1).int()
    tsp = sp.SparseTensor(feats=feats6, coords=tx_coords).cuda()
    with torch.no_grad():
        tex_slat = encoders["tex"](tsp)

    # ss (occupancy @64³ → ss_enc)
    c64 = shape_slat.coords[:, 1:].long()
    occ = torch.zeros(1, 1, 64, 64, 64, device=shape_slat.device)
    occ[0, 0, c64[:, 0], c64[:, 1], c64[:, 2]] = 1.0
    with torch.no_grad():
        z_s = encoders["ss"](occ.float())

    return {
        "shape_feats": shape_slat.feats.cpu().numpy().astype(np.float32),
        "shape_coords": shape_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "tex_feats": tex_slat.feats.cpu().numpy().astype(np.float32),
        "tex_coords": tex_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "ss": z_s.detach().cpu().numpy().astype(np.float32),
    }


def _coord_overlap(orig_coords, after_coords):
    """Fraction of after-voxels that coincide with an original voxel (frame
    alignment sanity: a deletion's surviving geometry should mostly overlap)."""
    a = {tuple(map(int, c)) for c in orig_coords}
    b = [tuple(map(int, c)) for c in after_coords]
    if not b:
        return 0.0
    return sum(1 for c in b if c in a) / len(b)


# ───────────────────────── per-edit / per-object ─────────────────────────
def _load_parsed_edits(v1_obj_dir):
    """v1 phase1/parsed.json → list of {edit_type, prompt}; edit seq == list index."""
    p = v1_obj_dir / "phase1" / "parsed.json"
    if not p.is_file():
        return []
    try:
        return (json.loads(p.read_text()).get("parsed") or {}).get("edits") or []
    except Exception:
        return []


def _v1_add_meta(v1_obj_dir, obj, seq):
    """v1 add_<obj>_<seq>/meta.json — carries prompt + rich part metadata."""
    mp = v1_obj_dir / "edits_3d" / f"add_{obj}_{seq}" / "meta.json"
    if mp.is_file():
        try:
            return json.loads(mp.read_text())
        except Exception:
            pass
    return {}


def process_object(encoders, v1_obj_dir, orig_mesh_npz, p1_encode_dir, out_obj_dir,
                   shard, obj, grid_size, canonical, smoke=False):
    v1_status = {}
    sp = v1_obj_dir / "edit_status.json"
    if sp.is_file():
        try:
            v1_status = json.loads(sp.read_text())
        except Exception:
            pass

    del_dirs = sorted((v1_obj_dir / "edits_3d").glob("del_*"))
    del_dirs = [d for d in del_dirs if (d / "after_new.glb").is_file()]
    if not del_dirs:
        return 0, []

    M = _norm_M_from_original(orig_mesh_npz, canonical)
    orig_shape_coords = np.load(p1_encode_dir / "shape_slat.npz")["coords"]
    parsed_edits = _load_parsed_edits(v1_obj_dir)

    edits_status = {}
    n = 0
    diags = []
    for di, ddir in enumerate(del_dirs):
        del_id = ddir.name                       # del_<obj>_<seq>
        seq = del_id.rsplit("_", 1)[-1]
        try:
            enc = _encode_glb_in_frame(encoders, ddir / "after_new.glb", M, grid_size, canonical)
        except Exception as e:
            log.warning("[%s/%s] encode failed: %s", obj, del_id, e)
            continue
        ov = _coord_overlap(orig_shape_coords, enc["shape_coords"])
        diags.append((del_id, len(enc["shape_coords"]), len(orig_shape_coords), round(ov, 3)))

        if not smoke:
            # ── del dir: latents (the AFTER = deleted state) + glb + meta
            ed = out_obj_dir / "edits_3d" / del_id
            (ed / "latents").mkdir(parents=True, exist_ok=True)
            np.savez(ed / "latents" / "shape_slat.npz",
                     feats=enc["shape_feats"], coords=enc["shape_coords"])
            np.savez(ed / "latents" / "tex_slat.npz",
                     feats=enc["tex_feats"], coords=enc["tex_coords"])
            np.savez(ed / "latents" / "ss.npz", ss=enc["ss"])
            shutil.copy2(ddir / "after_new.glb", ed / "after_new.glb")
            # del prompt: parsed.edits[seq] (seq == global edit index); add meta = rich fields
            pe = parsed_edits[int(seq)] if str(seq).isdigit() and int(seq) < len(parsed_edits) else {}
            dprompt = pe.get("prompt", "") if pe.get("edit_type") == "deletion" else pe.get("prompt", "")
            am = _v1_add_meta(v1_obj_dir, obj, seq)
            rich = {k: am[k] for k in ("target_part_desc", "object_desc", "part_labels",
                                       "selected_part_ids", "view_index") if k in am}
            (ed / "meta.json").write_text(json.dumps({
                "edit_id": del_id, "edit_type": "deletion",
                "obj_id": obj, "shard": shard,
                "prompt": dprompt,
                **rich,
                "before": "p1_encode (original, reused)",
                "after": "latents/ (re-encoded after_new.glb, T2)",
                "reused_from_v1": str(ddir),
            }, ensure_ascii=False, indent=2))

            # ── add dir: inverse (before=del.after, after=original); latents referenced
            add_id = f"add_{obj}_{seq}"
            ad = out_obj_dir / "edits_3d" / add_id
            ad.mkdir(parents=True, exist_ok=True)
            from partcraft.pipeline_v3.addition_utils import invert_delete_prompt
            aprompt = am.get("prompt") or (invert_delete_prompt(dprompt) if dprompt else "")
            (ad / "meta.json").write_text(json.dumps({
                "edit_id": add_id, "edit_type": "addition",
                "obj_id": obj, "shard": shard,
                "source_del_id": del_id,
                "prompt": aprompt,
                **rich,
                "before": f"edits_3d/{del_id}/latents (del.after)",
                "after": "p1_encode (original)",
                "rationale": f"inverse of {del_id}",
            }, ensure_ascii=False, indent=2))

            edits_status[del_id] = {"edit_type": "deletion",
                                    "stages": {"reencode": {"status": "done"}},
                                    "frame_overlap": round(ov, 3)}
            edits_status[add_id] = {"edit_type": "addition",
                                    "stages": {"reencode": {"status": "done"}},
                                    "source_del_id": del_id}
        n += 1

    if not smoke and edits_status:
        out_obj_dir.mkdir(parents=True, exist_ok=True)
        (out_obj_dir / "edit_status.json").write_text(json.dumps({
            "obj_id": obj, "shard": shard, "schema_version": 1,
            "source": "backfill_del_add_v2 (reuse v1 glb + T2 re-encode)",
            "edits": edits_status,
        }, ensure_ascii=False, indent=2))
    return n, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", default="data/partverse/outputs/partverse/shard00/mode_e_text_align/objects")
    ap.add_argument("--shard", default="00")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--p1-root", default="data/Pxform_v2/prod_posthoc_no2dqc/objects",
                    help="v2 prod tree holding original p1_encode (T2 before)")
    ap.add_argument("--out", default="data/Pxform_v2/del_add_reuse/objects")
    ap.add_argument("--codebase", default="/mnt/zsn/3dobject/TRELLIS.2")
    ap.add_argument("--grid-size", type=int, default=1024)
    ap.add_argument("--canonical", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="max objects (0=all)")
    ap.add_argument("--smoke", action="store_true", help="encode + print frame diag, write nothing")
    a = ap.parse_args()

    p25_cfg = {"trellis2_codebase": a.codebase}
    encoders = load_encoders(p25_cfg)

    v1_root = Path(a.v1_root)
    obj_parent = v1_root / a.shard               # v1 layout: objects/<shard>/<obj_id>
    objs = sorted([p.name for p in obj_parent.iterdir() if p.is_dir()])
    if a.limit:
        objs = objs[:a.limit]
    log.info("objects to process: %d (shard %s)", len(objs), a.shard)

    tot = 0
    for obj in objs:
        v1_obj = obj_parent / obj
        orig_mesh = Path(a.mesh_root) / a.shard / f"{obj}.npz"
        p1_dir = Path(a.p1_root) / a.shard / obj / "p1_encode"
        if not orig_mesh.is_file():
            log.warning("[%s] missing original mesh.npz %s — skip", obj, orig_mesh); continue
        if not (p1_dir / "shape_slat.npz").is_file():
            log.warning("[%s] missing v2 p1_encode %s — skip", obj, p1_dir); continue
        out_obj = Path(a.out) / a.shard / obj
        n, diags = process_object(encoders, v1_obj, orig_mesh, p1_dir, out_obj,
                                  a.shard, obj, a.grid_size, bool(a.canonical), smoke=a.smoke)
        tot += n
        for did, na, no, ov in diags:
            log.info("  %s  after_vox=%d orig_vox=%d  frame_overlap=%.3f", did, na, no, ov)
    log.info("DONE: %d del/add pairs across %d objects → %s", tot, len(objs),
             "(smoke, nothing written)" if a.smoke else a.out)


if __name__ == "__main__":
    main()
