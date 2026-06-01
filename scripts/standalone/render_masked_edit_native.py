#!/usr/bin/env python3
"""Run ONE masked 3-stage edit and render it with TRELLIS.2's OWN renderers.

This is the model-native counterpart to ``render_before_after_compare.py``
(which goes through GLB-export → blender).  Here we stay entirely inside the
TRELLIS.2 frame ([-0.5,0.5]^3, z-up look-at) and use ``trellis2.utils.
render_utils`` so what you see is exactly what the model produced — no GLB
export, no reframe, no blender:

  * Stage S1 (structure / 形变): the SS-edited occupancy.  Rendered as a
    ``Voxel`` BEFORE (coords0) vs AFTER (coords_new) so the grow/shrink of the
    edited part is directly visible.
  * Final decoded mesh: ``MeshWithVoxel`` via ``PbrMeshRenderer`` + HDR envmap
    → shaded + normal (the actual edited 3D result).

It calls ``trellis2_3d._build_p4_mesh`` (the real pipeline path), so it also
exercises + dumps the retained latents (``coords0/coords_new``, shape SLat,
tex SLat) to ``<out>/latents/``.

Run under the trellis2 env, pinning a GPU via CUDA_VISIBLE_DEVICES:

  CUDA_VISIBLE_DEVICES=3 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
  /mnt/zsn/miniconda3/envs/trellis2/bin/python \
    scripts/standalone/render_masked_edit_native.py \
      --shard 08 --obj <obj_id> --edit-id <edit_id> \
      --s1-pad 3 --s1-thresh 0.1 \
      --out /tmp/native_<edit_id>
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

TRELLIS2_DIR = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("native-viz")


# ─────────────────── rendering helpers ──────────────────────────────

def _tile(frames, cols=2, bg=255):
    """List of HxWx3 uint8 → single 2×N tiled image."""
    frames = list(frames)
    n = len(frames)
    rows = (n + cols - 1) // cols
    h, w, _ = frames[0].shape
    canvas = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return canvas


def _snapshot(render_utils, sample, *, nviews=4, r=2.0, fov=40.0, res=512, **kw):
    """4-view snapshot dict {stream: [frames]} via render_utils.render_snapshot."""
    return render_utils.render_snapshot(
        sample, resolution=res, r=r, fov=fov, nviews=nviews, **kw)


def _voxel_from_coords(Voxel, coords3, res=64):
    """Build a height-colored Voxel for a [N,3] (0..res-1) occupancy."""
    import torch
    c = coords3.int().to("cuda")
    if c.dim() == 2 and c.shape[1] == 4:
        c = c[:, 1:]
    y = c[:, 1].float()
    y = (y - y.min()) / max(1e-6, float(y.max() - y.min()))
    color = torch.stack([0.15 + 0.85 * y,
                         0.45 + 0.4 * (1 - y),
                         0.95 - 0.7 * y], dim=1).float()
    return Voxel(origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / res,
                 coords=c, attrs=color, layout={"color": slice(0, 3)},
                 device="cuda")


def _pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    if isinstance(d, dict):
        return next(iter(d.values()))
    return d


def _dump_alpha_stats(mesh, coords0, coords_new, log):
    """Report decoded PBR alpha in the edited DOME vs the rest of the body.

    The dome (edited turret) is classified spatially: the MeshWithVoxel lives in
    the TRELLIS canonical Z-up frame, so the turret is the high-Z, centered-XY
    cluster.  If dome alpha ≈ body alpha (~1), the see-through look is NOT a PBR
    transparency issue — it is the shape-decode geometry.
    """
    import numpy as _np
    import torch
    attrs = getattr(mesh, "attrs", None)
    lay = getattr(mesh, "layout", None)
    if attrs is None or lay is None or "alpha" not in lay:
        log.info("[diag] no alpha layout on mesh (attrs=%s layout=%s)",
                 None if attrs is None else tuple(attrs.shape), lay)
        return
    coords = mesh.coords
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    res = int(round(1.0 / float(mesh.voxel_size)))
    origin = torch.tensor(mesh.origin, device=coords.device).float()
    pos = origin + (coords.float() + 0.5) * float(mesh.voxel_size)   # [-.5,.5]
    alpha = attrs[..., lay["alpha"]].float().reshape(attrs.shape[0], -1).mean(1)
    a = alpha.detach().cpu().numpy()
    p = pos.detach().cpu().numpy()
    z = p[:, 2]                                   # canonical up
    zhi = z.min() + 0.60 * (z.max() - z.min())
    dome = (z > zhi) & (_np.abs(p[:, 0]) < 0.14) & (_np.abs(p[:, 1]) < 0.14)
    body = z < z.min() + 0.45 * (z.max() - z.min())
    log.info("[diag] alpha: res=%d  overall mean=%.3f  N=%d", res, a.mean(), len(a))
    for nm, m in [("dome", dome), ("body", body)]:
        if m.any():
            log.info("[diag] alpha %-4s n=%-7d mean=%.3f min=%.3f p10=%.3f frac<0.5=%.3f",
                     nm, int(m.sum()), a[m].mean(), a[m].min(),
                     float(_np.percentile(a[m], 10)),
                     float((a[m] < 0.5).mean()))


def _glb_open_frac(gp, log):
    """Report dome vs body open-edge fraction of a GLB (mesh holeyness)."""
    import numpy as _np
    try:
        import trimesh
        import collections
        s = trimesh.load(str(gp), process=False)
        g = list(s.geometry.values())[0] if hasattr(s, "geometry") else s
        V = _np.asarray(g.vertices); F = _np.asarray(g.faces)
        y = V[:, 1]; x = V[:, 0]; zc = V[:, 2]                 # GLB is Y-up
        yhi = y.min() + 0.62 * (y.max() - y.min())
        dome_v = (y > yhi) & (_np.abs(x) < 0.13) & (_np.abs(zc) < 0.13)
        body_v = y < y.min() + 0.40 * (y.max() - y.min())

        def open_frac(vmask):
            fm = vmask[F].all(1)
            if not fm.any():
                return None, 0
            ec = collections.Counter()
            for f in F[fm]:
                for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    ec[(min(a_, b_), max(a_, b_))] += 1
            opn = sum(1 for v in ec.values() if v == 1)
            return opn / max(1, len(ec)), int(fm.sum())

        do, dn = open_frac(dome_v)
        bo, bn = open_frac(body_v)
        log.info("[diag]   %s: V=%d F=%d watertight=%s | dome open_frac=%s (F=%d)"
                 "  body open_frac=%s (F=%d)", gp.name, len(V), len(F),
                 g.is_watertight, None if do is None else round(do, 3), dn,
                 None if bo is None else round(bo, 3), bn)
        return do
    except Exception as e:  # pragma: no cover
        log.warning("[diag] GLB inspect failed: %s", e)
        return None


def _export_and_inspect_glb(mesh, out_dir, p25_cfg, log, bands=(1,)):
    """o_voxel.to_glb at each remesh ``band`` → <out>/after[_bN].glb, reporting
    dome/body open-edge fraction.  A wider band shrink-wraps a closed surface
    around the holey flexicubes mesh and bridges voxel-scale holes (cheap,
    export-side, no re-decode)."""
    import o_voxel
    aabb = p25_cfg.get("trellis2_aabb", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    for band in bands:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=aabb, decimation_target=1_000_000, texture_size=4096,
            remesh=True, remesh_band=int(band), remesh_project=0, verbose=False,
        )
        gp = out_dir / ("after.glb" if band == bands[0] else f"after_b{band}.glb")
        glb.export(str(gp))
        log.info("[diag] wrote %s (remesh_band=%d)", gp.name, band)
        _glb_open_frac(gp, log)


# ─────────────────── main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit-id", default=None,
                    help="edit_id to render; default = first flux spec")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--hdri", default=f"{TRELLIS2_DIR}/assets/hdri/forest.exr")
    ap.add_argument("--s1-pad", type=int, default=3)
    ap.add_argument("--s1-thresh", type=float, default=0.1)
    ap.add_argument("--canonical", action="store_true",
                    help="encode + mask in TRELLIS canonical Z-up frame")
    ap.add_argument("--p1-path", default=None,
                    help="shape_slat.npz to use; default = ctx p1_encode")
    ap.add_argument("--encode", action="store_true",
                    help="(re)encode mesh → --p1-path first (honors --canonical)")
    ap.add_argument("--s2-warmstart", action="store_true",
                    help="warm-start edit-region shape tokens from inverted "
                         "original (reduces spikes)")
    ap.add_argument("--s2-nn-init", action="store_true",
                    help="also nearest-neighbor-init newly-grown edit tokens "
                         "(implies warmstart; finishes the core spikes)")
    ap.add_argument("--s1-densify", type=int, default=0,
                    help="dilate the edited-region S1 occupancy by N cells so "
                         "the shape decoder can CLOSE the surface (fixes the "
                         "see-through holey dome); 0 = off")
    ap.add_argument("--s2-anchor-mode", default="perstep",
                    choices=["perstep", "release_late", "posthoc"],
                    help="how preserved tokens are pinned: perstep (legacy, "
                         "holey part), release_late, or posthoc (free gen + "
                         "overwrite body = solid part)")
    ap.add_argument("--s2-anchor-cutoff", type=float, default=0.3,
                    help="release_late: anchor only while t>=cutoff")
    ap.add_argument("--white-model", action="store_true",
                    help="decode with FORCED solid grey PBR (alpha=1) — the "
                         "geometry-vs-alpha ablation: if the dome is still "
                         "see-through, the transparency is shape-decode geometry")
    ap.add_argument("--export-glb", action="store_true", default=True,
                    help="also o_voxel.to_glb → <out>/after.glb and report "
                         "dome/body open-edge fraction (mesh holeyness)")
    ap.add_argument("--no-export-glb", dest="export_glb", action="store_false")
    ap.add_argument("--remesh-bands", default="1",
                    help="comma list of remesh_band values to export+inspect "
                         "(e.g. 1,2,3) — wider bridges holey-dome gaps")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--nviews", type=int, default=4)
    ap.add_argument("--num-frames", type=int, default=36,
                    help="final-mesh turntable frames (0 = skip the mp4)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # heavy imports (after CUDA_VISIBLE_DEVICES already set by the launcher)
    import cv2
    import torch
    import imageio
    from PIL import Image

    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    from trellis2.representations import Voxel

    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    root = Path(args.root)
    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj}.npz"
    image_npz = Path(args.images_root) / args.shard / f"{args.obj}.npz"
    ctx = PipelineRoot(root=root).context(
        args.shard, args.obj, mesh_npz=mesh_npz, image_npz=image_npz)

    specs = list(iter_flux_specs(ctx))
    if not specs:
        raise SystemExit(f"no flux specs for {args.obj}")
    if args.edit_id:
        spec = next((s for s in specs if s.edit_id == args.edit_id), None)
        if spec is None:
            raise SystemExit(f"edit_id {args.edit_id} not in "
                             f"{[s.edit_id for s in specs]}")
    else:
        spec = specs[0]
    log.info("edit_id=%s  type=%s  parts=%s", spec.edit_id, spec.edit_type,
             getattr(spec, "selected_part_ids", None))

    e2d = ctx.dir / args.edits_2d_subdir
    input_png = e2d / f"{spec.edit_id}_input.png"
    edited_png = e2d / f"{spec.edit_id}_edited.png"
    for p in (input_png, edited_png):
        if not p.is_file():
            raise SystemExit(f"missing 2D image: {p}")
    orig_img = Image.open(input_png).convert("RGB")
    edited_img = Image.open(edited_png).convert("RGB")

    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR,
        "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
        "trellis2_s1_pad": args.s1_pad,
        "trellis2_s1_keep_thresh": args.s1_thresh,
        "trellis2_canonical_frame": args.canonical,
        "trellis2_s2_warmstart": args.s2_warmstart or args.s2_nn_init,
        "trellis2_s2_nn_init": args.s2_nn_init,
        "trellis2_s1_densify": args.s1_densify,
        "trellis2_s2_anchor_mode": args.s2_anchor_mode,
        "trellis2_s2_anchor_cutoff": args.s2_anchor_cutoff,
    }

    pipeline = T._ensure_pipeline(p25_cfg, log)

    p1_path = (Path(args.p1_path) if args.p1_path
               else ctx.dir / "p1_encode" / "shape_slat.npz")
    if args.encode:
        from partcraft.pipeline_v3 import trellis2_encode as TE
        log.info("re-encoding mesh → %s (canonical=%s) ...", p1_path,
                 args.canonical)
        enc = TE._ensure_encoder(p25_cfg, log)
        feats, coords = TE.encode_full_mesh(enc, ctx.mesh_npz,
                                            canonical=args.canonical)
        p1_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p1_path, feats=feats, coords=coords)
        del enc
        torch.cuda.empty_cache()
        log.info("encoded %d tokens → %s", coords.shape[0], p1_path)
    d = np.load(str(p1_path))
    p1_feats = torch.from_numpy(d["feats"]).float()
    p1_coords3 = torch.from_numpy(d["coords"]).int()
    log.info("P1 SLat: %d tokens (canonical=%s, from %s)",
             int(p1_coords3.shape[0]), args.canonical, p1_path)

    log.info("running masked 3-stage edit (s1_pad=%d s1_thresh=%.2f white_model=%s) ...",
             args.s1_pad, args.s1_thresh, args.white_model)
    mesh, latents = T._build_p4_mesh(
        pipeline, spec, edited_img, orig_img, p1_feats, p1_coords3,
        ctx.mesh_npz, p25_cfg, log, white_model=args.white_model)
    T._save_edit_latents(latents, out_dir, log)

    n0 = 0 if latents["coords0"] is None else latents["coords0"].shape[0]
    n1 = 0 if latents["coords_new"] is None else latents["coords_new"].shape[0]
    log.info("structure: %d → %d voxels", n0, n1)

    coords_new_t = torch.from_numpy(latents["coords_new"].astype(np.int32))
    coords0_t = torch.from_numpy(latents["coords0"].astype(np.int32))

    # ── geometry-vs-alpha diagnostics ─────────────────────────────────
    _dump_alpha_stats(mesh, coords0_t, coords_new_t, log)
    if args.export_glb:
        bands = tuple(int(b) for b in str(args.remesh_bands).split(",") if b.strip())
        _export_and_inspect_glb(mesh, out_dir, p25_cfg, log, bands=bands)

    # ── native renders ────────────────────────────────────────────────
    hdr = cv2.cvtColor(cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))

    coords0 = torch.from_numpy(latents["coords0"].astype(np.int32))
    coords_new = torch.from_numpy(latents["coords_new"].astype(np.int32))

    log.info("render SS before (voxel) ...")
    vb = _voxel_from_coords(Voxel, coords0)
    ss_before = _pick(_snapshot(render_utils, vb, nviews=args.nviews,
                                res=args.resolution), "color")
    log.info("render SS after (voxel) ...")
    va = _voxel_from_coords(Voxel, coords_new)
    ss_after = _pick(_snapshot(render_utils, va, nviews=args.nviews,
                               res=args.resolution), "color")

    log.info("render final mesh (PBR snapshot) ...")
    pbr = _snapshot(render_utils, mesh, nviews=args.nviews,
                    res=args.resolution, envmap=envmap)
    final_shaded = _pick(pbr, "shaded", "color")
    final_normal = _pick(pbr, "normal")

    # save 2×2 tiles
    for name, frames in [("ss_before", ss_before), ("ss_after", ss_after),
                         ("final_shaded", final_shaded),
                         ("final_normal", final_normal)]:
        Image.fromarray(_tile(frames, cols=2)).save(out_dir / f"{name}_grid.png")

    # ── single-row contact sheet (first view of each) ─────────────────
    def _load_resize(p, sz=384):
        im = Image.open(p).convert("RGB"); im.thumbnail((sz, sz)); return im

    def _arr_resize(a, sz=384):
        im = Image.fromarray(a); im.thumbnail((sz, sz)); return im

    SZ = 384
    panels = [
        ("2D input", _load_resize(input_png, SZ)),
        ("2D edited", _load_resize(edited_png, SZ)),
        ("SS before", _arr_resize(ss_before[0], SZ)),
        ("SS after", _arr_resize(ss_after[0], SZ)),
        ("final shaded", _arr_resize(final_shaded[0], SZ)),
        ("final normal", _arr_resize(final_normal[0], SZ)),
    ]
    pad, cap = 8, 26
    W = len(panels) * SZ + (len(panels) + 1) * pad
    H = SZ + 2 * pad + cap + 30
    from PIL import ImageDraw
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6),
           f"{args.obj[:12]} | {spec.edit_id} | {spec.edit_type} | "
           f"parts={list(getattr(spec, 'selected_part_ids', []))} | "
           f"SS {n0}->{n1} vox | s1_pad={args.s1_pad} thresh={args.s1_thresh}",
           fill=(0, 0, 0))
    for i, (lab, im) in enumerate(panels):
        x = pad + i * (SZ + pad)
        y = cap
        canvas = Image.new("RGB", (SZ, SZ), (242, 242, 242))
        canvas.paste(im, ((SZ - im.width) // 2, (SZ - im.height) // 2))
        sheet.paste(canvas, (x, y))
        d.text((x + 4, y + SZ + 4), lab, fill=(40, 40, 40))
    sheet_path = out_dir / "native_sheet.png"
    sheet.save(sheet_path)
    log.info("wrote %s", sheet_path)

    # ── optional final-mesh turntable (forest PBR) ────────────────────
    if args.num_frames > 0:
        log.info("render final turntable (%d frames) ...", args.num_frames)
        raw = render_utils.render_video(
            mesh, resolution=args.resolution, num_frames=args.num_frames,
            r=2, fov=40, envmap=envmap)
        shaded = raw["shaded"] if "shaded" in raw else _pick(raw, "color")
        imageio.mimsave(out_dir / "final_turntable.mp4", list(shaded),
                        fps=15, macro_block_size=1)
        log.info("wrote %s", out_dir / "final_turntable.mp4")

    log.info("DONE → %s", out_dir)


if __name__ == "__main__":
    main()
