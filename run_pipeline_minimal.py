#!/usr/bin/env python
"""Minimal single-object TRELLIS.2 edit-data pipeline.

The smallest end-to-end version of the data pipeline: take ONE partverse
object + one edit instruction and produce an edited 3D asset, exercising the
real masked latent editing (structure + geometry + material).

    partverse mesh.npz/image.npz + edit instruction
        │  1. pick the best view of the edited part            (PartVerseDataset)
        │  2. FLUX.1-Kontext 2D edit  →  edited view           (in-process)
        │  3. P1 encode original mesh → shape SLat             (trellis2_encode)
        │  4. masked 3D edit (SS / geometry / material)        (trellis2_3d._build_p4_mesh)
        ▼
    out/<shard>/<obj_id>/{input.png, edited.png, before.glb, after.glb}

This deliberately skips the heavy front-end (VLM edit generation + QC gates) and
the multi-server scheduler — the edit is given on the CLI. For the full,
multi-object, gated pipeline use ``partcraft.pipeline_v3.run_trellis2``.

Example:
    TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
    CUDA_VISIBLE_DEVICES=0 python run_pipeline_minimal.py \
        --shard 08 --obj-id bdd36c94f3f74f22b02b8a069c8d97b7 \
        --edit-type scale --part-id 1 \
        --instruction "Make the wooden bowl taller and deeper"
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("minimal")


def _part_label(mesh_npz: Path, part_id: int) -> str:
    try:
        d = np.load(mesh_npz, allow_pickle=True)
        caps = json.loads(d["part_captions.json"].tobytes().decode("utf-8"))
        c = caps.get(str(part_id))
        return (c[0] if isinstance(c, list) else str(c)) if c else f"part {part_id}"
    except Exception:
        return f"part {part_id}"


def _render_compare(pipeline, before_mesh, after_mesh, input_pil, edited_pil,
                    out_path, codebase, logger):
    """Render before/after MeshWithVoxel at matched views + show input/condition.

    Layout (each cell 512²):
        row0: input (orig 2D view) | condition (FLUX-edited 2D target) | (pad)
        row1: BEFORE render @ 3 yaws
        row2: AFTER  render @ 3 yaws
    """
    import cv2
    import numpy as np
    import torch
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap

    hdri = f"{codebase}/assets/hdri/forest.exr"
    exr = cv2.cvtColor(cv2.imread(hdri, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(exr, dtype=torch.float32, device="cuda"))
    yaws = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
    pitch = [0.35] * 3
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, 2, 40)

    def render(m):
        r = render_utils.render_frames(
            m, extr, intr, {"resolution": 512}, envmap=envmap, verbose=False)
        out = []
        for shaded, alpha in zip(r["shaded"], r["alpha"]):
            a = alpha.astype(np.float32) / 255.0          # HWC, 3 equal chans
            comp = shaded.astype(np.float32) * a + 255.0 * (1 - a)
            out.append(comp.clip(0, 255).astype(np.uint8))
        return out

    def cell(arr):
        return np.asarray(Image.fromarray(arr).resize((512, 512)))

    pad = np.full((512, 512, 3), 255, np.uint8)
    ref = np.concatenate([cell(np.asarray(input_pil.convert("RGB"))),
                          cell(np.asarray(edited_pil.convert("RGB"))), pad], axis=1)
    bef = np.concatenate([cell(x) for x in render(before_mesh)], axis=1)
    aft = np.concatenate([cell(x) for x in render(after_mesh)], axis=1)
    grid = np.concatenate([ref, bef, aft], axis=0)
    Image.fromarray(grid).save(out_path)
    logger.info("compare grid → %s (rows: input|condition / before / after)", out_path)


def _flux_edit(flux_ckpt: str, image: Image.Image, prompt: str, steps: int) -> Image.Image:
    """Edit a view with FLUX.1-Kontext in-process; frees the model after."""
    import torch
    from diffusers import DiffusionPipeline
    log.info("loading FLUX.1-Kontext %s ...", flux_ckpt)
    pipe = DiffusionPipeline.from_pretrained(
        flux_ckpt, torch_dtype=torch.bfloat16, device_map="cuda")
    with torch.inference_mode():
        out = pipe(image=image, prompt=prompt,
                   num_inference_steps=steps, num_images_per_prompt=1)
    edited = out.images[0].convert("RGB")
    del pipe
    torch.cuda.empty_cache()
    return edited


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--edit-type", default="scale",
                    choices=["modification", "scale", "material", "color", "global"])
    ap.add_argument("--part-id", type=int, default=1)
    ap.add_argument("--instruction", required=True,
                    help="natural-language edit, e.g. 'make the bowl taller'")
    ap.add_argument("--after-desc", default="")
    ap.add_argument("--data-root", default="data/partverse/inputs")
    ap.add_argument("--out-dir", default="outputs/minimal")
    ap.add_argument("--view", type=int, default=-1, help="-1 = auto best view")
    ap.add_argument("--flux-steps", type=int, default=28)
    ap.add_argument("--trellis2-codebase",
                    default=os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2"))
    ap.add_argument("--trellis2-ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--flux-ckpt", default="/mnt/zsn/ckpts/FLUX.1-Kontext-dev")
    ap.add_argument("--skip-flux", action="store_true",
                    help="reuse existing edited.png (re-run only the 3D edit)")
    ap.add_argument("--render", action="store_true",
                    help="also render a before/after/condition comparison grid")
    args = ap.parse_args()

    os.environ.setdefault("TRELLIS2_DIR", args.trellis2_codebase)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, args.trellis2_codebase)

    data_root = Path(args.data_root)
    mesh_npz = data_root / "mesh" / args.shard / f"{args.obj_id}.npz"
    images_root = data_root / "images"
    mesh_root = data_root / "mesh"
    for p in (mesh_npz, images_root / args.shard / f"{args.obj_id}.npz"):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    out_dir = Path(args.out_dir) / args.shard / args.obj_id
    out_dir.mkdir(parents=True, exist_ok=True)
    edit_id = f"{args.edit_type}_{args.part_id}"
    input_png = out_dir / f"{edit_id}_input.png"
    edited_png = out_dir / f"{edit_id}_edited.png"
    p25_cfg = {"trellis2_codebase": args.trellis2_codebase,
               "trellis2_ckpt": args.trellis2_ckpt}

    # ── 1. best view + input image ────────────────────────────────────
    from partcraft.io.partverse_dataset import PartVerseDataset
    from scripts.run_2d_edit import prepare_input_image, _build_edit_prompt
    ds = PartVerseDataset(str(images_root), str(mesh_root), [args.shard])
    rec = ds.load_object(args.shard, args.obj_id)
    view = args.view if args.view >= 0 else rec.get_best_view_for_parts([args.part_id])
    _, input_pil = prepare_input_image(rec, view, [args.part_id])
    input_pil.save(input_png)
    log.info("view=%d  input → %s", view, input_png)

    # ── 2. FLUX 2D edit ───────────────────────────────────────────────
    if args.skip_flux and edited_png.is_file():
        edited_pil = Image.open(edited_png).convert("RGB")
        log.info("reusing existing edited image %s", edited_png)
    else:
        label = _part_label(mesh_npz, args.part_id)
        prompt = _build_edit_prompt(args.instruction, args.after_desc,
                                    old_part_label=label, before_part_desc="",
                                    edit_type=args.edit_type)
        log.info("FLUX prompt: %s", prompt.replace("\n", " ")[:200])
        edited_pil = _flux_edit(args.flux_ckpt, input_pil, prompt, args.flux_steps)
        edited_pil.save(edited_png)
        log.info("edited → %s", edited_png)

    # ── 3. load TRELLIS.2 + P1 encode ─────────────────────────────────
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.pipeline_v3.trellis2_encode import encode_full_mesh
    from partcraft.pipeline_v3 import trellis2_3d as T
    import trellis2.models as t2_models
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    patch_dinov3_extractor()

    log.info("loading shape encoder + P1 encode ...")
    enc = t2_models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").eval().cuda()
    import torch
    feats, coords = encode_full_mesh(enc, mesh_npz, grid_size=1024)
    p1_feats = torch.from_numpy(feats).float()
    p1_coords = torch.from_numpy(coords).int()
    del enc
    torch.cuda.empty_cache()
    log.info("P1: %d tokens", p1_coords.shape[0])

    log.info("loading Trellis2ImageTo3DPipeline ...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.trellis2_ckpt)
    pipeline.cuda()

    # ── 4. masked 3D edit → after.glb ─────────────────────────────────
    spec = SimpleNamespace(edit_id=edit_id, edit_type=args.edit_type,
                           selected_part_ids=[args.part_id])
    mesh, _latents = T._build_p4_mesh(pipeline, spec, edited_pil, input_pil,
                                      p1_feats, p1_coords, mesh_npz, p25_cfg, log,
                                      white_model=False)
    after_glb = out_dir / "after.glb"
    T._run_and_export(pipeline, edited_pil, after_glb, p25_cfg, log, mesh_obj=mesh)

    # before.glb = original asset for side-by-side comparison
    d = np.load(mesh_npz, allow_pickle=True)
    (out_dir / "before.glb").write_bytes(d["full.glb"].tobytes())

    # ── 5. (optional) same-view before/after/condition render ─────────
    if args.render:
        from partcraft.pipeline_v3 import trellis2_edit_stages as t2e
        log.info("rendering before/after comparison ...")
        cond_orig = pipeline.get_cond([pipeline.preprocess_image(input_pil)], 1024)
        shape0 = t2e.sparse_denorm_shape(pipeline, p1_feats, p1_coords)
        tex0 = pipeline.sample_tex_slat(
            cond_orig, pipeline.models["tex_slat_flow_model_1024"], shape0, {})
        before_mesh = pipeline.decode_latent(shape0, tex0, 1024)[0]
        before_mesh.simplify(16_777_216)
        _render_compare(pipeline, before_mesh, mesh, input_pil, edited_pil,
                        out_dir / "compare.png", args.trellis2_codebase, log)

    log.info("DONE  →  %s", out_dir)
    for f in ("input", "edited"):
        log.info("  %s.png", f)
    log.info("  before.glb (original)  after.glb (edited, %d verts)",
             int(mesh.vertices.shape[0]))


if __name__ == "__main__":
    main()
