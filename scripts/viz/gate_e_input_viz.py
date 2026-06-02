"""Visualise the gate-E input pair for already-rendered edits (no decode).

For each edit dir with an ``after_shaded.png`` (the post-edit-latents render),
render the BEFORE = original-mesh o-voxel at the SAME ``render_snapshot`` cameras
(sin/cos convention, r=2, fov=40, yaw_off=-16°, pitch=20°, 4 views tiled 2×2),
then stack BEFORE over AFTER with labels — exactly what gate-E would judge.

    CUDA_VISIBLE_DEVICES=3 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/gate_e_input_viz.py \
      --run data/Pxform_v2/_rerun_v2/08_ss_alignt1 --shard 08 \
      --out data/Pxform_v2/_scratch/gate_e_viz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _snapshot_cams(nviews=4, r=2.0, fov_deg=40.0, yaw_off_deg=-16.0,
                   pitch_deg=20.0, device="cuda"):
    """render_snapshot's cameras (TRELLIS sin/cos convention, Z-up look-at)."""
    import torch, utils3d
    yaws = [2 * np.pi * k / nviews + np.radians(yaw_off_deg) for k in range(nviews)]
    pitchs = [np.radians(pitch_deg)] * nviews
    extr, intr = [], []
    up = torch.tensor([0., 0., 1.], device=device)
    look = torch.tensor([0., 0., 0.], device=device)
    for y, p in zip(yaws, pitchs):
        fov = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
        Y = torch.tensor(float(y), device=device); P = torch.tensor(float(p), device=device)
        orig = torch.tensor([torch.sin(Y) * torch.cos(P),
                             torch.cos(Y) * torch.cos(P),
                             torch.sin(P)], device=device) * r
        extr.append(utils3d.torch.extrinsics_look_at(orig, look, up))
        intr.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return extr, intr


def _tile(frames, cols=2, bg=20):
    rows = (len(frames) + cols - 1) // cols
    h, w, _ = frames[0].shape
    canvas = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return canvas


def _render_before_tile(mesh_npz, res=512, device="cuda"):
    import torch, o_voxel
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    coords, attr = OVR.mesh_to_colored_ovox(mesh_npz, grid_size=512)
    pos = torch.from_numpy(coords.astype(np.float32)).to(device) / 512 - 0.5
    attrs = torch.from_numpy(attr["base_color"].cpu().numpy().astype(np.float32)
                             if hasattr(attr["base_color"], "cpu")
                             else np.asarray(attr["base_color"], np.float32)).to(device) / 255.0
    extr, intr = _snapshot_cams(device=device)
    rend = o_voxel.rasterize.VoxelRenderer(rendering_options={"resolution": res, "ssaa": 2})
    frames = []
    for e, i in zip(extr, intr):
        out = rend.render(pos, attrs, 1.0 / 512, e, i)
        color = out.attr * out.alpha.clamp(0, 1)[None]   # composite on black
        frames.append((color.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return _tile(frames, cols=2)


def _label(img, text, color=(60, 230, 60)):
    import cv2
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="rerun dir with <obj>/<edit>/after_shaded.png")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/gate_e_viz")
    args = ap.parse_args()

    import cv2
    run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edits = sorted(p.parent for p in run.glob("*/*/after_shaded.png"))
    print(f"[gate-E viz] {len(edits)} edits in {run}")

    before_cache: dict[str, np.ndarray] = {}
    pairs = []
    for ed in edits:
        obj = ed.parent.name; edit = ed.name
        mesh_npz = Path(args.mesh_root) / args.shard / f"{obj}.npz"
        if not mesh_npz.is_file():
            print(f"  skip {edit}: no mesh"); continue
        if obj not in before_cache:
            before_cache[obj] = _render_before_tile(mesh_npz)
        before = before_cache[obj]
        after = cv2.cvtColor(cv2.imread(str(ed / "after_shaded.png")), cv2.COLOR_BGR2RGB)
        if after.shape[:2] != before.shape[:2]:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        col = _label(before, f"BEFORE (o-voxel)  {obj[:8]}")
        cor = _label(after, f"AFTER (edited latents)  {edit}", color=(230, 160, 60))
        pair = np.concatenate([col, cor], axis=0)
        cv2.imwrite(str(out / f"{edit}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
        pairs.append(pair)
        print(f"  {edit}")

    if pairs:
        W = max(p.shape[1] for p in pairs)
        tiles = []
        for p in pairs:
            if p.shape[1] < W:
                p = np.concatenate([p, np.full((p.shape[0], W - p.shape[1], 3), 255, np.uint8)], axis=1)
            tiles.append(p)
            tiles.append(np.full((10, W, 3), 255, np.uint8))
        sheet = np.concatenate(tiles[:-1], axis=0)
        cv2.imwrite(str(out / "ALL_gate_e_pairs.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"[gate-E viz] wrote {out/'ALL_gate_e_pairs.png'}")


if __name__ == "__main__":
    main()
