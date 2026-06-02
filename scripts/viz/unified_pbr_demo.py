"""Demo: before & after BOTH via the native PbrMeshRenderer (unified).

BEFORE = original glb → MeshWithPbrMaterial → render_snapshot (+envmap)
AFTER  = existing after_shaded.png (decoded edited mesh → same render_snapshot)
Both use TRELLIS.2's native PBR renderer + envmap at the SAME cameras, so the
only difference is the edit.  No Blender, no decode (after already rendered).

    CUDA_VISIBLE_DEVICES=3 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/standalone/unified_pbr_demo.py \
      --run data/Pxform_v2/_rerun_v2/08_ss_alignt1 --shard 08 \
      --out data/Pxform_v2/_scratch/unified_pbr
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def _tile(frames, cols=2, bg=20):
    rows = (len(frames) + cols - 1) // cols
    h, w, _ = frames[0].shape
    c = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, f in enumerate(frames):
        r, cc = divmod(i, cols); c[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = f
    return c


def _label(img, text, color):
    import cv2
    o = img.copy(); cv2.rectangle(o, (0, 0), (o.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(o, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shard", default="08")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--hdri", default=f"{TRELLIS2_DIR}/assets/hdri/forest.exr")
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/unified_pbr")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    import cv2, torch
    from PIL import Image
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR

    hdr = cv2.cvtColor(cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))

    run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edits = sorted(p.parent for p in run.glob("*/*/after_shaded.png"))
    print(f"[unified-pbr] {len(edits)} edits")

    before_cache: dict[str, np.ndarray] = {}
    pairs = []
    for ed in edits:
        obj = ed.parent.name; edit = ed.name
        mesh_npz = Path(args.mesh_root) / args.shard / f"{obj}.npz"
        if not mesh_npz.is_file():
            continue
        if obj not in before_cache:
            mesh = OVR.glb_to_pbr_mesh(mesh_npz)
            snap = render_utils.render_snapshot(
                mesh, resolution=args.res, r=2.0, fov=40.0, nviews=4, envmap=envmap)
            shaded = snap["shaded"] if "shaded" in snap else next(iter(snap.values()))
            before_cache[obj] = _tile(shaded, cols=2)
        before = before_cache[obj]
        after = cv2.cvtColor(cv2.imread(str(ed / "after_shaded.png")), cv2.COLOR_BGR2RGB)
        if after.shape[:2] != before.shape[:2]:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        pair = np.concatenate([
            _label(before, f"BEFORE PBR  {obj[:8]}", (60, 230, 60)),
            _label(after, f"AFTER PBR  {edit}", (230, 160, 60))], axis=0)
        cv2.imwrite(str(out / f"{edit}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
        pairs.append(pair); print(f"  {edit}")

    if pairs:
        W = max(p.shape[1] for p in pairs); tiles = []
        for p in pairs:
            if p.shape[1] < W:
                p = np.concatenate([p, np.full((p.shape[0], W - p.shape[1], 3), 255, np.uint8)], 1)
            tiles.append(p); tiles.append(np.full((10, W, 3), 255, np.uint8))
        cv2.imwrite(str(out / "ALL_unified_pbr.png"),
                    cv2.cvtColor(np.concatenate(tiles[:-1], 0), cv2.COLOR_RGB2BGR))
        print(f"[unified-pbr] wrote {out/'ALL_unified_pbr.png'}")


if __name__ == "__main__":
    main()
