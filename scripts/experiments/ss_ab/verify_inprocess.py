#!/usr/bin/env python3
"""VERIFY in-process TRELLIS.1-SS — reproduce the offline bridge's S1 occupancy
INSIDE the trellis2 env, with NO `import trellis` (avoids kaolin) and NO
vinedresser3d / old repo.  Loads:

  * SS VAE enc/dec  ← trellis2.models.from_pretrained (shared, T1=T2 weights)
  * T1 SS flow      ← trellis2.models.SparseStructureFlowModel + T1 ckpt
  * DINOv2-L cond   ← bundled encode_asset/dinov2_hub (offline)

then runs run_t1's EXACT masked RF recipe (inversion under orig img → masked
forward repaint under edited img) on a chosen edit and compares coords_new to
the bridge output ss1/<obj>/<eid>/ss1_coords.npz (IoU + counts).

    CUDA_VISIBLE_DEVICES=1 /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/experiments/ss_ab/verify_inprocess.py --io data/Pxform_v2/_scratch/ss_ab_t1t2_pad3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
T2_CODEBASE = "/mnt/zsn/3dobject/TRELLIS.2"
T1_CKPT_DIR = "/mnt/zsn/ckpts/TRELLIS-image-large/ckpts"
T1_SS_FLOW = f"{T1_CKPT_DIR}/ss_flow_img_dit_L_16l8_fp16"
SS_ENC = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
SS_DEC = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
STEPS, CFG, CFG_INTERVAL, RESCALE_T = 25, 5.0, (0.5, 1.0), 3.0


# ── run_t1's exact RF sampler (ported verbatim) ──────────────────────────────
def get_times(steps, rescale_t, int_len, num_iter, inverse):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_seq = t_seq[::-1]
    t_new = []
    for i in range(0, steps + 1, int_len):
        interval = t_seq[i:min(i + int_len, steps + 1)]
        if len(interval) == 1:
            t_new.extend(interval); continue
        for cnt in range(num_iter):
            t_new.extend(interval)
            if cnt < num_iter - 1:
                t_new.extend(interval[::-1][1:-1])
    t_seq = np.array(t_new[::-1])
    if inverse:
        t_seq = t_seq[::-1]
    return list((t_seq[i], t_seq[i + 1]) for i in range(steps))


def _infer(model, x_t, t, cond, **kw):
    import torch
    tt = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
    if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
        cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
    return model(x_t, tt, cond, **kw)


def sample_once(model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval):
    if cfg_interval[0] <= t <= cfg_interval[1]:
        pred = _infer(model, x_t, t, cond)
        neg = _infer(model, x_t, t, neg_cond)
        return (1 + cfg_strength) * pred - cfg_strength * neg
    return _infer(model, x_t, t, cond)


def rf_sample_once(model, x_t, t_curr, t_prev, **kw):
    pred = sample_once(model, x_t, t_curr, **kw)
    mid = x_t + (t_prev - t_curr) / 2 * pred
    pred_mid = sample_once(model, mid, (t_curr + t_prev) / 2, **kw)
    first = (pred_mid - pred) / ((t_prev - t_curr) / 2)
    return x_t + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first


# ── T1 image cond (ported from trellis_image_to_3d.py) ──────────────────────
def preprocess_image(input_img, rembg_session):
    from PIL import Image
    has_alpha = input_img.mode == 'RGBA' and not np.all(np.array(input_img)[:, :, 3] == 255)
    if has_alpha:
        output = input_img
    elif rembg_session is not None:
        import rembg
        inp = input_img.convert('RGB')
        scale = min(1, 1024 / max(inp.size))
        if scale < 1:
            inp = inp.resize((int(inp.width * scale), int(inp.height * scale)), Image.Resampling.LANCZOS)
        output = rembg.remove(inp, session=rembg_session)
    else:
        # white-bg fallback (these are clean white-bg renders): alpha = non-white px
        rgb = np.array(input_img.convert('RGB'))
        a = (~np.all(rgb >= 248, axis=-1)).astype(np.uint8) * 255
        output = Image.fromarray(np.dstack([rgb, a]), mode='RGBA')
    arr = np.array(output); alpha = arr[:, :, 3]
    bb = np.argwhere(alpha > 0.8 * 255)
    x0, y0, x1, y1 = np.min(bb[:, 1]), np.min(bb[:, 0]), np.max(bb[:, 1]), np.max(bb[:, 0])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sz = int(max(x1 - x0, y1 - y0) * 1.2)
    output = output.crop((cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2))
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    o = np.array(output).astype(np.float32) / 255
    o = o[:, :, :3] * o[:, :, 3:4]
    return Image.fromarray((o * 255).astype(np.uint8))


def get_cond(dino, img_pil, dev):
    import torch
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = img_pil.resize((518, 518)).convert('RGB')
    x = torch.from_numpy(np.array(im).astype(np.float32) / 255).permute(2, 0, 1)[None]
    x = norm(x).to(dev)
    feats = dino(x, is_training=True)['x_prenorm']
    cond = torch.nn.functional.layer_norm(feats, feats.shape[-1:])
    return {"cond": cond, "neg_cond": torch.zeros_like(cond)}


def iou(a, b):
    sa = {tuple(r) for r in a.tolist()}; sb = {tuple(r) for r in b.tolist()}
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default="data/Pxform_v2/_scratch/ss_ab_t1t2_pad3")
    ap.add_argument("--only", default=None, help="substring filter; default = first edit")
    args = ap.parse_args()

    sys.path.insert(0, T2_CODEBASE)
    sys.path.insert(0, str(ROOT / "third_party/encode_asset"))
    sys.path.insert(0, str(ROOT / "third_party/dinov2"))
    import torch
    torch.set_grad_enabled(False)
    from PIL import Image
    import trellis2.models as t2m
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from safetensors.torch import load_file
    from dinov2_hub import load_dinov2_vitl14_reg
    dev = "cuda"
    try:
        import rembg
        rsess = rembg.new_session('u2net')
        print("[load] rembg session (matches bridge preprocess exactly)")
    except Exception as e:
        rsess = None
        print(f"[load] rembg unavailable ({type(e).__name__}); white-bg alpha fallback")

    print("[load] SS enc/dec (trellis2.models, shared VAE)")
    ss_enc = t2m.from_pretrained(SS_ENC).eval().cuda()
    ss_dec = t2m.from_pretrained(SS_DEC).eval().cuda()
    print("[load] T1 SS flow via trellis2 SparseStructureFlowModel")
    fargs = json.load(open(T1_SS_FLOW + ".json"))["args"]
    ss_flow = SparseStructureFlowModel(**fargs)
    msg = ss_flow.load_state_dict(load_file(T1_SS_FLOW + ".safetensors"), strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, msg
    ss_flow = ss_flow.eval().to(dev)
    ss_flow.convert_to(torch.float16)   # blocks→fp16 (flash_attn); input/t_embedder stay fp32
    print("[load] DINOv2-L (bundled, offline)")
    dino = load_dinov2_vitl14_reg(pretrained=True).eval().to(dev)

    io = ROOT / args.io
    inputs = sorted((io / "inputs").glob("*/*.npz"))
    if args.only:
        inputs = [p for p in inputs if args.only in str(p)]
    p = inputs[0]
    obj, eid = p.parent.name, p.stem
    d = np.load(p, allow_pickle=True)
    coords0 = d["coords0"].astype("int64")
    keep16 = torch.from_numpy(d["keep16"].astype(bool)).to(dev)
    keep = keep16[None, None].float().expand(1, 8, 16, 16, 16)
    orig = preprocess_image(Image.open(str(d["input_png"])), rsess)
    edit = preprocess_image(Image.open(str(d["edited_png"])), rsess)
    c_orig = get_cond(dino, orig, dev)
    c_edit = get_cond(dino, edit, dev)

    occ = torch.zeros(1, 1, 64, 64, 64, device=dev)
    occ[0, 0, coords0[:, 0], coords0[:, 1], coords0[:, 2]] = 1.0
    z_s0 = ss_enc(occ.float())

    # RF inversion under ORIGINAL image (cfg off)
    sample = z_s0; inv = {}
    for tc, tp in get_times(STEPS, RESCALE_T, 1, 1, True):
        sample = rf_sample_once(ss_flow, sample, tc, tp, cond=c_orig["cond"],
                                neg_cond=c_orig["neg_cond"], cfg_strength=0.0,
                                cfg_interval=(0.0, 1.0))
        inv[round(float(tp), 6)] = sample
    # masked forward repaint under EDITED image; anchor keep region to inv
    sample = sample
    for tc, tp in get_times(STEPS, RESCALE_T, 1, 1, False):
        x = rf_sample_once(ss_flow, sample, tc, tp, cond=c_edit["cond"],
                           neg_cond=c_edit["neg_cond"], cfg_strength=CFG,
                           cfg_interval=CFG_INTERVAL)
        f = inv.get(round(float(tp), 6))
        if f is not None:
            x = x * (1.0 - keep) + f * keep
        sample = x
    dec = ss_dec(sample.float()) > 0
    cn = torch.argwhere(dec)[:, 2:5].int().cpu().numpy().astype(np.int32)

    bridge_p = io / "ss1" / obj / eid / "ss1_coords.npz"
    print(f"\n=== {obj}/{eid} ===")
    print(f"in-process coords_new : {cn.shape[0]} voxels  (occ0={coords0.shape[0]})")
    if bridge_p.is_file():
        br = np.load(bridge_p)["coords"].astype(np.int32)
        print(f"bridge    coords_new : {br.shape[0]} voxels")
        print(f"IoU(in-process, bridge) = {iou(cn, br):.4f}   "
              f"count Δ = {cn.shape[0]-br.shape[0]:+d}")
    else:
        print(f"(no bridge file at {bridge_p})")


if __name__ == "__main__":
    main()
