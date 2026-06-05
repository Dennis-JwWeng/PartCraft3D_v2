"""Self-contained HTML A/B report at the **condition viewpoint**.

For each sampled edit, everything is shown from the edit's condition view
(spec.view_name = the best_view FLUX was conditioned on), and the FLUX 2D
condition pair is shown alongside:

  COND in (flux input) | COND edited (flux target) |
  BEFORE 3D | A after(mesh) | A SS-voxel | B after(mesh) | B SS-voxel

  A = _exp_flowedit_free_r1024    (FlowEdit S1 + free S2)
  B = _exp_masked_posthoc_r1024 (masked contact-soft S1 + ss_align_t1 + posthoc S2)
SS-voxel = coords_new (64³ occupancy) rendered as grey voxels at the cond view.
All images base64-embedded into one .html.  Samples 20 edits round-robin.

    CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/ab_ss_html.py
"""
from __future__ import annotations
import sys, base64, io
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/mnt/zsn/3dobject/TRELLIS.2")
A = ROOT / "data/Pxform_v2/_exp_flowedit_free_r1024/objects/08"
B = ROOT / "data/Pxform_v2/_exp_masked_posthoc_r1024/objects/08"
OUT = ROOT / "data/Pxform_v2/_scratch/ab_compare/compare_ab_ss.html"
N_SAMPLE = 20
RES = 320
DEFAULT_VIEW = "front"


def b64(arr_or_path) -> str:
    import cv2
    from PIL import Image
    if isinstance(arr_or_path, (str, Path)):
        im = cv2.imread(str(arr_or_path))
        if im is None:
            im = np.full((RES, RES, 3), 235, np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = arr_or_path
    buf = io.BytesIO(); Image.fromarray(im).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(src, label, cls="") -> str:
    return f'<div class="col {cls}"><div class="lbl">{label}</div><img src="{b64(src)}"></div>'


def main() -> None:
    import logging; logging.basicConfig(level=logging.WARNING)
    import torch, yaml
    from partcraft.pipeline_v3.run_trellis2 import resolve_root
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.render import ovox_views as ov

    cfg = yaml.safe_load(open(ROOT / "configs/experiments/pipeline_v3_trellis2_flowedit_free_r1024.yaml"))
    root = resolve_root(cfg)

    by_obj: dict[str, list[str]] = {}
    for objdir in sorted(B.glob("*/")):
        o = objdir.name
        common = []
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            eid = ed.name
            if ((ed / "after_view_front.png").is_file()
                    and (A / o / "edits_3d" / eid / "after_view_front.png").is_file()
                    and (ed / "latents/ss.npz").is_file()
                    and (A / o / "edits_3d" / eid / "latents/ss.npz").is_file()):
                common.append(eid)
        if common:
            by_obj[o] = common

    picks: list[tuple[str, str]] = []
    pos = {o: 0 for o in by_obj}
    while len(picks) < N_SAMPLE and any(pos[o] < len(by_obj[o]) for o in by_obj):
        for o in by_obj:
            if pos[o] < len(by_obj[o]):
                picks.append((o, by_obj[o][pos[o]])); pos[o] += 1
                if len(picks) >= N_SAMPLE:
                    break

    # spec map: edit_id -> (label, condition view_name)
    spec: dict[str, tuple[str, str]] = {}
    for o in {o for o, _ in picks}:
        try:
            for s in iter_flux_specs(root.context("08", o)):
                vn = s.view_name if s.view_name in ov.VIEW_ORDER else DEFAULT_VIEW
                spec[s.edit_id] = (f"[{s.edit_type}] {s.prompt}", vn)
        except Exception:
            pass

    def ss_voxel(ss_npz: Path, view: str):
        z = np.load(ss_npz)
        c = z["coords_new"].astype(np.float32)
        pos_t = torch.from_numpy(c / 64.0 - 0.5).cuda()
        attr = torch.full((c.shape[0], 3), 0.62).cuda()
        d = ov.render_voxel_positions(pos_t, attr, 1.0 / 64.0, [view],
                                      resolution=RES, ssaa=2, bg=(1, 1, 1))
        return d[view], int(c.shape[0])

    rows = []
    for o, eid in picks:
        label, view = spec.get(eid, (eid, DEFAULT_VIEW))
        aed = A / o / "edits_3d" / eid
        bed = B / o / "edits_3d" / eid
        e2d = A / o / "edits_2d"
        cond_in = e2d / f"{eid}_input.png"
        cond_ed = e2d / f"{eid}_edited.png"
        before = A / o / "gate_views" / f"before_view_{view}.png"
        a_ss, a_n = ss_voxel(aed / "latents/ss.npz", view)
        b_ss, b_n = ss_voxel(bed / "latents/ss.npz", view)
        cells = (
            img(cond_in, "COND in (2D)", "cond")
            + img(cond_ed, "COND edited (2D)", "cond")
            + img(before, f"BEFORE 3D · {view}")
            + img(aed / f"after_view_{view}.png", "A · FlowEdit+free", "a")
            + img(a_ss, f"A · SS voxel ({a_n})", "a")
            + img(bed / f"after_view_{view}.png", "B · masked+t1+posthoc", "b")
            + img(b_ss, f"B · SS voxel ({b_n})", "b")
        )
        rows.append(f"""
<section>
  <h3>{o[:12]} · {eid.split('_')[-1]} · view=<b>{view}</b> &nbsp; <span class="p">{label}</span></h3>
  <div class="grid">{cells}</div>
</section>""")
        print(f"  {o[:8]} {eid.split('_')[-1]:<4} view={view:<6} A_ss={a_n:<6} B_ss={b_n}")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>A/B 3D-edit @ condition view + SS voxel ({len(rows)} edits)</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f4f6;margin:0;padding:18px;color:#222}}
 h1{{font-size:19px}} h2.legend{{font-size:13px;font-weight:400;color:#555;margin-top:-6px;max-width:1100px}}
 section{{background:#fff;border-radius:8px;padding:12px 14px;margin:14px 0;box-shadow:0 1px 4px #0001}}
 h3{{font-size:14px;margin:0 0 8px}} h3 .p{{font-weight:400;color:#0a6;font-size:13px}}
 .grid{{display:flex;gap:9px;flex-wrap:nowrap;overflow-x:auto;align-items:flex-start}}
 .col{{display:flex;flex-direction:column;align-items:center;border:1px solid #eee;border-radius:6px;padding:4px;background:#fafafa;flex:0 0 auto}}
 .col.cond{{background:#eef3fb;border-color:#c9d8ef}}
 .col.a{{background:#fff7f0;border-color:#f3d9bf}} .col.b{{background:#f0faf2;border-color:#bfe6c8}}
 .col img{{width:180px;height:180px;object-fit:contain;display:block;background:#fff}}
 .lbl{{font-size:11px;font-weight:600;margin-bottom:3px;color:#444;text-align:center;max-width:180px}}
</style></head><body>
<h1>A/B 3D-edit at the condition viewpoint + after SS decoded voxel — {len(rows)} edits</h1>
<h2 class="legend">Each row is rendered at that edit's <b>condition view</b> (the best_view FLUX saw). Blue = the FLUX 2D condition pair (input → edited target). A = FlowEdit S1(no mask)+free S2; B = masked contact-soft S1 + ss_align_t1 + posthoc S2. SS voxel = coords_new (64³ occupancy) as grey voxels; (N) = voxel count.</h2>
{''.join(rows)}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"\nwrote {OUT}  ({len(rows)} edits, {OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
