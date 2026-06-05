# Runbook — TRELLIS.1-SS native (in-process) masked 3D edit @512

最后更新:2026-06-04 · 分支 `experiment/flowedit-s1`

把「TRELLIS.1 SS mask 编辑」做成 trellis2 进程内的正式 S1 阶段(不再走 vinedresser3d 离线桥接)。
单 conda 环境(`trellis2`)、单代码库(当前目录)、单次模型常驻即可端到端跑完带纹理的 3D 编辑。

## ⭐ 当前最优方案 = 三件事

| 阶段 | 方案 | 开关 |
|------|------|------|
| **S1 结构** | TRELLIS.1 SS flow + DINOv2(进程内)masked，pad=4 膨胀 + 同帧 restore | `trellis2_s1_ss_model: t1`、`s1_pad: 4`、`s2_restore_preserved: true` |
| **S2 shape** | masked **per-step** 锚定(实心编辑部件) | `trellis2_s2_anchor_mode: perstep` |
| **S2 texture** | **posthoc-restore**：编辑图下自由生成 → 终点把 P1 编码的**原始 tex latent** 硬贴回保留 token | `trellis2_s2_tex_anchor_mode: posthoc`（有 before-tex 边车自动走 restore） |

**这三个文件就是你要的:**

| | 文件 |
|---|------|
| 🟢 **管线脚本** | `run_pipeline_v3_shard_trellis2.sh`（内部 dispatch `python -m partcraft.pipeline_v3.run_trellis2`） |
| 🟢 **配置(最优)** | `configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml` |
| 🟢 **runbook** | `docs/experiment/t1ss_native_runbook.md`（本文件） |

---

## 1. 启动方式(回答:用哪个脚本)

两个入口都能用,用途不同:

| 入口 | 用途 | 何时用 |
|------|------|--------|
| **`run_pipeline_v3_shard_trellis2.sh`** | 多物体、多卡并行、可选 VLM/FLUX/gate 的完整分片管线;内部 dispatch `python -m partcraft.pipeline_v3.run_trellis2` | **正式跑批 / 复现实验** |
| `run_pipeline_minimal.py` | 单物体、CLI 给定 edit、跳过 VLM 与 gate | 单物体快速 debug / 看一条 edit |

### 正式跑最优方案(8 卡里用 0,1 两卡)

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2

# 1) 先 seed 输出树(从已有 native 模板克隆 edits_2d / p1_encode / phase1 / edit_status)
bash scripts/experiments/seed_masked_e512_variant.sh \
     data/Pxform_v2/_exp_t1ss_native_r512_pad2_restore \
     data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore

# 2) 跑 3D 编辑阶段(trellis2_preview 只跑 trellis2_3d,前面 2D/gate 已 seed 好)
SHARD=08 \
OBJ_IDS_FILE=data/Pxform_v2/_exp_masked_perstep_r512_pad0/seeded_ids.txt \
FORCE=1 \
STAGES=trellis2_preview,gate_quality \
MACHINE_ENV=configs/machine/local_trellis2.env \
PIPELINE_GPUS="0,1" \
  bash run_pipeline_v3_shard_trellis2.sh shard08_t1native_texrestore \
       configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml \
  > data/Pxform_v2/_scratch/t1native_texrestore.log 2>&1 &
```

- `PIPELINE_GPUS="0,1"` → 脚本把物体在两卡上 round-robin(`--single-gpu --gpu-shard 0/2` 和 `1/2`),真正双卡并行。
- `STAGES=trellis2_preview,gate_quality`:`trellis2_3d` 跑 3D 编辑并渲染 gate-E before/after 视图,`gate_quality`(Gate E)再起一批 VLM 判每条编辑的视觉质量,写 `gate_e: pass/fail` 进 `edit_status.json`(`stages.gate_e` + `gates.E.vlm`)。只要 latents+视图、不要 VLM 判分就去掉 `,gate_quality`;要从头(encode+VLM+FLUX+gate)则整个去掉 `STAGES`。
  - Gate E 读 `gate_views/before_view_*` 和 `edits_3d/<id>/after_view_*`(都由 `trellis2_preview` 在 `trellis2_render_gate_views: true` 下产出);没有 after 视图的编辑类型(本配方 `qc.edit_types: [modification, scale]` 之外的)会判 `missing_previews=fail`,属正常。
- `FORCE=1`:无视已完成状态重跑。
- `OBJ_IDS_FILE`:限定物体集合;不给则 `--all`。

### 环境变量速查(shard 脚本)

| 变量 | 含义 |
|------|------|
| `SHARD` | 分片号(如 `08`);也可让 tag `shard08_xxx` 自动推断 |
| `OBJ_IDS_FILE` | 只跑文件里列出的 obj id;否则全跑 |
| `STAGES` | 逗号分隔的阶段子集 |
| `FORCE=1` | 重跑已完成步骤 |
| `PIPELINE_GPUS` | 用哪些卡(逗号分隔),决定并行度 |
| `MACHINE_ENV` | 机器 env 文件,默认 `configs/machine/local_trellis2.env` |

### 单物体 debug(可选)

```bash
TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 CUDA_VISIBLE_DEVICES=0 \
python run_pipeline_minimal.py --shard 08 --obj-id <obj_id> \
    --edit-type scale --part-id 1 --instruction "..."
```
> 注:minimal 入口走的是同一套 `_build_p4_mesh`,但**不读 config 里的 `trellis2_s1_ss_model` 等开关**,
> 默认 T2 S1。要测 T1-native 用上面的 shard 脚本 + config。

---

## 2. 当前确认的参数(最优 config)

`configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml`,`services.image_edit` 段关键开关:

| key | 值 | 含义 |
|-----|----|------|
| `trellis2_edit_res` | `512` | 编辑分辨率;S1 仍 64³,S2 SLat 降到 32³(`res//16`),写 `shape/tex_slat_e512.npz` 边车 |
| `trellis2_s1_mode` | `masked` | masked SS 编辑(occ→enc→原图反演→编辑图 masked 重绘→dec) |
| **`trellis2_s1_ss_model`** | **`t1`** | **S1 用 TRELLIS.1 SS flow + DINOv2(进程内 native);默认 `t2`** |
| `trellis2_s1_pad` | `4` | 64³ edit grid 切比雪夫膨胀 4 |
| `trellis2_s1_contact_soft` | `true` | 接触边界软 mask |
| `trellis2_ss_align_t1` | `true` | S1 调度对齐 T1(steps25/cfg5/interval[.5,1]/rt3) |
| `trellis2_s2_restore_preserved` | `true` | S1 在 mask 外丢掉的源 body 体素同帧 64³ 补回 |
| **`trellis2_s2_anchor_mode`** | **`perstep`** | **S2 SHAPE：masked 逐步锚定(实心编辑部件)** |
| **`trellis2_s2_tex_anchor_mode`** | **`posthoc`** | **S2 TEXTURE：posthoc-restore(见下);默认沿用 shape 的模式** |
| `force_white_model` | *(无)* | 不跳纹理 → 带纹理 512 mesh decode |
| `trellis2_emit_glb` | `false` | 默认关 GLB(省 4096 纹理烘焙);只要 latents + after-views。要 GLB 改 `true` |
| `trellis2_seed` | `1` | |
| `trellis2_texture_size` | `4096` | |

> **纹理 = restore（核心）**：`trellis2_s2_tex_anchor_mode: posthoc` + 存在 `tex_slat_e512.npz` 边车时,
> `masked_tex_slat` 走 **posthoc-restore** 分支:编辑图下自由生成整张纹理 → 终点把 **P1 编码的原始 tex latent**
> 通过 `build_coord_bridge` 的 `src_idx` **硬贴回**保留 token(等价 S1 的 `restore_preserved`)。保留区 decode
> 出**逐像素原始材质**,只有编辑区重画。日志出现 `tex anchor_mode=posthoc-restore (... N/M preserved tokens)`。
> 关键前提:shape 与 tex 编码坐标实测**逐位一致**,所以 shape 的桥接 `src_idx` 能直接索引编码 tex。
>
> 其它纹理模式(对照):`perstep` = 每步锚到重采样参考(保留区会随编辑图全局条件**漂色**);
> `posthoc`(无 before-tex 时) = 贴回重采样的 `tex0`(原图条件,挡全局漂但仍是重画);`free` = 完全不 mask。

### 对照配置

- `..._pad4_full.yaml`：tex 用 perstep(漂色基线),`emit_glb: true`。
- `..._pad4_texposthoc.yaml`：tex 用 posthoc 但贴重采样 `tex0`(非编码真值)。
- `..._pad2_restore.yaml`：`force_white_model: true` 白模 + `s1_pad: 2`,做 S1 几何 parity(vs 桥接 IoU≈0.96)。

---

## 3. 实现要点(进程内 native 怎么搭的)

- **`partcraft/pipeline_v3/trellis1_ss.py`**(新增):T1-SS 的加载器 + 条件,缓存在 pipeline 对象上。
  - `load_t1_ss_flow`:把 T1 `ss_flow_img_dit_L_16l8_fp16` 权重载进 trellis2 自己的
    `SparseStructureFlowModel`(`missing=0 unexpected=0`)→ `.cuda()` → `convert_to(torch.float16)`
    (blocks 走 flash_attn 半精度,input/t_embedder 留 fp32,模型 `manual_cast` 处理)。默认 ckpt
    `/mnt/zsn/ckpts/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16`,可用 `trellis2_t1_ss_flow` 覆盖。
  - `load_t1_dino`:bundled `third_party/encode_asset/dinov2_hub.load_dinov2_vitl14_reg`(离线,**不 import trellis**)。
  - `t1_preprocess`(rembg 抠图→1.2×bbox 裁→518→premultiply;rembg session 缓存,缺失则白底阈值回退) +
    `t1_get_cond`(518→imagenet norm→`dino(x,is_training=True)['x_prenorm']`→layer_norm;neg=zeros)。
- **`trellis2_structure.py:edit_structure`**:加 `ss_flow=None` 入参;`ss_flow=None` 时回落到
  `pipeline.models["sparse_structure_flow_model"]`(T2)。T1/T2 共用同一条 masked 路径。
- **`trellis2_3d.py`**:masked S1 分支读 `trellis2_s1_ss_model`;为 `t1` 时载 T1 flow+dino+rembg,
  用原始 PIL(非 `pipeline.preprocess_image`)算 `c_orig/c_edit`,强制 T1 调度,调
  `edit_structure(..., ss_flow=t1_flow)`。下游 restore / `masked_shape_slat` / `masked_tex_slat` 原样复用。
- **纹理 restore(新)**:
  - `_load_p1_slat(ctx, res, which="tex")` 读 P1 编码的原始 tex 边车 `tex_slat_e512.npz`,
    在 `run_for_object` 里得 `p1_tex_s2`,经 `_build_p4_mesh` 透传到 `masked_tex_slat(before_tex_denorm=...)`。
  - `masked_tex_slat` 新增 `before_tex_denorm` 入参 + **posthoc-restore 分支**:`anchor_mode=="posthoc"` 且
    传入 before-tex 时,自由生成后 `new_feats[preserved] = before_norm[src_idx]`(跳过参考重采样 + 反演,更快)。
    行数不匹配时 warn → 回退到重采样 posthoc(不崩)。
  - `trellis2_s2_tex_anchor_mode` 让纹理独立于 shape 选锚定方式(`trellis2_3d.py`,默认 = shape 模式)。
- 显存:T1 flow(556M)+ DINOv2-L 与 T2 3B 同进程常驻,增量 ~3–4GB,143GB 卡无压力。

> **可复现性注意**:每次 `FORCE` 跑是从头重算 S1+shape+tex(不复用旧几何)。S1 占据(coords)逐位确定;
> 但 shape latent 数值在独立进程间**不是 bit 级一致**(fp16/flash_attn 噪声,meanΔ≈1.3、maxΔ≈30)。
> 故不同 tex 变体严格说几何略有差异。要做**干净的纹理 A/B**(同一几何),后续可加「复用已存
> coords_new+shape_slat、只重跑 tex」的路径 —— 尚未实现。

---

## 4. 验证状态

| 层级 | 结果 |
|------|------|
| S1 占据 parity(native vs 离线桥接) | IoU≈0.96(装 rembg 后) |
| 全量 13 物体 pad2 回归 | 32/32,中位 \|Δ\|≈12 体素(~1%),无失败 |
| pad4 带纹理全量(perstep tex) | 29/29 完成,`white_model=False`,`tex=(N,32)` 实跑 |
| tex posthoc-restore | 3 物体/9 edit 验证通过,全走 `posthoc-restore` 分支,保留区贴编码真值 |

---

## 5. 产物路径

```
data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore/objects/08/<obj>/
  edits_2d/<eid>_input.png, <eid>_edited.png     # seed 来的 2D 条件
  gate_views/before_view_*.png                    # decode 原始 latent 的 before
  edits_3d/<eid>/
    latents/  (ss.npz, shape_slat.npz, tex_slat.npz)
    after_view_{front,right,back,left,down}.png    # GLB 默认关;要 after.glb 把 emit_glb 改 true
```
日志:`data/Pxform_v2/_scratch/t1native_texrestore.log`
对比可视化:`scripts/viz/ab_tex_perstep_vs_posthoc_html.py` →
`data/Pxform_v2/_scratch/ab_compare/tex_perstep_vs_posthoc_vs_restore.html`(perstep / 重采样 posthoc / restore 三路)

---

## 6. 离线桥接(已弃用,保留作 A/B)

旧路径:`prep.py`(trellis2)→ `run_t1.py`(**vinedresser3d** 环境,import 旧仓库 trellis 包)→ repack →
config 设 `trellis2_ss1_coords_dir` 注入 `coords_new`。native 路径上线后不再需要,`ss1_coords_dir` 分支仍保留
做回退/对照。详见记忆 `t1-ss-mask-bridge`。
