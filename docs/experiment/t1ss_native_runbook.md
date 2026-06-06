# Runbook — TRELLIS.1-SS native (in-process) masked 3D edit @512

最后更新:2026-06-06 · 分支 `main` · 生产用 **posthoc_no2dqc**(见下「⭐⭐」节);texrestore 仍作对照

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

## ⭐⭐ 生产配置(no-2DQC + 全 posthoc + PBR 渲染)— 当前正式跑用这个

`configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml`,与上面的 texrestore 并列。三处关键差异 + 一套统一渲染:

| 维度 | texrestore | **posthoc_no2dqc(本)** |
|---|---|---|
| **S2 SHAPE** | `perstep`(跑 flow inversion) | **`posthoc`** — 自由生成实心部件 + 硬贴 P1 编码 body latent;**跳过 invert_clean**(实测 ~1.22×/edit、编辑区更实、body 逐位精确) |
| **S2 TEXTURE** | `posthoc`-restore | 同(本就不反演)→ **S2 全程零 flow inversion** |
| **2D QC** | 有 `gate_2d`(Gate C) | **删除** — `build_prereq_map()` 自动把 `trellis2_3d` 的 prereq 从 `gate_c` 回退到 `gate_a`,每个 FLUX 2D 编辑都进 3D |
| **输出根** | `data/Pxform_v2/prod` | **`data/Pxform_v2/prod_posthoc_no2dqc`** |

**渲染(唯一一套,VLM 全程 PBR;旧的混杂渲染已删):**
- before/after/gate-E:`decode_latent` + `render_sample`(PbrMeshRenderer)= `_render_before/after_named_views`。
- **encode 阶段就把 `gate_views/before_view_*` 渲好**(PBR)→ flux 直接读图,**不再每线程实时 o-voxel 渲染**(那条会 segfault;毒 mesh 会撞 CUDA 700 级联)。开关 `trellis2_encode_render_overview: true`。
- gate-A overview = 上行 **PBR RGB**(复用 gate_views)+ 下行 **o-voxel seg**(robust,`render_overview_from_ovox(skip_rgb=True)`;原始-mesh seg 光栅化才会崩,体素化 seg 不崩)。
- 相机 `partcraft/render/ovox_views.py:NAMED_VIEWS`:front/right/back/left = **0/90/180/270 轴对齐 + 俯视 22°**(正对面,不再 ~45° 偏置的 3/4 侧身);`down` 保留。
- 已删的死渲染:`_render_overview_at_encode`(render_pbr_overview 的 seg→CUDA 崩)、`prepare_input_image_ovox`(flux o-voxel)、`select_best_view`。

**正式跑(GPU 2-7,tmux):**
```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2
mkdir -p data/Pxform_v2/prod_posthoc_no2dqc
tmux new -s prod00 -d
tmux send-keys -t prod00 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 S1_PER_GPU_CONCURRENCY=4 SHARD=00 MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="2,3,4,5,6,7" bash run_pipeline_v3_shard_trellis2.sh posthoc_shard00 configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml 2>&1 | tee data/Pxform_v2/prod_posthoc_no2dqc/_run_shard00.log' Enter
# 看进度: tmux attach -t prod00   （脱离: Ctrl-b 然后 d）
```
> `OMP_NUM_THREADS=1` 等:flux_2d 的 16 线程客户端避免原生库线程超额订阅崩溃的兜底。
> 续跑只补缺失(encode 跳过已编码 latent,只补渲缺的 gate_views/overview);全 shard 改 `SHARD`/tag 即可。

### ⚡ 提速(两个批处理杠杆,默认已开 — 144GB 卡)

两处默认值原本为「小显存」调优,在这批 144GB 卡上白白浪费算力。已在代码里改成高吞吐默认,跑批直接带上即可。**注意:慢的根因不是 thinking 模式**(`enable_thinking:False` 在 Qwen3.6 上已生效,采样到的 raw.txt 0/159 含 `<think>`),而是下面两点没批处理 + 反复搬权重。

| 杠杆 | 问题 | 改法 | 实测 |
|---|---|---|---|
| **gate_a / s1 每卡并发** | `gen_edits_image.py` 每个 VLM server 只有 1 个 consumer + `Semaphore(1)` → sglang `#running-req:1`、96% 利用率却只 ~35 req/min | env **`S1_PER_GPU_CONCURRENCY`(默认4)**:每卡起 N 个 consumer + `Semaphore(N)` + 哨兵×N;内联 gate_a 候选判别也跟着并发 → `#running-req` 16–21 | **35→261 req/min(~7.5×)**,gate_a ~5h→~40min |
| **3D 编辑模型驻留** | `Trellis2ImageTo3DPipeline` 默认 `low_vram=True` → `cuda()` 几乎空操作,每对象每 stage 把 SS/shape/shape_lr/tex flow + decoder + DINOv3 在 CPU↔GPU 来回搬(纯 PCIe 浪费) | `trellis2_3d._ensure_pipeline` 读 `services.image_edit.low_vram`(**默认 `False`**)→ `cuda()` 前置 `pipeline.low_vram=False` → 全模型常驻 | 采样零搬运;`trellis2_preview` 是 `servers:none`,整张 144GB 给 TRELLIS,绰绰有余 |

- 不爆显存:sglang KV 池由 `--mem-fraction-static` 静态预分配,并发超了**排队**不 OOM。
- phase1 `max_tokens` 12288→4096(实测最长 2562 tok,纯省调度)。
- 内存紧的机器回退:`S1_PER_GPU_CONCURRENCY=1` + config 里 `services.image_edit.low_vram: true`。
- 细节见记忆 `v2-throughput-levers.md`。

---

## 1. 启动方式(回答:用哪个脚本)

两个入口都能用,用途不同:

| 入口 | 用途 | 何时用 |
|------|------|--------|
| **`run_pipeline_v3_shard_trellis2.sh`** | 多物体、多卡并行、可选 VLM/FLUX/gate 的完整分片管线;内部 dispatch `python -m partcraft.pipeline_v3.run_trellis2` | **正式跑批 / 复现实验** |
| `run_pipeline_minimal.py` | 单物体、CLI 给定 edit、跳过 VLM 与 gate | 单物体快速 debug / 看一条 edit |

### 配置位置(已定稿)

- **唯一正式配置** = `configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml`(顶层)。
- 其余所有配方都已挪到 `configs/experiments/`,只作对照/复现,不用于正式跑批。
- **输出根** = `data/Pxform_v2/prod`,每个 shard 落在 `prod/objects/<shard>/<obj_id>/`(见 §产物路径)。

### 正式跑批 —— 单 shard 端到端(全 8 卡)

从头跑完整管线(encode → VLM 出 edit + gate_a → FLUX 2D → gate_c → 3D 编辑 → Gate-E):

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2

SHARD=08 \
MACHINE_ENV=configs/machine/local_trellis2.env \
PIPELINE_GPUS="0,1,2,3,4,5,6,7" \
  bash run_pipeline_v3_shard_trellis2.sh prod_shard08 \
       configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml \
  > data/Pxform_v2/prod/_run_shard08.log 2>&1 &
```

### 正式跑批 —— 全量 10 个 shard(00..09)

每个 shard 一次调用,全部落进同一棵 `data/Pxform_v2/prod` 树:

```bash
for S in 00 01 02 03 04 05 06 07 08 09; do
  SHARD=$S MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0,1,2,3,4,5,6,7" \
    bash run_pipeline_v3_shard_trellis2.sh prod_shard$S \
         configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_texrestore.yaml \
    > data/Pxform_v2/prod/_run_shard$S.log 2>&1
done
```

> 实验式快跑(复用 sibling 树、只重跑 3D 编辑)仍可用 `STAGES=trellis2_preview,gate_quality`
> + `OBJ_IDS_FILE=...` + `FORCE=1` 指向某棵 `_exp_*` 树;见 `data/Pxform_v2/README.md`。

- `PIPELINE_GPUS="0,1,...,7"` → 脚本把该 shard 的物体在这些卡上 round-robin(`--single-gpu --gpu-shard k/N`),真正多卡并行。
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

### 对照配置(都在 `configs/experiments/` 下,只作 A/B)

- `configs/experiments/..._t1ss_native_r512_pad4_full.yaml`:tex 用 perstep(漂色基线),`emit_glb: true`。
- `configs/experiments/..._t1ss_native_r512_pad4_texposthoc.yaml`:tex 用 posthoc 但贴重采样 `tex0`(非编码真值)。
- `configs/experiments/..._t1ss_native_r512_pad2_restore.yaml`:`force_white_model: true` 白模 + `s1_pad: 2`,做 S1 几何 parity(vs 桥接 IoU≈0.96)。

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

正式跑批的输出根是 `data/Pxform_v2/prod`,每个 shard 落在 `objects/<shard>/`:

```
data/Pxform_v2/prod/objects/<shard>/<obj>/
  p1_encode/  (ss.npz, shape_slat.npz, tex_slat.npz, *_slat_e512.npz)   # P1 编码 latent
  edits_2d/<eid>_input.png, <eid>_edited.png     # FLUX 2D 编辑前/后图
  gate_views/before_view_*.png                    # decode 原始 latent 的 before
  edits_3d/<eid>/
    latents/  (ss.npz, shape_slat.npz, tex_slat.npz)
    after_view_{front,right,back,left,down}.png    # GLB 默认关;要 after.glb 把 emit_glb 改 true
  edit_status.json, qc.json                        # 各 gate(A/C/E)判定
```
每个 shard 的运行日志:`data/Pxform_v2/prod/_run_shard<NN>.log`
对比可视化:`scripts/viz/ab_tex_perstep_vs_posthoc_html.py` →
`data/Pxform_v2/_scratch/ab_compare/tex_perstep_vs_posthoc_vs_restore.html`(perstep / 重采样 posthoc / restore 三路)

---

## 6. 离线桥接(已弃用,保留作 A/B)

旧路径:`prep.py`(trellis2)→ `run_t1.py`(**vinedresser3d** 环境,import 旧仓库 trellis 包)→ repack →
config 设 `trellis2_ss1_coords_dir` 注入 `coords_new`。native 路径上线后不再需要,`ss1_coords_dir` 分支仍保留
做回退/对照。详见记忆 `t1-ss-mask-bridge`。
