# PartCraft3D v2 — 架构文档 (ARCH.md)

> 当前架构:**基于 TRELLIS.2 的 prompt 驱动、part 级 3D 编辑数据管线**。
> 输入 PartVerse 对象 + 编辑指令,输出"编辑前/后" 3D 资产对 (`.glb`)。
> 核心是 **Vinedresser3D 式的 latent 掩码编辑**:保留 mask 外区域,只对 mask 内的
> 3D 表征做 inversion-based rectified-flow 重绘。设计细节见
> [`trellis1_to_trellis2_vinedresser3d_migration.md`](trellis1_to_trellis2_vinedresser3d_migration.md)。

---

## 1. 入口 (repo 根目录)

| 脚本 | 用途 |
|---|---|
| `run_pipeline_v3_shard_trellis2.sh` | **按 shard 的多 GPU 编排管线启动脚本**(本文档主线)。起停 VLM/FLUX 服务池、按拓扑串/并跑各 stage。 |
| `run_pipeline_minimal.py` | 单对象端到端(无需服务):best view → FLUX 2D 编辑 → P1 encode → 掩码 3D 编辑 → before/after.glb。调试/快速验证用。 |

启动(全 shard,8 GPU 并发):
```bash
MACHINE_ENV=configs/machine/local_trellis2.env \
  bash run_pipeline_v3_shard_trellis2.sh \
       shard08 configs/pipeline_v3_trellis2_partverse.yaml
```
冒烟(2 对象):加 `OBJ_IDS_FILE=<ids.txt> SHARD=08 LIMIT=2`,tag 换 `shard08_smoke`。
单步续跑:`STAGES=flux_2d,gate_2d` 限定 stage;`FORCE=1` 重跑已完成项(仍尊重 gate 前置)。

---

## 2. 数据布局 与 软连接

仓库通过 `data/` 下的软连接挂载外部存储(与代码解耦,路径在 config 里写成 repo 相对):

| repo 路径 (软连接) | 实际存储 | 角色 |
|---|---|---|
| `data/partverse` → `/mnt/zsn/data/partverse` | 只读 | **输入**:PartVerse 分片数据集 |
| `data/Pxform_v2` → `/mnt/zsn/data/Pxform_v2` | 读写 | **输出**:本管线产出 (== `/root/zsn/data/Pxform_v2`) |

**输入** (bench split,已是该格式,10 shard × 1203 obj):
```
data/partverse/inputs/
├── mesh/<shard>/<obj_id>.npz       # 几何 (parts + 顶点/面)
├── images/<shard>/<obj_id>.npz     # 预渲染多视图 + transforms.json (相机) + split_mesh.json
└── slat/<shard>/<obj_id>_*.pt      # v1 SLAT — TRELLIS.2 路径不消费,可忽略
```

**输出** (管线命名约定,`output_dir = data/Pxform_v2`):
```
data/Pxform_v2/
├── _global/manifest.jsonl                       # 全局清单 (rebuild_manifest 维护)
└── objects/<shard>/<obj_id>/
    ├── phase1/{overview.png, parsed.json}        # gen_edits:概览图 + VLM 编辑清单
    ├── edits_2d/<edit_id>_{input,edited}.png     # flux_2d:2D 编辑前/后
    ├── p1_encode/shape_slat.npz                  # trellis2_encode:原 mesh 的 shape latent 参考
    ├── edits_3d/<edit_id>/{before,after}.glb     # trellis2_3d:3D 编辑前/后 (del 分支为 after_new.glb)
    ├── edit_status.json                          # 每条 edit 的 stage 状态 (resume 真相源)
    ├── status.json                               # 对象级 step 聚合
    └── qc.json                                   # gate 质量信号
```

---

## 3. 运行环境

| 资源 | 值 |
|---|---|
| Pipeline conda env | `trellis2` (`/mnt/zsn/miniconda3/envs/trellis2`) — TRELLIS.2 codebase 依赖 (o_voxel/sparse/diffusers/nvdiffrast) |
| Server conda env | `pipeline_server` — SGLang VLM + FLUX image-edit server |
| VLM | SGLang × Qwen3.6-27B (`/mnt/zsn/ckpts/Qwen3.6-27B`),多模态,端口 8200+ |
| FLUX | `scripts/tools/image_edit_server.py` × `FLUX.1-Kontext-dev`,端口 8020+ |
| TRELLIS.2 | codebase `/mnt/zsn/3dobject/TRELLIS.2`,ckpt `/mnt/zsn/ckpts/TRELLIS.2-4B` |
| Blender | 4.2.19 LTS (`.../VoxHammer/third_party/blender-4.2.19-linux-x64/blender`) |
| GPU | L20 × 8 |

机器相关变量在 `configs/machine/local_trellis2.env`;run 相关参数在 `configs/pipeline_v3_trellis2_partverse.yaml`。

> **环境坑(已在仓库内自动修补)**:① transformers 5.9 把 DINOv3 层挪到 `model.model.layer`,
> 破坏 `pipeline.get_cond` → `trellis2_compat.patch_dinov3_extractor()`(`_ensure_pipeline` 调用)。
> ② 本机 Pillow 无 webp → `trellis2_3d._run_and_export` 有 PNG 贴图回退。

---

## 4. 编排拓扑 (stage DAG)

`pipeline.stages`(config)是调度真相源;`scheduler.dump_stage_chains` 解析为
**batch(串)→ chain(并)→ stage(串)**。默认拓扑:

```
batch1:  text_gen_gate_a
batch2:  flux_2d > gate_2d > trellis2_encode > trellis2_preview
         └────────── flux_chain (逐 stage 起停服务;GPU 阶段在服务释放后) ──────────┘
```

- **chain 内逐 stage 串行,每 stage 自管服务生命周期**:`flux_2d` 起 FLUX→停,
  `gate_2d` 起 VLM→停,之后 `trellis2_encode`/`trellis2_preview` 是 GPU step
  (`servers: none, use_gpus: true`),服务已释放才占 GPU。
- **GPU step 的多 GPU 派发在 Python 内部完成**(`run_trellis2.dispatch_gpus`,按
  `CUDA_VISIBLE_DEVICES` + `--gpu-shard k/N` 切分),bash 只负责服务池。
- **del/add 不在当前流程**:`del_mesh`/`preview_del` stage 已从 config 移除。
  以后要恢复删除分支,把 stage 加回来 + 在 `qc.edit_types` 里加 `deletion` 即可。

> scheduler 会就 "flux_chain 里有 2 个带服务的 stage(flux + vlm)" 发一条 warning —
> **预期且无害**:两者服务类型不同、逐 stage 起停。

---

## 5. 各 stage 职责

| stage | step / 状态键 | 服务 | 作用 |
|---|---|---|---|
| `text_gen_gate_a` | `gen_edits` `s1_phase1` + `gate_text_align` `sq1_qc_A` | vlm | **gen_edits (Mode A)**:渲概览图(上排 RGB 来自 npz,下排分割图 Blender 现渲)→ VLM 看图 + part 菜单产出编辑清单。**Gate A** 内联校验指令清晰、目标 part 可见。 |
| `flux_2d` | `flux_2d` `s4_flux_2d` | flux | 对通过 Gate A、且类型在 `qc.edit_types` 内的编辑(现 **mod/scl**),用最佳视角图调 FLUX → `_input.png`/`_edited.png`。 |
| `gate_2d` | `gate_2d` `sq2_qc_C` | vlm | **Gate C**:VLM 判 2D 前后对是否符合 prompt(改对 part + 保留其余)。**失败的不进 3D**(`s5` 前置 = `gate_c`)。 |
| `trellis2_encode` | `trellis2_encode` `s4b_t2_encode` | none(GPU) | **P1**:编码原 mesh → `shape_slat.npz`,作为掩码 inversion 锚定保留区的干净 latent。 |
| `trellis2_preview` | `trellis2_3d` `s5_trellis2` | none(GPU) | **掩码 3D 编辑**(下节)→ `before/after.glb`。 |

> **编辑类型 PROCESSING allow-list**(`qc.edit_types`,经 `EDIT_GEN_TYPES` 注入):当前 = `{modification, scale}`。
> **只门控处理**(`specs.iter_flux_specs` → flux_2d/gate_2d/trellis2_3d),**不影响生成**:
> `gen_edits` 始终一次性产出**全类型**全配额。所以以后开 `material`/`color`/`global` =
> 在 `qc.edit_types` 加上 + 对**已生成**的 edit 重跑 flux_2d/trellis2,**无需重新生成**。
> 临时收窄;删空该键 → 全类型处理。`del/add` 另由 stage 列表控制(已移除 del_mesh)。

---

## 6. 核心:掩码 3 层 latent 编辑 (`trellis2_3d._build_p4_mesh`)

TRELLIS.2 三阶段表征:**SS 稀疏结构 → shape SLat(几何)→ texture SLat(材质)**。
按编辑类型路由(`partcraft/edit_types.py`):

- **`S1_S2_TYPES = {modification, scale}`** — 改几何:
  1. **SS / structure**(`trellis2_structure.edit_structure`):occupancy → `ss_enc` → z_s0[1,8,16³];
     在**原图** cond 下 inversion → 16³ 掩码重绘(edit 区) → `ss_dec` → `coords_new`。
     (关键事实:TRELLIS.2 原样复用 TRELLIS.1 的 SS VAE `ss_enc/dec_conv3d_16l8`。)
  2. **shape SLat**(`trellis2_edit_stages.masked_shape_slat`):原图 cond 下 inversion P1 shape →
     在 `coords_new` 上构 x_init(保留 token 取 inversion 轨迹,edit/new token 重采)→ 编辑图 cond 前向。
  3. **texture SLat**(`masked_tex_slat`):同样掩码前向,材质重生。
- **`S2_ONLY_TYPES = {material, color, global}`** — 锁几何:`shape_new = shape0`(零漂移),只掩码材质层。
- **`MESH_ONLY = {deletion}` / `NO_GEN = {identity, addition}`** 走 mesh 路径,不在此。

要点:
- **所有 inversion 用原图**(`{edit_id}_input.png`),前向用编辑图。
- 用 **1024(非 cascade)** flow models,使 SS/shape/tex 共享同一套 64³ 坐标。
- **坐标桥** `trellis2_part_mask.build_coord_bridge`:SS 改坐标后,既在 C0 又在 edit 区外的
  保留 token 通过 index map 锚回 inversion 轨迹(`make_bridged_anchor_callback`)。
- 由 `use_mask: true`(默认)启用,**要求 `trellis2_encode` 先跑**。`false` 退回从 2D 编辑图整体重生。

---

## 7. QC gates 与 resume 模型

**三道 gate**(`final_pass = A ∧ C ∧ E`,缺席的 gate 视为通过):
- **Gate A** (`gate_text_align` / `sq1_qc_A`):文本-图像对齐(指令是否可执行、part 是否可见)。**启用**。
- **Gate C** (`gate_2d` / `sq2_qc_C`):2D 编辑前后是否符合 prompt。**启用**,且门控 3D 编辑。
- **Gate E** (`gate_quality` / `sq3_qc_E`):最终视觉质量。**停用**(需要 GLB 的 after 预览,尚未实现)。

**前置门控**(`edit_status_io.build_prereq_map`,由 config active stages 推导):
- `s4`(flux_2d)← gate_a;`s5`(trellis2_3d)← **gate_c**(无 gate_c 时退回 gate_a)。
- `edit_needs_step(ctx, edit_id, stage_key, prereq_map)` 是唯一 resume 真相源:前置 gate 非 `pass` →
  不跑;否则缺失/`error` → 跑,`done`/`pass` → 跳(`force` 仍尊重前置)。
- 状态写在 `edit_status.json`(每条 edit 的 `stages.<key>.status`)。

---

## 8. 模块地图

| 文件 | 角色 |
|---|---|
| `run_trellis2.py` | 编排 CLI:解析 config/对象列表、按 step 派发、GPU 多进程 `dispatch_gpus`、`--count-pending`。 |
| `scheduler.py` | YAML stages → batch/chain/stage 拓扑;GPU/端口/URL 解析;`dump_shell_env`(给 bash eval)。 |
| `gen_edits_image.py` | Mode A 编辑生成(概览图 + 可见性预筛 + VLM)。`gen_edits.py` 为 Mode B(纯文本)。 |
| `vlm_core.py` | VLM 调用 + 概览图渲染 (`render_overview_png`) + Gate A/C/E 判官。 |
| `flux_2d.py` | FLUX 2D 编辑 step(调 image-edit 服务)。 |
| `trellis2_3d.py` | 3D 编辑主入口 `_build_p4_mesh`、`_ensure_pipeline`、`_run_and_export`。 |
| `trellis2_structure.py` | S1 SS/structure 掩码编辑。 |
| `trellis2_edit_stages.py` | shape / texture SLat 掩码采样。 |
| `trellis2_part_mask.py` | 64³ edit grid、16³ keep mask、坐标桥。 |
| `trellis2_masked_sampler.py` | 掩码前向 + bridged anchor callback。 |
| `trellis2_encode.py` | P1:原 mesh → shape_slat.npz。 |
| `trellis2_compat.py` | DINOv3 / transformers 5.9 兼容补丁。 |
| `trellis2_white.py` | 白模(无材质)路径。 |
| `edit_types.py` | 编辑类型枚举 + 路由集合 (S1_S2 / S2_ONLY / …)。 |
| `edit_status_io.py` | edit_status.json 读写、`build_prereq_map`、`edit_needs_step`。 |
| `validators.py` | step 产物文件级校验 (`apply_check`)。 |
| `mesh_deletion.py` | del/add mesh 切割。 |
| `paths.py` / `specs.py` / `status.py` / `qc_io.py` | 路径解析 / EditSpec 迭代 / step 聚合 / qc 信号。 |

---

## 9. 已知限制 / 调优 follow-up

- **S1 硬掩码**:目前 16³ keep mask 是硬边界(v1 用距离软边界 `get_s1_soft_mask`);
  scale 这类 edit 区≈整体时易过编辑/碎裂,growth 类易欠编辑(pad=3 footprint 不够)。
  → 待加软边界混合 + 自适应 pad。
- **2D 编辑跑偏**:FLUX 对 scale(变高/变深)易反向 —— Gate C 能拦住,但浪费一次 2D 算力。
- **Gate E 停用**:缺 GLB 的 after 5 视图预览;补上后可重启最终质量门。
- **编辑类型当前收窄到 mod/scale**(`qc.edit_types`):material/color/global 待后续补回
  (S2_ONLY 路径已实现,扩 config 即可)。del/add 暂出流程(无 del_mesh stage)。

---

## 10. 环境变量速查

| 变量 | 作用 |
|---|---|
| `MACHINE_ENV` | 机器 env 文件(默认 `configs/machine/local_trellis2.env`)。 |
| `STAGES="a,b"` | 限定要跑的 stage 子集。 |
| `OBJ_IDS_FILE` / `SHARD` / `LIMIT` | 限定对象 / 分片 / 数量(冒烟)。 |
| `FORCE=1` | 重跑已完成 step(仍尊重前置 gate)。 |
| `EDIT_GEN_MODE` | `image`(默认 Mode A)/ `text`(Mode B)。 |
| `EDIT_GEN_TYPES` | 编辑类型 allow-list(csv)。`run_trellis2` 从 `qc.edit_types` 注入,显式 env 覆盖之。现 `modification,scale`。 |
| `TRELLIS_WORKERS_PER_GPU` | 每 GPU 并发 TRELLIS.2 worker 数(L20 建议 1)。 |
| `VLM_MEM_FRAC` | SGLang 显存占比(默认 0.57)。 |
