# Pipeline v3（对象中心制作管线）

`partcraft.pipeline_v3` 是一套**按对象（object-centric）**组织的 3D 编辑数据制作管线：从文本 caption 与部件表出发，经 VLM 生成编辑指令、门禁质检，再走 **CPU 删除网格**或 **FLUX + Trellis GPU 分支**，最终得到可被下游训练/评测消费的目录结构与状态文件。

更宏观的架构说明见仓库根目录 `docs/ARCH.md`（若与环境路径、历史决策相关，请以文档 + 代码为准）。本 README 仅覆盖 **v3 模块内**的行为与操作面。

---

## 1. 解决什么问题

- **Mode E（文本驱动部件编辑）**：Phase 1 用「原始文本 + 部件列表」生成编辑方案，**不依赖**在提示生成阶段对图像做编码。
- **统一编辑契约**：所有步骤通过内存中的 `EditSpec`（由每对象的 `phase1/parsed.json` 派生）描述单次编辑；与旧流水线中的 `to_legacy_dict()` / FLUX worker 仍可对接。
- **可恢复运行**：每对象维护 `edit_status.json` 与全局 `manifest.jsonl`，便于断点续跑与统计。

---

## 2. 唯一 Python 入口

```bash
python -m partcraft.pipeline_v3.run \
  --config <path/to/config.yaml> \
  --shard <NN> \
  [--steps <逗号分隔步骤>] | [--stage <阶段名>] \
  [--obj-ids ... | --obj-ids-file FILE | --all] \
  [其他 CLI 选项]
```

- **`--config`**：YAML 配置文件（必填）。
- **`--shard`**：分片 ID，内部会规范为两位（如 `8` → `08`）。
- **对象选择**：必须指定 **`--obj-ids`**、**`--obj-ids-file`**（每行一个 ID，支持 `#` 注释）或 **`--all`** 之一。
  - `--all` 时：以 `data.mesh_root/<shard>/*.npz` 的对象集合为主，并与**已有输出目录**取并集，避免续跑时丢掉「有输出但当前分片 mesh 列表里暂时没有」的对象。

---

## 3. 典型流程（Mode E vs 全功能）

### 3.1 Mode E 定稿链（文档中的默认决策）

适用于以**删除类**编辑为主、依赖 Blender 小预览做最终视觉门禁的路径：

```text
gen_edits → gate_text_align → del_mesh → preview_del → gate_quality
```

### 3.2 含 FLUX / Trellis 的扩展链

对 modification / scale / material / global / color 等需 **2D 编辑 + 3D latent** 的类型，在 Gate A 之后通常还有：

```text
flux_2d → trellis_3d → preview_flux → gate_quality
```

可选：`render_3d`（40 视角完整渲染，偏报告或下游需要稠密视图时）。

### 3.3 注释但未默认启用的步骤

- **`reencode_del`**：删除编辑的 GPU 重编码（Blender 多视角 → SLAT 等），在 `run.py` 中保留为注释分支，需显式接入时再启用。

---

## 4. 步骤说明（`ALL_STEPS`）

| CLI 步骤名 | 含义 | 备注 |
|------------|------|------|
| `gen_edits` | Phase 1：VLM 根据文本与部件表生成 `parsed.json`、总览图等 | 可多 VLM URL 扇出；可用 `pipeline.prerender_workers` 控制预渲染并发 |
| `gate_text_align` | **Gate A**：指令是否清晰、总览图中目标部件可定位、类型与 prompt 一致 | 可与 `gen_edits` 同一次 `--steps` 串联时通过 **per-object 回调**在 Phase1 后立即执行 |
| `del_mesh` | **CPU**：对通过 Gate A 的删除编辑做面片级修改，产出 `after_new.glb` | 依赖归一化 GLB 路径等配置 |
| `preview_del` | **Blender**：对删除结果渲 5 视角 `preview_{0..4}.png` | 供 Gate E「after」行；可 `best_view_only` 加速 |
| `flux_2d` | **HTTP**：FLUX 图像编辑，写出 `edits_2d/<edit_id>_*.png` | 需 `services.image_edit` 与 URL |
| `trellis_3d` | **GPU**：Trellis 3D latent 编辑 → `edits_3d/<edit_id>/after.npz` | 多卡时由 `dispatch_gpus` 子进程 + `--gpu-shard` 切分 |
| `preview_flux` | **GPU**：从 Trellis 结果解码并渲 5 视角 preview | Gate E 的 flux 分支「after」来源 |
| `render_3d` | **GPU**：40 视角完整 3D 渲染 | 可选 |
| `gate_quality` | **Gate E**：VLM 对 before/after 拼图打分（如视觉质量、区域正确性、非编辑区保持） | 可用环境变量或配置限制只评部分 `edit_type` |

---

## 5. 磁盘布局（摘要）

输出根目录由配置 **`data.output_dir`** 指定。每个对象目录在：

```text
{output_dir}/objects/<shard>/<obj_id>/
```

常见内容（不同步骤逐步增量）包括：

- `meta.json`、`edit_status.json`（**步骤级与编辑级状态的统一落点**，含 `steps`）
- `phase1/`：`parsed.json`、`overview.png`、`raw.txt` 等
- `edits_2d/`：FLUX 输入输出 PNG
- `edits_3d/<edit_id>/`：`after.npz`、`preview_*.png`、`after_new.glb`（删除分支）等

全局：

- `{output_dir}/_global/manifest.jsonl`：由 `status.rebuild_manifest` 汇总各行对象步骤状态

完整字段与约定见 `paths.py` 文档串。

---

## 6. CLI 选项一览

| 选项 | 作用 |
|------|------|
| `--config PATH` | 配置文件（必填） |
| `--shard NN` | 分片 ID |
| `--steps a,b,c` | 逗号分隔步骤列表，与 `--stage` 二选一 |
| `--stage NAME` | 按 YAML `pipeline.stages` 中名为 `NAME` 的阶段运行其 `steps` |
| `--obj-ids ...` | 仅处理列出的对象 |
| `--obj-ids-file FILE` | 从文件读取对象列表 |
| `--all` | 处理本分片全部对象（逻辑见上文） |
| `--gpus a,b,c` | 覆盖配置中的 GPU 列表；用于需要多卡派发的步骤 |
| `--vlm-url URLs` | 逗号分隔，覆盖 VLM 服务地址 |
| `--flux-url URLs` | 逗号分隔，覆盖 FLUX 服务地址 |
| `--force` | 强制重跑（由各 runner 解释） |
| `--best-view-only` | `preview_del` 仅处理 Gate A 最优视角（更快） |
| `--skip-input-check` | 跳过启动前的 mesh/image/slat 文件检查（例如只跑不读某些输入的步骤时） |
| `--dry-run` | 打印各对象已标记完成的步骤与 manifest 摘要 |
| `--count-pending` | 按当前 stage/steps（及 Gate E 类型过滤）打印仍待处理的对象数量 |

内部参数（子进程使用）：`--single-gpu`、`--gpu-shard i/n`（一般无需手填）。

---

## 7. 配置文件结构

### 7.1 必填与常用：`data`

- **`output_dir`**：输出根（必填）。
- **`mesh_root` / `images_root`**：输入 `npz`，相对路径默认为 `data/partverse/mesh` 与 `data/partverse/images`（可在 YAML 覆盖）。
- **`slat_dir`**（可选）：若配置，则运行前检查 `{shard}/{obj_id}_coords.pt` 与 `_feats.pt`。
- **`normalized_glb_dir` / `anno_dir`**：删除分支与标注相关路径（按实际 bench 布局填写）。

### 7.2 `services`（由 `services_cfg.py` 统一读取）

- **`services.vlm`**：至少为 mapping；**`model` / `vlm_model`** 指定 VLM 模型名或路径。
  - 可用 **`base_urls`**（或兼容键 `vlm_base_urls`）显式列出 `http://host:port/v1`，覆盖默认的 `localhost:vlm_port_base + i*stride` 推导。
- **`services.image_edit`**：启用 FLUX/Trellis 相关步骤时使用。
  - 常用键：`base_urls`、`workers_per_server`、`trellis_text_ckpt`、`trellis_workers_per_gpu` 等。
  - **`trellis_workers_per_gpu`**：仅 `trellis_3d` 多进程每卡 worker 数；可被环境变量 **`TRELLIS_WORKERS_PER_GPU`** 覆盖。

### 7.3 `pipeline`

- **`gpus`**：整数列表，如 `[0,1,2,3]` — `run` 在需要时解析 VLM/FLUX 默认 URL 与 GPU 派发；若缺失，依赖 GPU 默认 URL 或 `--gpus` 的步骤可能报错（以实际报错为准）。
- **`vlm_port_base` / `vlm_port_stride`**：未设置 `base_urls` 时构造默认 VLM URL。
- **`flux_port_base` / `flux_port_stride`**：同上，用于 FLUX。
- **`prerender_workers`**：`gen_edits` 预渲染并发度。
- **`gate_a_concurrency` / `gate_a_per_obj_concurrency`**：Gate A 并发策略。
- **`s6p_del_workers`**：`preview_del` Blender 并行 worker 数。
- **`stages`**：供 **`--stage`** 使用；非空列表，每项含 `name`、`steps`、可选 `use_gpus`、`optional`、`parallel_group`、`chain_id`、`chain_order` 等（详见 `scheduler.Phase`）。

项目内模板：`configs/templates/pipeline_v3_bench.template.yaml`。

### 7.4 `step_params`

按步骤名（如 `preview_del`）分块，例如：

```yaml
step_params:
  preview_del:
    best_view_only: false
```

CLI `--best-view-only` 与 YAML 任一为真即可启用「仅最佳视角」。

### 7.5 `qc`

- **`gate_quality_types`**：列表，若设置则 Gate E **仅**对这些编辑类型打分（与下文环境变量二选一逻辑见 `run.py`）。
- **`thresholds_by_type`**：各 `edit_type` 的阈值示例见模板（具体键以 `qc_rules.py` / `vlm_core` 实现为准）。

---

## 8. CLI 步骤名 ↔ `edit_status.json` 内键名

`run.py` 中为兼容历史状态字段，将公开步骤名映射为内部键：

| CLI | `steps` 内键 |
|-----|----------------|
| `gen_edits` | `s1_phase1` |
| `gate_text_align` | `sq1_qc_A` |
| `del_mesh` | `s5b_del_mesh` |
| `preview_del` | `s6p_del` |
| `flux_2d` | `s4_flux_2d` |
| `trellis_3d` | `s5_trellis` |
| `preview_flux` | `s6p_flux` |
| `render_3d` | `s6_render_3d` |
| `gate_quality` | `sq3_qc_E` |

调试 `--dry-run` 时看到的「已存在步骤」即以上键。

---

## 9. 环境变量（常用）

| 变量 | 作用 |
|------|------|
| `LIMIT` | 正整数时，在 GPU 分片等逻辑之后**仅保留前 N 个对象**（用于小样本试跑） |
| `TRELLIS_SHARD_MODE` | 设为 `roundrobin` / `rr` 时，`trellis_3d` 的 GPU 分片退回按对象下标轮转；默认倾向 **LPT**（按待跑 s5 工作量分配） |
| `TRELLIS_WORKERS_PER_GPU` | 覆盖每 GPU 上 `trellis_3d` 并发 worker 数 |
| `QC_ONLY_TYPES` | 逗号分隔编辑类型，限制 **`gate_quality`** 只评这些类型（优先级高于 YAML `qc.gate_quality_types`） |
| `ATTN_BACKEND` | 多 GPU 子进程内默认设为 `flash_attn`（见 `dispatch_gpus`） |

调度脚本层另有：`STAGES`、`FORCE`、`OBJ_IDS_FILE`、`MACHINE_ENV`、`VLM_MEM_FRAC` 等，见 `scripts/tools/run_pipeline_v3_shard.sh` 注释。

---

## 10. 多 GPU 派发（`dispatch_gpus`）

- 对 **`trellis_3d`、`preview_flux`、`render_3d`**，若配置了 `--gpus`（或由 `--stage` 从配置注入 `pipeline.gpus`），父进程会为每块 GPU（及 `trellis_3d` 的每卡多 worker）拉起子进程，设置 **`CUDA_VISIBLE_DEVICES`**，并传递 **`--gpu-shard i/total`**。
- **`trellis_3d`** 的 `total` 含「每 GPU 多 worker」：`total = n_gpus * trellis_workers_per_gpu`。
- 子对象列表在子进程内用 **`_gpu_shard_ctxs`** 切分；当运行 **`trellis_3d`** 且未设置 `TRELLIS_SHARD_MODE=roundrobin` 时，对「仍有 s5 待办编辑数」做 **LPT** 均衡，**零待办**对象再 round-robin 摊到各 shard，避免空转。

---

## 11. 启动前输入检查（`check_inputs`）

默认在**非** `--single-gpu` 子进程且未 `--skip-input-check` 时执行：

- 每个对象：`mesh_root` 与 `images_root` 下对应 `npz` 存在；
- 若配置了 `slat_dir`：`coords.pt` 与 `feats.pt` 存在。

缺失则直接 `SystemExit`，避免跑到一半才发现缺文件。

---

## 12. Shell 调度（可选）

在同一台机器上按 **YAML 中的 stage 拓扑**（顺序 batch、并行 group、串链子链）起停 VLM/FLUX 服务并调用 `python -m partcraft.pipeline_v3.run --stage ...` 时，可使用：

```bash
bash scripts/tools/run_pipeline_v3_shard.sh <tag> <config.yaml>
```

其它变体如 `run_pipeline_v3_bench.sh`、`run_pipeline_v3_bench100.sh` 见 `scripts/tools/` 内注释。该类脚本通常依赖 `configs/machine/<hostname>.env` 中的 conda 与 checkpoint 环境变量。

---

## 13. 与 `pipeline_v2` 的关系

仓库内**生产编排的主线**仍以 `python -m partcraft.pipeline_v2.run` 等为统一入口（见仓库规则）；**v3 是并行演进的模块**，CLI 为 `partcraft.pipeline_v3.run`。两者共享部分概念（如 `EditSpec` 字段命名、`services_cfg`、调度思想），但**输出目录布局与步骤集合以各模块文档与代码为准**，勿混用路径假设。

---

## 14. 模块内文件索引（便于深入阅读）

| 文件 | 职责 |
|------|------|
| `run.py` | CLI、步骤派发、GPU 子进程、输入检查、manifest |
| `scheduler.py` | `pipeline.stages` 解析、URL 推导、stage/chain 拓扑辅助 |
| `services_cfg.py` | `services` / `step_params` / 模型名 / Trellis worker 数 |
| `paths.py` | `PipelineRoot`、`ObjectContext`、数据集根路径 |
| `specs.py` | `EditSpec` 与 `parsed.json` 迭代器 |
| `gen_edits.py` | Phase 1 生成 |
| `vlm_core.py` | Gate A / Gate E 等 VLM 逻辑 |
| `flux_2d.py` / `trellis_3d.py` / `preview_render.py` / `render_3d.py` | 各步 runner |
| `mesh_deletion.py` | 删除网格 |
| `status.py` / `edit_status_io.py` | 状态持久化与 prerequisite |
| `validators.py` | 步骤完成后对磁盘产物的校验 |

---

## 15. 最短 smoke 建议

1. 准备小分片输入：`mesh` / `images`（及若需要则 `slat`）路径正确。  
2. 使用极小 `OBJ_IDS` 或 `LIMIT=1`。  
3. 先 `--dry-run` 看已有进度，再 `--steps gen_edits,gate_text_align`（或你需要的子集）。  

若 Gate E 与 `preview_del` 分阶段跑，注意用 **`QC_ONLY_TYPES`** 或 **`qc.gate_quality_types`** 与 `--count-pending` 对齐，避免调度器误以为已全部完成而跳过 VLM。

---

*本 README 描述以当前仓库中 `partcraft/pipeline_v3` 源码为准；若行为与文档冲突，以代码为最终依据。*

