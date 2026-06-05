# TRELLIS.2 3D 编辑：S1/S2 阶段搭配 & 管线开关速查

> 范围：`trellis2_preview` 阶段（`partcraft/pipeline_v3/trellis2_3d.py` →
> `trellis2_structure.py` / `trellis2_edit_stages.py` / `trellis2_masked_sampler.py`）。
> 所有开关都写在 config 的 `services.image_edit:` 块里（被 `trellis_image_edit_flat()`
> 摊平成 `p25_cfg`）。本文给出每个搭配的思路 + 完整开关表 + 运行命令。
> 最后更新：2026-06-02（FlowEdit-S1 + free-S2 smoke）。

---

## 0. 两阶段总览

一次 image-conditioned 3D 编辑 = **S1（结构）→ S2（外观）**：

| 阶段 | 作用 | latent | 产物 |
|---|---|---|---|
| **S1 结构** | 决定「哪些 voxel 存在」 | SS latent 16³×8（占据 64³） | `coords_new` [M,3] |
| **S2 几何** | 在 `coords_new` 上生成 shape 细节 | shape SLat @1024 | `shape_new` |
| **S2 材质** | 在 `coords_new` 上生成 PBR | tex SLat @1024 | `tex_new` |

最后 `decode_latent(shape_new, tex_new) → mesh → GLB`。

### 编辑类型 → 走哪条路（`partcraft/edit_types.py`）

| 类型 | 常量集合 | 走法 |
|---|---|---|
| `modification`, `scale` | `S1_S2_TYPES` | **S1 + S2**（几何变） |
| `material`, `color`, `global` | `S2_ONLY_TYPES` | **S2 only**（`coords_new=coords0`，几何锁死，仅重生成材质） |
| `deletion` | `MESH_ONLY_TYPES` | GT mesh 布尔删除，不走生成 |
| `identity`, `addition` | `NO_GEN_TYPES` | 不生成 |

> 当前 smoke 的 `qc.edit_types: [modification, scale]` → 只跑 S1+S2 路径。

---

## 1. S1（结构）阶段：模式与开关

S1 路径在 `trellis2_3d.py:396-553`，**按优先级**选分支（前者命中即跳过后者）：

### 1a. `trellis2_ss1_coords_dir`（TRELLIS.1-SS 桥接，实验对照）
外部离线（vinedresser3d 环境）生成的 TRELLIS.1 占据 `ss1_coords.npz` 直接当结构 →
**强制 free S2**。用于检验「T1 的 SS flow 是否比 T2 masked S1 几何更干净」。
命中条件：该路径下存在 `<dir>/<obj>/<edit_id>/ss1_coords.npz`。

### 1b. `trellis2_ss_vanilla: true`（T2 原生 vanilla-SS，机制对照）
用 T2 **自己**的 SS flow 在**整物体 vanilla 模式**（无 mask）采样结构 → **强制 free S2**。
隔离「vanilla vs masked（机制）」与「T1 vs T2（模型）」两个变量。

### 1c. `trellis2_s1_mode`（主开关，需要 `target_part_ids`）

| 值 | 思路 | 实现 |
|---|---|---|
| `masked`（默认） | **反演 + keep-mask 重绘**：反演原 SS latent，编辑区按 keep-mask 重新去噪 | `edit_structure()` |
| `flowedit` | **源/目标速度差 ODE**：不反演、不 mask，编辑区由「条件之差」自然涌现；条件一致处 `v_delta≈0` 不动 | `flowedit_structure()` |

**FlowEdit 要点（硬约束）**：
- **对称 CFG**：`gs_src == gs_tgt`（默认都 7.5）。不等会注入「即使条件相同也非零」的
  (pos−neg) 推力，在精细结构上打碎 occupancy。
- **必须 canonical frame**：`trellis2_canonical_frame: true`，否则结构错乱。
- **keep_mask 安全网（当前未从 config 接线）**：`flowedit_sample` 支持每步把非编辑
  voxel 硬钉回源 latent（混合式 masked-FlowEdit）→ 保留区逐位冻结，代价是重新引入
  mask 接缝。目前 `flowedit_structure` 调用未传 `keep_mask`，是**纯 FlowEdit**。

### 1d. S2-only 类型（material/color/global）
`coords_new = coords0`，`shape_new = shape0`（几何逐位复用），只走 S2 材质。

### S1 通用旋钮

| key | 默认 | 作用 |
|---|---|---|
| `trellis2_s1_pad` | `0` | part 编辑区在 64³ 上的膨胀半径（Chebyshev 盒式全向外扩）。**默认 0 = 不膨胀，编辑区 == 纯 part-id mask**；v1 用 3，设 >0 重新开启（过大→结构糊成块） |
| `trellis2_s1_keep_thresh` | `0.1` | keep-mask 阈值；也决定 16³ keep mask |
| `trellis2_mask_subtract_preserved` | `=contact_soft` | 从编辑区减掉相邻保留部件 |
| `trellis2_s1_densify` | `0` | 编辑区占据加厚迭代次数（合上薄壳/透壳） |
| `trellis2_s2_remove_small` | `0` | 删掉小于 N voxel 的漂浮碎块 |
| `trellis2_s2_restore_preserved` | `False` | S2 前补回 source slat 在 mask 外、被 S1 丢掉的 body 占据（占据+latent 一起经 bridge 锚定）。配合 `s1_pad>0` 抵消膨胀对 body 的侵蚀。见 `trellis2_512_pad2_restore.md` |
| **FlowEdit 专属** | | |
| `trellis2_s1_fe_gs_tgt` | `7.5` | 目标分支 CFG |
| `trellis2_s1_fe_gs_src` | `=gs_tgt` | 源分支 CFG（保持相等＝对称） |
| `trellis2_s1_fe_navg` | `1` | 每步 Monte-Carlo 平均次数 |
| **contact-soft（v1 interweave）** | | |
| `trellis2_s1_contact_soft` | `False` | 主开关：S1 加 contact-aware 距离变换软 mask（S2 不受影响） |
| `trellis2_s1_soft_sigma` | `None` | 软 mask sigma；None＝按接触比例动态 |
| `trellis2_s1_soft_feather` | `0.0` | masked 模式 keep 边界羽化 |
| **SS 采样器覆盖（masked & flowedit 都吃）** | | |
| `trellis2_ss_align_t1` | `False` | 换 T1 温和调度 `{steps25, gs5, interval[0.5,1], rt3}`（修大件塌陷） |
| `trellis2_ss_steps` | — | 覆盖 SS 步数 |
| `trellis2_ss_cfg` | — | 覆盖 SS guidance_strength |

---

## 2. S2（shape+tex）阶段：anchor 模式与开关

`trellis2_s2_anchor_mode`（默认 `perstep`），实现见
`masked_shape_slat()` / `masked_tex_slat()`：

| 模式 | shape 生成 | body 保真 | 接缝 | 思路 |
|---|---|---|---|---|
| `perstep`（默认） | 每步把保留区锚回反演原 latent | 逐位精确 | 无 | 老路；编辑区一路盯着分布外邻居 → **透壳/holey** |
| `release_late` | `t>=cutoff` 锚定，之后释放一起去噪 | 早期锁结构 | 弱 | 让编辑区末段「愈合」成闭合面 |
| `posthoc` | 整场自由生成 → **结尾硬粘回原 body clean latent** | **逐位精确** | 可能 1-voxel seam | 实心编辑区 + 精确 body |
| `free` | 整场自由生成，**不粘** | 仅结构保留（沿用原 occupancy 坐标），细节会漂 | 无（全连贯） | 无缝优先于逐位保真 |
| `contact_soft` | 接触边界附近软混、远处硬锚 | 接近精确 | 自愈 | v1-faithful，按接触距离逐 token 加权 |

**`free` vs `posthoc`：内核相同，只差结尾那一步** —— 都「整场自由生成」，区别只是
posthoc 最后把保留 token 粘回原始 clean latent（精确但可能有缝），free 不粘（无缝但 body 漂）。

> ⚠️ **tex 的不对称**：`masked_tex_slat` 里**只有 `free` 有专门分支**（无参考/无反演/无锚，
> `:256`）。`posthoc` 在 tex 里**没有单独实现**，会落到 per-step 锚定的 masked 路径
> （`:311`）。即 `posthoc` ≈「shape 自由生成+粘回原几何」+「tex 反演原纹理逐步锚定」；
> 只有 `free` 是 shape 和 tex **都**自由。

### S2 其它旋钮

| key | 默认 | 作用 |
|---|---|---|
| `trellis2_s2_anchor_cutoff` | `0.3` | `release_late` 的释放时刻 |
| `trellis2_s2_warmstart` | `False` | 编辑区已存在的 token 用反演原 latent 暖启动 |
| `trellis2_s2_nn_init` | `False` | 新长出的 token 用最近邻 seed（隐含 warmstart） |

---

## 3. 常用搭配（recipes）

| # | 名称 | S1 | S2 | 思路 / 适用 |
|---|---|---|---|---|
| R1 | legacy masked | `masked` | `perstep` | 最早基线；编辑区易透壳 |
| R2 | masked + posthoc | `masked` | `posthoc` | 实心编辑区 + 逐位保 body，代价 seam |
| R3 | masked + contact-soft | `masked`(+`s1_contact_soft`) | `contact_soft` | **v1 interweave 复刻**；接触自愈 |
| R4 | **FlowEdit + free**（当前 smoke） | `flowedit` | `free` | 最大自由度、最少约束、无缝；body 会漂 |
| R5 | FlowEdit + posthoc | `flowedit` | `posthoc` | FlowEdit 结构 + 逐位保 body（对照 R4） |
| R6 | vanilla-SS + free | `ss_vanilla` | free(强制) | 机制对照：T2 自己的 vanilla SS |
| R7 | T1-SS bridge + free | `ss1_coords_dir` | free(强制) | 模型对照：外部 TRELLIS.1 占据 |
| R8 | **512 编辑分辨率** | 任意 S1 + `edit_res:512` | 任意 S2 @512 | 分辨率对照：S2 在 32³ + `_512` 模型 + decode@512。`configs/pipeline_v3_trellis2_flowedit_free_r512.yaml`(R4@512) / `_masked_posthoc_r512_pad0.yaml`(R3@512)；输出 `_exp_flowedit_free_r512` / `_exp_masked_posthoc_r512_pad0`，A/B vs 1024。 |

> R4 = 当前 `configs/pipeline_v3_trellis2_flowedit_free_r1024.yaml` 的配置。
> 想往回收 body：S1 接 keep_mask（收结构漂移），或 S2 `free→posthoc`（粘回几何）。

---

## 4. 其它管线开关（encode / 渲染 / 导出 / 生成）

| key | 默认 | 作用 |
|---|---|---|
| **编辑分辨率（512 实验）** | | |
| `trellis2_edit_res` | `1024` | **主开关**：S2 SLat 分辨率（`1024`/`512`）。`512` → S2 用 `shape/tex_slat_flow_model_512`、conds@512、`decode_latent(.,.,512)`，coords 网格 = `res//16`（512→32³，1024→64³）。**S1 永远 64³**（TRELLIS.1 SS VAE 固定 64³→16³）：先在 64³ 出 `coords_new`，再 max-pool 降到 32³ 喂 S2。需要 grid-512 sidecar body（见下）。 |
| **帧 / encode** | | |
| `trellis2_canonical_frame` | `False`（smoke=true） | Y-up→Z-up `_CANON_ROT`（encode + mask 同帧）；**FlowEdit 必须 true** |
| `trellis2_p1_grid` | `1024` | encode 体素分辨率（64³ 主编码，驱动 S1） |
| `trellis2_edit_res` sidecar | — | `edit_res=512` 时，encode 阶段额外在 grid 512 重编码 shape+tex → `p1_encode/shape_slat_e512.npz`+`tex_slat_e512.npz`（32³），作为 S2 body 锚点。64³ 隐变量不能喂 `_512` 模型（非线性，不能下采样）。 |
| `trellis2_shape_enc`/`tex_enc`/`ss_enc` | 见 `trellis2_encode.py` | 三个编码器名 |
| **统一 PBR 渲染**（见 [`ovox-render`]） | | |
| `trellis2_encode_render_overview` | `False`（smoke=true） | encode 阶段前置渲染 overview（RGB=decode latents，seg=part-mesh）|
| `trellis2_render_gate_views` | `False`（smoke=true） | GPU 阶段渲 gate-E before/after 命名视角 |
| `trellis2_gate_view_res` | `512` | gate 视角分辨率 |
| `trellis2_hdri` | `forest.exr` | PBR envmap |
| **导出 / 产物** | | |
| `use_mask`（旧名 `use_p4`） | `True` | P4 masked-edit 主路径开关 |
| `emit_before` | `True` | 同时导出 before.glb（false 省一半时间） |
| `trellis2_export_partverse_frame` | — | 把导出 GLB 反变换回 partverse 世界帧 |
| `trellis2_save_latents` | — | 落盘中间 SS/shape/tex latent |
| `trellis2_pipeline_type` | `1024_cascade` | TRELLIS 采样管线 |
| `trellis2_seed` | `1` | 随机种子 |
| `trellis2_num_samples` | `1` | 每条编辑采样数 |
| `trellis2_decimation_target` | `1000000` | 导出面数目标 |
| `trellis2_texture_size` | `4096` | 导出贴图分辨率 |
| `trellis2_extension_webp` | `True` | GLB 贴图用 webp |

---

## 5. 怎么跑 / 怎么切换

### 改搭配 = 改 config 的 `services.image_edit:` 块
这些 `trellis2_*` 开关**只走 YAML**（没有 CLI override）。例如当前 R4：

```yaml
services:
  image_edit:
    use_mask: true
    trellis2_canonical_frame: true      # FlowEdit 必须
    trellis2_s1_mode: flowedit          # S1 = FlowEdit
    trellis2_s2_anchor_mode: free       # S2 = free
```

切到 R5（FlowEdit + posthoc）：把 `trellis2_s2_anchor_mode: posthoc`。
切到 R3（v1 interweave）：`trellis2_s1_mode: masked` + `trellis2_s1_contact_soft: true`
+ `trellis2_s2_anchor_mode: contact_soft`。

### 运行命令（runner = `run_pipeline_v3_shard_trellis2.sh`）

```bash
# 全 6 阶段，15 物体，GPU 0/2（gpus 写在 config 的 pipeline.gpus）
OBJ_IDS_FILE=data/Pxform_v2/_exp_flowedit_free_r1024/smoke15_ids.txt SHARD=08 FORCE=1 \
MACHINE_ENV=configs/machine/local_trellis2.env \
  bash run_pipeline_v3_shard_trellis2.sh shard08_glbsmoke \
       configs/pipeline_v3_trellis2_flowedit_free_r1024.yaml
```

### runner 环境开关（env，不进 config）

| env | 作用 |
|---|---|
| `FORCE=1` | 重跑已完成步骤（改了开关后必须带，否则跳过） |
| `OBJ_IDS_FILE=<txt>` | 限定物体 id 列表；缺省 `--all` |
| `STAGES="a,b,c"` | 只跑部分阶段（如 `STAGES=trellis2_preview` 单测 3D 编辑） |
| `LIMIT=N` | 限制物体数（快速 smoke） |
| `SHARD=08` | 分片（数据在 `.../mesh/08/`） |
| `MACHINE_ENV=<env>` | conda 环境 + ckpt 路径 |
| `EDIT_GEN_MODE=image\|text` | gate-A 编辑生成模式（默认 image=Mode A overview） |
| `VLM_MEM_FRAC=0.57` | SGLang 显存比例 |

> 只重跑 3D 编辑、不动前面的 encode/flux/gate：
> `STAGES=trellis2_preview FORCE=1 ... bash run_pipeline_v3_shard_trellis2.sh ...`
> （前提：encode/flux/gate-2d 产物已在；gate-2d==pass 的编辑才会进 3D 阶段）

### 512 编辑分辨率实验（R8）

复用现有 smoke 的上游产物，只在 512 下重跑 encode（加 32³ sidecar）+ 3D 编辑：

```bash
# 1) 从 1024 smoke 树 seed 出 512 sibling（symlink phase1/edits_2d，copy edit_status+p1_encode，
#    gate_views 用空目录，去掉 edits_3d）
bash scripts/experiments/seed_e512_sibling.sh \
     data/Pxform_v2/_exp_flowedit_free_r1024    data/Pxform_v2/_exp_flowedit_free_r512
bash scripts/experiments/seed_e512_sibling.sh \
     data/Pxform_v2/_exp_masked_posthoc_r1024 data/Pxform_v2/_exp_masked_posthoc_r512_pad0

# 2) 跑 encode(grid512 sidecar) + 3D 编辑(512)
OBJ_IDS_FILE=data/Pxform_v2/_exp_flowedit_free_r512/seeded_ids.txt SHARD=08 FORCE=1 \
STAGES=trellis2_encode,trellis2_preview MACHINE_ENV=configs/machine/local_trellis2.env \
  bash run_pipeline_v3_shard_trellis2.sh e512_flowedit_full \
       configs/pipeline_v3_trellis2_flowedit_free_r512.yaml
# masked 同理，换 _masked_posthoc_r512_pad0.yaml / seeded_ids.txt

# 3) 生成 1024 vs 512 对比 HTML（masked / flowedit）
python scripts/viz/ab_res_html.py masked
python scripts/viz/ab_res_html.py flowedit
```

> 命名说明：`_exp_flowedit_free_r1024`(1024) 这棵树跑的是 FlowEdit+free，512 sibling 按 recipe 命名为
> `_exp_flowedit_free_r512`（"glb" 只指渲染来源，不是编辑方案）。

A/B：`_exp_flowedit_free_r1024`(1024) vs `_exp_flowedit_free_r512`(512) 的 `edits_3d/<id>/after_view_*.png`。
日志确认：`loaded P1 SLat (S1 N tok @64³, S2 M tok @32³) edit_res=512` + `decode_latent` 不崩。
