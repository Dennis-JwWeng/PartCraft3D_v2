# pad0 重贴方案（offline re-paste，无重新生成）

实验结论先行：**对 shard00 全量（2284 个 gate-E pass edit）做了 pad0 重贴并出了
同视角四列对比（`reports/repaste_pad0_shard00/index.html`），目测改善不明显。**
原因分析见末节 —— 该方案只动了一层很薄的 latent 环带，动不了几何占据，
这是它的结构性上限。

## 动机

生产配置（`pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml`）里
`trellis2_s1_pad: 4` 是 S1/S2 共用的唯一编辑区定义：64³ 编辑格被 Chebyshev
膨胀 4 格后，既决定 S1 的 keep-mask / restore，也决定 S2 shape+tex posthoc
贴回时的 preserved 区域。膨胀环（pad ring）内的 token 在 S2 是**自由生成**的
—— 怀疑这圈"本不该动却被重新生成"的区域造成保留区外观漂移。

想法：S2 的贴回掩码其实不需要跟 S1 共用 pad。生产 run 已经存了所有需要的
latents，可以**离线**用 pad0 掩码重新执行一次 posthoc 贴回，零生成、零采样，
把环带 token 也贴回原始编码 latent。

## 方案定义

逐 edit（仅 gate-E pass）：

1. **占据 AS-IS**：直接用生产 run 的 `latents/ss.npz` 里的 `coords_new`
   （pad4 run 的 S1 结果，含 in-run 64³ restore）。**不改任何坐标**。
2. **重建掩码**：`part_edit_grid_64(mesh_npz, parts, pad=0, canonical=True,
   subtract_preserved=True)` 在 64³ 上重建编辑格（不膨胀），再
   `downsample_edit_grid` max-pool 到 32³（512 分辨率的 SLat 格）。
3. **桥接**：`build_coord_bridge(coords0, coords_new, grid32)` →
   `preserved = token ∈ C0 且 ∉ edit_grid`。相比 pad4，preserved 多出
   "pad 环 ∩ 两侧坐标都存在" 的环带 token。
4. **硬贴回**（denorm 空间，与管线内归一化贴回等价）：
   - `shape_after[preserved] = shape_slat_e512[src]`
   - `tex_after[preserved]   = tex_slat_e512[src]`
   white-model 物体没有 tex latent → 只贴 shape，`build_white_model_mesh` 解码。
5. **解码 + 渲染**：decode @512，只渲 gate-A best view
   （`VIEW_ORDER[gate_a.vlm.best_view]`，即 FLUX condition 相机），便于与
   `edits_2d/<id>_input.png` / 生产 after 同视角对比。

## 代码与产物

| 件 | 路径 |
|---|---|
| 单物体预览（6 视图 + compare 条） | `scripts/viz/repaste_pad0_preview.py` |
| 批处理（按 shard，gate-E pass only） | `scripts/repaste_pad0_batch.py` |
| 8 卡驱动 | `scripts/run_repaste_pad0_shard.sh`（`GPUS`/`PAD` 可覆写） |
| 对比 HTML（4 列同视角，分页内嵌） | `scripts/viz/build_repaste_compare_html.py` |

每 edit 输出 `edits_3d/<id>/repaste_pad0/`：
`shape_slat.npz` / `tex_slat.npz`（重贴后 latent，coords_new 帧、denorm）、
`after_view_<view>.png`（sentinel，断点续跑判据）、`meta.json`（preserved 统计）。

吞吐：~14–17 edit/min/卡（解码 + 单视图渲染，约 3s/个），一个 shard 8 卡
15–20 分钟。

## 与管线内方案的关系

等价于在管线 S2 posthoc 贴回处把掩码换成 pad0 —— 贴回是幂等覆写，preserved
集只增不减，所以"生产 pad4 生成 + 离线 pad0 再贴"与"直接 pad0 贴回"结果一致
（自由生成部分逐 token 相同，只是更多 token 被覆写）。无需重跑任何 flow。

## 为什么看不出明显改善（结构性上限）

1. **几何占据没变。** `coords_new` 原样保留：pad 环里被 S1 增删/挪动的
   voxel 还是增删/挪动的。重贴只能改"已存在 token 的特征"，**补不回丢失的
   voxel，也去不掉多长出来的 voxel** —— 而几何漂移恰恰是占据层面的。
2. **只有"两侧都在"的 token 受益。** `build_coord_bridge` 要求 token 同时
   出现在 coords0 和 coords_new；环带里坐标变了的 token 桥不上，贴不到。
3. **增量本来就薄。** pad4@64³ 下采样到 32³ 后≈pad2@32³，再加 max-pool 的
   ANY 语义，pad0 多覆盖的只有 1–2 格环带。实测 preserved 增量普遍是
   总 token 的几个百分点（如 434 token 的 edit 仅 +30 左右）。
4. **best view 看不见环带。** 环带紧贴编辑部件边缘，在 512 渲染里就是部件
   轮廓外侧几个像素；除非生产 run 恰好在环带上生成了明显错误的材质/细节，
   单视角缩略图上几乎不可分辨。
5. **真正的退化大多不在环带里。** 保留区大面积变样多半来自
   (a) S1 占据本身的漂移（重贴管不着），或 (b) 编辑部件内部的自由生成质量
   （本来就该生成，重贴不该管）。

## 推论 / 下一步选项

- 若要修**几何**层面的保留区漂移，得动占据：用 pad0 格重做
  `restore_preserved_occupancy`（会改 coords_new，环带 voxel 回到 before 状态，
  随后 shape/tex 全部可桥接可贴回）—— 代价是环带处可能出现 S1 新旧占据的
  硬接缝。
- 若问题在编辑部件内部质量，则与 pad 无关，方向是 S2 生成本身
  （condition、步数、anchor 模式）。
- pad0 重贴产物（`repaste_pad0/*.npz`）保留：即便视觉无感，它在数值上是
  "保留区=原始编码"的更严格版本，若下游训练对保留区一致性敏感可直接换用。
