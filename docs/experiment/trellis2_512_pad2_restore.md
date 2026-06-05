# 实验：512 编辑分辨率 + padding=2 + preserved 占据补救

## 目标

在 TRELLIS.2 masked 3D 编辑上,同时引入三个改动并做 A/B,看是否能在**更便宜的 512
分辨率**下、用**更宽的编辑区(pad=2)**给 part 留出生长空间,同时用一个**占据补救**机制
把因膨胀/边界侵蚀而丢掉的 body 占据补回来,避免 preserved 区域出现破洞/缺块。

## 变量

| 旋钮 | 值 | 说明 | 实现状态 |
|---|---|---|---|
| `trellis2_edit_res` | **512** | S1 仍 64³,S2 用 `*_slat_flow_model_512` 在 32³ 上跑 + grid-512 sidecar body,decode@512 | 已实现 |
| `trellis2_s1_pad` | **2** | 64³ edit grid 做 Chebyshev 方盒膨胀 2 cell(同时放大 S1 keep mask 和 S2 edit region) | 已有旋钮,默认 0 |
| `trellis2_s2_restore_preserved` | **true** | **本次新增**:S2 前,把 source slat 占据在「mask 之外的 preserved 区域」里丢掉的体素补回 coords_new | 本次实现 |

A/B 三方对照(masked S1 + contact-soft + ss_align_t1,**全部 S2=perstep**(纯 mask 逐步锚定),唯一变量是 pad/restore):

- **控制组 pad0**(`_exp_masked_perstep_r512_pad0`):512 + pad=0 + 无补救 ← 干净基线
- **e512-pad2**(`_exp_masked_perstep_r512_pad2`):512 + pad=2 + 无补救 ← 隔离「膨胀」的净效果
- **e512-pad2-restore**(`_exp_masked_perstep_r512_pad2_restore`):512 + pad=2 + 补救 ← 完整方案

对应 config:`pipeline_v3_trellis2_masked_perstep_r512_{pad0,pad2,pad2_restore}.yaml`。

> S2 用 perstep:不走 posthoc 的「自由生成 + 末尾粘回 body」,而是**每一步**都把 preserved
> token 硬锚回反演原 latent。补回的体素(restore)在 mask 外 → 自动属 preserved → 每步都被
> 锚回 source,占据与 latent 一并保真。
> ⚠️ `_exp_masked_posthoc_r512_pad0` 与 1024 基线是 **posthoc** S2,属另一条线,勿与本组混比。

## 补救机制(核心新增)

### 动机

`pad=2` 把 edit region 向外膨胀 2 个 cell,16³ keep mask 也随之扩大 → S1 重绘时会把
**紧贴 part 边界的 body 体素**也划进编辑区/丢掉;再叠加 16³→keep 的降采样侵蚀,decode
出来的 body 容易出现破洞或缺块。补救的思路:**body 的占据(几何)应当无损保留**,只让
part 区域自由生长。

### 判定逻辑(同帧 64³ SS,**已实现版本**)

⚠️ 早期版本在 **32³** 上拿 shape-VAE sidecar(`shape_slat_e512.npz`)当参照 union——
但 sidecar 与 S1 的 SS 占据来自**不同编码器**,body 边界天然不重合(实测 c0only 58~791
体素),补回的体素飘在 S1 没占据的地方 → mesh **错位/连不上**。已废弃。

**当前实现:同一编码器帧(64³ SS)。** `edit_structure(return_orig_occ=True)` 额外用**同一个
`ss_dec`** 解码原始 SS latent `z_s0`,得到**编辑前的同帧占据** `coords_orig_ss`(与 `coords_new
= ss_dec(z_s_new)` 完全可比):

```
# 都在 64³ SS 解码帧;edit_grid 也是 64³(未降采样)
preserved_src = { c ∈ coords_orig_ss : edit_grid64[c] == False }   # 原始 body(mask 外)
missing       = { c ∈ preserved_src : c ∉ coords_new }             # 编辑里被丢的 body
coords_new   := coords_new ∪ missing                                # 64³ union
# 再 _to_s2:64³ → //2 → 32³ 喂 S2
```

因为两边都是 `ss_dec` 输出,补回的体素**保证与 coords_new 同帧、相邻/对齐**,不会飘空。
后续 `build_coord_bridge`(在 32³ 上、参照 sidecar coords0)对其中落在 sidecar 里的锚定到
source latent,其余在 perstep 下随场生成——占据先合上,几何再去噪,无错位碎片。

### 作用网格:64³(S1 之后、`_to_s2` 之前)

在 64³ 上 union(同 SS 帧)再 //2 到 32³,而**不是**在 32³ 上对 sidecar union。这正是之前
讨论的「64」选项,实测能把占据并成**单连通块**(见验证)。

## 改动清单

1. `partcraft/pipeline_v3/trellis2_part_mask.py`
   - 新增 `restore_preserved_occupancy(coords0, coords_new, edit_grid, grid=GRID_LO)`
     → 返回 `(coords_new_union[int32], n_restored[int])`。加入 `__all__`。

2. `partcraft/pipeline_v3/trellis2_3d.py`
   - import `restore_preserved_occupancy`。
   - 在 `_build_p4_mesh` 主 S1+S2 路径的 `_to_s2(...)` 之后、`masked_shape_slat` 之前,
     读 `trellis2_s2_restore_preserved`(默认 False),开则补救并 log `+N source voxels`。
   - 把开关记入 `_collect_edit_latents`(便于复现)。

3. `configs/pipeline_v3_trellis2_masked_perstep_r512_pad2_restore.yaml`(新)
   - 拷贝 `pipeline_v3_trellis2_masked_posthoc_r512_pad0.yaml`,加
     `trellis2_s1_pad: 2`、`trellis2_s2_restore_preserved: true`,
     `output_dir: data/Pxform_v2/_exp_masked_perstep_r512_pad2_restore`。
   - 同时给「不补救」对照留一份 `..._e512_pad2.yaml`(pad=2,restore off)。

4. `docs/experiment/trellis2_3d_edit_recipes.md` §4 旋钮表新增
   `trellis2_s2_restore_preserved`。

## 验证(逐步 smoke,每步等确认)

1. **单元函数 smoke**:构造小 `coords0/coords_new/edit_grid`,验证 `restore_preserved_occupancy`
   只补「mask 外、source 有、after 无」的体素,edit 区内与已存在体素不动。
2. **管线 smoke(1 物体)**:
   `STAGES=trellis2_encode LIMIT=1 FORCE=1 ... masked_perstep_r512_pad2_restore.yaml`
   确认 `shape_slat_e512.npz`(32³)写出;再 `STAGES=trellis2_preview LIMIT=1 FORCE=1 ...`
   日志里应看到 `S1 structure: N→M`、`S2 restore-preserved: +K source voxels`、
   `decode_latent res=512` 不崩、`edits_3d/<id>` 生成。
3. **A/B 全量**:三套 config(baseline-1024 已有 / e512-pad2 / e512-pad2-restore)各跑
   `STAGES=trellis2_encode,trellis2_preview FORCE=1 OBJ_IDS_FILE=...smoke ids`。
4. **可视化**:用现成 `scripts/viz/compare_ab_edits.py` / `decode_res_compare.py` 把三套 after
   视图并排,重点看 body 破洞是否被补救修复、part 在 pad=2 下是否更完整。

## 风险 / 备注

- 补救只 union 占据 + 走 bridge 锚定,**不**改 part 区域,所以不会污染编辑结果。
- 32³ 下补救粒度较粗:若 S1 在 body 大块丢失,union 回来的是 source sidecar 的 32³ 占据,
  与 S1 降采样占据在边界可能不逐 voxel 对齐 —— 属可接受近似(与现有 part_mask 16×块启发同源)。
- `pad=2` 在 16³ keep mask 上等于把编辑区扩约半个 SS-cell,需观察是否把 part 重绘成糊块
  (这正是默认 pad=0 的原因);补救机制是为抵消它对 body 的副作用而引入的。
