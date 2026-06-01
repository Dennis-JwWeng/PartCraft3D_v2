# Vinedresser3D 从 TRELLIS.1 迁移到 TRELLIS.2 的编辑数据管线开发文档

## 0. 目标

本文档描述如何将 **Vinedresser3D 基于 TRELLIS.1 的 3D 编辑数据生成管线**，迁移到 **基于 TRELLIS.2 表征与生成流程** 的版本。

迁移目标不是简单替换模型 checkpoint，而是把 Vinedresser3D 的核心思想：

> 保留未编辑区域，对编辑 mask 内的 3D 表征进行 inversion-based rectified-flow inpainting / flow matching 编辑

从 TRELLIS.1 的 **SLAT sparse voxel latent** 迁移到 TRELLIS.2 的 **O-Voxel + Sparse Compression VAE + shape/material latent** 框架中。

最终目标是构建一个可批量产出编辑数据的 pipeline：

```text
original 3D asset + edit prompt
        ↓
auto parsing / grounding / mask construction
        ↓
TRELLIS.2 latent-space masked flow editing
        ↓
edited asset + masks + metadata + renders + optional latent pairs
```

---

## 1. 背景：Vinedresser3D 的 TRELLIS.1 编辑逻辑

Vinedresser3D 的核心是一个 agentic 3D editing pipeline。输入原始 3D asset 和自然语言编辑指令后，它会：

1. 用 MLLM 分析原始资产和编辑意图；
2. 生成原始描述、新描述、编辑部位、编辑类型；
3. 自动选择 informative view；
4. 调用 image editing model 生成 2D visual guidance；
5. 用 3D grounding / segmentation 得到编辑区域；
6. 在 TRELLIS.1 的 3D latent space 中进行 inversion-based rectified-flow inpainting；
7. 在 flow sampling 的每一步保留 mask 外区域，只对 mask 内区域进行编辑。

Vinedresser3D 最关键的工程思想是：

```python
z_t_pred = flow_step(z_t, cond_new, t)
z_t = mask * z_t_pred + (1 - mask) * z_t_orig_trajectory[t]
```

也就是：

- mask 内：根据新 prompt / 新 visual guidance 生成；
- mask 外：每一步都替换回原始 asset inversion trajectory 中对应 timestep 的 latent；
- 这样可以最大化保留未编辑区域的 geometry 和 appearance。

在 TRELLIS.1 中，编辑对象主要是：

```text
sparse structure coords
SLAT latent features
```

输出通常是：

```text
*_edited_coords.pt
*_edited_feats.pt
```

---

## 2. TRELLIS.1 与 TRELLIS.2 表征差异

### 2.1 TRELLIS.1：SLAT 表征

TRELLIS.1 使用 Structured LATent，简称 SLAT。一个 3D asset 被表示为：

```text
z = {(z_i, p_i)}_{i=1}^{L}
```

其中：

- `p_i` 是 active voxel 在稀疏 3D grid 中的位置；
- `z_i` 是绑定在该 active voxel 上的 local latent；
- active voxels 表达粗结构；
- local latent 表达局部几何与外观细节。

TRELLIS.1 的 SLAT 由多视角渲染图像聚合得到。具体流程是：

```text
3D asset
  → render dense multiview images
  → DINOv2 feature extraction
  → project / aggregate features to active voxels
  → Sparse VAE encode
  → SLAT
```

SLAT 可以被不同 decoder 解码为：

```text
3D Gaussian
Radiance Field
Mesh
```

TRELLIS.1 的生成流程是两阶段：

```text
Stage 1: sparse structure generation
Stage 2: structured latent generation
```

也就是说，先生成 active voxel structure，再在这些 active voxels 上生成 latent features。

---

### 2.2 TRELLIS.2：O-Voxel + Sparse Compression VAE

TRELLIS.2 的表征核心变成 **O-Voxel**，它是一个 native 3D sparse voxel representation。每个 active voxel 直接包含 shape 和 material 信息：

```text
f = {(f_i^shape, f_i^mat, p_i)}_{i=1}^{L}
```

其中：

```text
f_i^shape = geometry-related feature
f_i^mat   = material-related feature
p_i       = active voxel coordinate
```

TRELLIS.2 的 shape 表达基于 **Flexible Dual Grid**，主要包含：

```text
dual vertex
edge intersection flags
splitting weights
```

TRELLIS.2 的 material 表达包含 PBR 属性：

```text
base color
metallic
roughness
alpha / opacity
```

因此，TRELLIS.2 不只是“更高分辨率的 TRELLIS.1”，而是从多视角视觉特征驱动的 latent，变成了更 native、更显式、更 compact 的 3D shape/material latent。

---

### 2.3 表征差异总表

| 维度 | TRELLIS.1 | TRELLIS.2 |
|---|---|---|
| 核心表征 | SLAT：active voxel + local latent | O-Voxel：shape feature + material feature + active voxel |
| 数据来源 | 多视角渲染图 + DINOv2 feature aggregation | native mesh / texture / PBR 直接转换 |
| 几何表达 | coarse sparse structure + latent detail | Flexible Dual Grid，支持 open / non-manifold / enclosed structures |
| 外观表达 | latent 中隐式表达 appearance | 显式 PBR material：base color、metallic、roughness、alpha |
| latent 结构 | 单套 SLAT latent | shape latent 与 material latent 更清晰解耦 |
| 生成阶段 | sparse structure → SLAT feature | sparse structure → geometry latent → material latent |
| 编辑 mask | structure mask + SLAT feature mask | structure mask + geometry latent mask + material latent mask |
| 输出资产 | 3DGS / mesh / RF decoder | O-Voxel → mesh + PBR material |
| 迁移重点 | 对 SLAT token inpaint | 对 O-Voxel-derived latent 分层 inpaint |

---

## 3. 资产生成阶段差异

### 3.1 TRELLIS.1 生成流程

TRELLIS.1 的生成流程：

```text
condition: text / image
        ↓
Stage 1: sparse structure flow model
        ↓
active voxel coords
        ↓
Stage 2: sparse latent flow model
        ↓
SLAT features
        ↓
decoder
        ↓
3DGS / RF / mesh
```

对应到 Vinedresser3D 的编辑时：

```text
原始 asset
  → encode to SLAT
  → invert flow trajectory
  → masked inpainting in SLAT space
  → decode edited SLAT
```

### 3.2 TRELLIS.2 生成流程

TRELLIS.2 更适合被拆成三层：

```text
condition: text / image
        ↓
Stage 1: sparse structure generation
        ↓
Stage 2: geometry generation
        ↓
Stage 3: material generation
        ↓
O-Voxel / mesh / PBR asset
```

其中：

- structure 决定哪里有 active voxels；
- geometry latent 决定局部形状、拓扑、surface；
- material latent 决定 base color、metallic、roughness、alpha 等材质。

### 3.3 对编辑的直接影响

TRELLIS.1 编辑时主要需要处理：

```text
coords mask
feature mask
```

TRELLIS.2 编辑时需要处理：

```text
structure mask
geometry latent mask
material latent mask
```

因此，迁移后的编辑模块应该支持分层编辑：

| 编辑类型 | structure | geometry | material |
|---|---:|---:|---:|
| 删除部件 | 必须改 | 必须改 | 清除对应区域 |
| 添加部件 | 必须改 | 必须改 | 必须生成 |
| 替换形状 | 通常要改 | 必须改 | 视情况重采样 |
| 姿态/局部几何修改 | 可能要改 | 必须改 | 可保留或轻微更新 |
| 改颜色 | 不改 | 不改 | 必须改 |
| 改材质 | 不改 | 不改 | 必须改 |
| 改透明度 | 不改 | 不改 | 改 alpha |
| 改金属/粗糙度 | 不改 | 不改 | 改 metallic / roughness |

---

## 4. 迁移总览

### 4.1 原 Vinedresser3D / TRELLIS.1 管线

```text
input asset
  ↓
TRELLIS.1 encode
  ↓
SLAT coords / feats
  ↓
MLLM prompt parsing
  ↓
3D grounding mask
  ↓
RF inversion
  ↓
masked flow inpainting
  ↓
edited coords / feats
  ↓
TRELLIS.1 decode
  ↓
edited asset
```

### 4.2 迁移后 TRELLIS.2 管线

```text
input asset
  ↓
asset normalization
  ↓
mesh / texture / PBR extraction
  ↓
O-Voxel conversion
  ↓
Sparse Compression VAE encode
  ↓
z_structure / z_geometry / z_material
  ↓
MLLM prompt parsing
  ↓
3D grounding mask
  ↓
mask projection to structure / geometry / material latent spaces
  ↓
RF inversion for required stages
  ↓
masked flow inpainting by stage
  ↓
Sparse Compression VAE decode
  ↓
O-Voxel to mesh + PBR
  ↓
edited asset
```

---

## 5. 模块级替换方案

### 5.1 Asset Encoder 替换

#### TRELLIS.1 旧逻辑

```text
asset
  → render multiview
  → DINOv2 feature aggregation
  → active voxel coords + SLAT feats
```

#### TRELLIS.2 新逻辑

```text
asset mesh + texture / material
  → normalize
  → mesh → O-Voxel shape
  → texture / PBR → O-Voxel material
  → Sparse Compression VAE encode
  → z_shape, z_material
```

建议新建模块：

```text
trellis2_adapter/
  asset_normalizer.py
  ov_converter.py
  vae_encoder.py
  vae_decoder.py
```

核心接口：

```python
class Trellis2AssetEncoder:
    def encode(self, asset_path: str) -> dict:
        return {
            "ovoxel": ov_data,
            "structure": structure_data,
            "z_shape": z_shape,
            "z_material": z_material,
            "meta": asset_meta,
        }
```

---

### 5.2 Mask 构造与投影替换

#### TRELLIS.1 旧逻辑

```text
PartField / segmentation
  → edit part points
  → voxel mask
  → stage1 mask
  → stage2 SLAT mask
```

#### TRELLIS.2 新逻辑

```text
PartField / segmentation
  → edit part points / triangles / voxels
  → O-Voxel resolution edit mask
  → structure mask
  → geometry latent mask
  → material latent mask
```

建议统一 mask 数据结构：

```python
@dataclass
class Trellis2EditMasks:
    ov_mask: Any              # high-res O-Voxel edit mask
    structure_mask: Any       # mask for sparse structure generation
    geometry_mask: Any        # mask for shape latent
    material_mask: Any        # mask for material latent
    preserve_mask: Any        # optional explicit preserve region
```

mask 投影建议不要只用 nearest downsample，而是用 overlap ratio：

```python
editable = overlap(latent_cell_receptive_field, edit_region) > tau_edit
preserve = overlap(latent_cell_receptive_field, preserve_region) > tau_pres
```

推荐参数：

```text
tau_edit_geometry  = 0.15 ~ 0.30
tau_edit_material  = 0.05 ~ 0.20
tau_preserve       = 0.50 ~ 0.80
geometry dilation  = 1 ~ 2 latent cells
material dilation  = 2 ~ 4 latent cells
```

material mask 可以比 geometry mask 更软、更大，因为材质变化通常需要边界过渡。

---

### 5.3 Flow Inversion 替换

#### TRELLIS.1 旧逻辑

```text
invert SLAT feature trajectory
maybe invert structure trajectory
```

#### TRELLIS.2 新逻辑

需要按 stage 分别 inversion：

```text
structure trajectory: z_struct_orig(t)
geometry trajectory:  z_geo_orig(t)
material trajectory:  z_mat_orig(t)
```

不是每种编辑都需要 invert 所有 stage。

推荐策略：

| 编辑类型 | 需要 inversion 的 stage |
|---|---|
| 删除 | structure + geometry，material optional |
| 添加 | structure + geometry + material |
| 几何替换 | structure + geometry + material optional |
| 颜色修改 | material only |
| 材质修改 | material only |
| 局部纹理替换 | material only |
| 透明度修改 | material only |

接口建议：

```python
class Trellis2FlowInverter:
    def invert_structure(self, structure, cond_orig) -> Trajectory:
        ...

    def invert_geometry(self, z_shape, structure, cond_orig) -> Trajectory:
        ...

    def invert_material(self, z_material, z_shape, cond_orig) -> Trajectory:
        ...
```

---

### 5.4 Masked Flow Editing 替换

#### 通用形式

对每个 stage，都保留 Vinedresser3D 的核心 masked replacement 思路：

```python
z_pred = flow_step(model, z_t, cond_edit, t)

z_t = mask * z_pred + (1.0 - mask) * z_orig_traj[t]
```

#### TRELLIS.2 分层形式

```python
# 1. structure editing
z_struct_edit = masked_flow_edit(
    model=structure_model,
    z_init=z_struct_noise,
    cond=cond_structure_edit,
    mask=structure_mask,
    orig_traj=structure_orig_traj,
)

# 2. geometry editing
z_shape_edit = masked_flow_edit(
    model=geometry_model,
    z_init=z_shape_noise,
    cond=cond_geometry_edit,
    mask=geometry_mask,
    orig_traj=geometry_orig_traj,
    extra_cond={"structure": z_struct_edit},
)

# 3. material editing
z_material_edit = masked_flow_edit(
    model=material_model,
    z_init=z_material_noise,
    cond=cond_material_edit,
    mask=material_mask,
    orig_traj=material_orig_traj,
    extra_cond={"geometry": z_shape_edit},
)
```

### 5.5 Decoder / Export 替换

#### TRELLIS.1 旧输出

```text
edited_coords.pt
edited_feats.pt
decoded 3DGS / mesh
```

#### TRELLIS.2 新输出

```text
edited_z_shape.pt
edited_z_material.pt
edited_ovoxel
edited_mesh.glb
edited_material / texture maps
metadata.json
```

建议每条数据同时保存 latent 与最终 asset，方便后续训练不同任务。

---

## 6. 分编辑类型的具体策略

### 6.1 Addition：添加新部件

例子：

```text
Add a billboard on the cart.
Add vegetables in the basket.
```

推荐策略：

```text
structure: allow new active voxels inside edit expansion region
geometry: generate new shape latent
material: generate new material latent
preserve: mask 外每步替换原 trajectory
```

关键点：

- edit mask 不能只包含原有 asset 表面，因为新增物体通常位于原资产外部；
- 需要构造 candidate addition region；
- 可以参考 Vinedresser3D 对 addition 的思路：从编辑部位附近扩展空间，而不是全局开放生成；
- TRELLIS.2 中 structure stage 必须开放新增 active voxels，否则 geometry/material stage 没地方生成。

建议：

```text
R_add = dilate(anchor_part_region, radius)
R_add = R_add - preserve_region
```

---

### 6.2 Deletion：删除部件

例子：

```text
Remove the canopy of the cart.
Remove the roof of the horse cart.
```

推荐策略：

```text
structure: remove active voxels in deletion region
geometry: inpaint deletion boundary
material: clear or regenerate boundary material
```

关键点：

- 删除不能只 mask material；
- 对 boundary 附近 geometry 需要轻微 inpainting，避免洞口/破面；
- material mask 应覆盖 geometry deletion mask 的边界 dilation 区域。

---

### 6.3 Modification：替换局部形状

例子：

```text
Change the toy car to a train.
Change the carriage to a load of watermelons.
```

推荐策略：

```text
structure: 根据变化幅度决定是否编辑
geometry: 必须编辑
material: 通常编辑
preserve: preserved part 强约束
```

关键点：

- 对大形状替换，structure mask 需要比原 edit part 更大；
- 对小局部形变，可以 preserve structure，只编辑 geometry latent；
- modification 最容易污染未编辑区域，因此 preserve mask 要比 edit mask 更严格。

---

### 6.4 Appearance / Material Editing：只改外观

例子：

```text
Change the purple flower to a white dandelion.
Make the metal part rougher.
Make the glass transparent.
```

推荐策略：

```text
structure: keep original
geometry: keep original
material: masked flow edit only
```

关键点：

- 这是 TRELLIS.2 相比 TRELLIS.1 最值得利用的优势；
- 颜色、材质、透明度、粗糙度、金属感都应该尽量限制在 material latent；
- 不要重新跑 structure / geometry generation，否则会引入不必要的形状漂移。

---

## 7. 条件构造：Prompt 从两阶段改成三阶段

Vinedresser3D 会用 MLLM 生成 stage-specific guidance。迁移后建议生成三类 prompt：

```json
{
  "original_description": "...",
  "new_description": "...",
  "edit_type": "addition | deletion | modification | material",
  "edit_part_names": ["..."],
  "structure_prompt": "...",
  "geometry_prompt": "...",
  "material_prompt": "...",
  "preserve_prompt": "..."
}
```

示例：

```json
{
  "edit_prompt": "Change the toy car to a train.",
  "edit_type": "modification",
  "edit_part_names": ["car-like body", "wheels"],
  "structure_prompt": "A toy with a locomotive-style train body while preserving the white figure on top.",
  "geometry_prompt": "Locomotive-style train body with train wheels and a light tan base.",
  "material_prompt": "Glossy red and blue train body, light tan bottom, white figure, yellow beak, orange crest.",
  "preserve_prompt": "Preserve the white spherical figure, beak, eyes, and orange crest."
}
```

对 TRELLIS.2 来说，`material_prompt` 很重要，因为 material stage 可以更明确地约束 PBR 属性。

---

## 8. 数据产出格式设计

建议每条编辑样本保存为一个目录：

```text
sample_000001/
  input/
    original_asset.glb
    original_renders/
    original_ovoxel.pt
    z_shape_orig.pt
    z_material_orig.pt

  guidance/
    edit_prompt.txt
    parsed_prompt.json
    selected_view.png
    edited_view.png

  masks/
    edit_mask_ovoxel.pt
    preserve_mask_ovoxel.pt
    structure_mask.pt
    geometry_mask.pt
    material_mask.pt

  trajectories_optional/
    structure_orig_traj.pt
    geometry_orig_traj.pt
    material_orig_traj.pt

  output/
    edited_asset.glb
    edited_renders/
    edited_ovoxel.pt
    z_shape_edit.pt
    z_material_edit.pt

  metadata.json
```

`metadata.json` 推荐字段：

```json
{
  "sample_id": "sample_000001",
  "source_asset": "...",
  "edit_prompt": "...",
  "edit_type": "modification",
  "edit_part_names": ["..."],
  "original_description": "...",
  "new_description": "...",
  "structure_prompt": "...",
  "geometry_prompt": "...",
  "material_prompt": "...",
  "mask_stats": {
    "ovoxel_edit_ratio": 0.12,
    "structure_edit_ratio": 0.08,
    "geometry_edit_ratio": 0.10,
    "material_edit_ratio": 0.15
  },
  "model_versions": {
    "trellis2": "...",
    "segmenter": "...",
    "image_editor": "...",
    "mllm": "..."
  },
  "quality_checks": {
    "preserve_region_chamfer": null,
    "render_clip_score": null,
    "mask_leakage_score": null
  }
}
```

---

## 9. 推荐代码结构

可以在 Vinedresser3D repo 外部新增一个 adapter 层，避免直接大改原 pipeline：

```text
vinedresser3d_trellis2/
  configs/
    default.yaml
    edit_types.yaml

  trellis2_adapter/
    __init__.py
    asset_encoder.py
    asset_decoder.py
    ov_converter.py
    vae_wrapper.py
    flow_wrapper.py

  masks/
    grounding.py
    ov_mask_builder.py
    latent_mask_projector.py
    mask_ops.py

  editing/
    planner.py
    inversion.py
    masked_flow_edit.py
    stage_router.py

  data/
    sample_schema.py
    exporter.py
    validator.py

  scripts/
    encode_asset_trellis2.py
    run_edit_trellis2.py
    batch_generate_edits.py
    render_compare.py
```

核心入口：

```bash
python scripts/run_edit_trellis2.py \
  --asset path/to/original.glb \
  --prompt "Change the toy car to a train" \
  --out_dir outputs/sample_000001 \
  --edit_type auto
```

---

## 10. Stage Router 设计

TRELLIS.2 迁移时建议加一个 `StageRouter`，根据 edit type 自动决定跑哪些 stage。

```python
class StageRouter:
    def route(self, edit_type: str, prompt_info: dict) -> dict:
        if edit_type == "material":
            return {
                "edit_structure": False,
                "edit_geometry": False,
                "edit_material": True,
            }

        if edit_type == "deletion":
            return {
                "edit_structure": True,
                "edit_geometry": True,
                "edit_material": True,
            }

        if edit_type == "addition":
            return {
                "edit_structure": True,
                "edit_geometry": True,
                "edit_material": True,
            }

        if edit_type == "modification":
            return {
                "edit_structure": "auto",
                "edit_geometry": True,
                "edit_material": True,
            }

        return {
            "edit_structure": True,
            "edit_geometry": True,
            "edit_material": True,
        }
```

对 `"auto"` 的判断可以基于：

```text
edit region bbox size
semantic category change
estimated topology change
whether addition/deletion keywords exist
```

---

## 11. 质量控制

### 11.1 Preserve 区域一致性

重点评估 mask 外区域有没有被破坏。

可选指标：

```text
preserve-region Chamfer Distance
preserve-region normal consistency
preserve-region render LPIPS / SSIM
preserve-region material L1
```

TRELLIS.2 由于有 explicit material，可以额外计算：

```text
base color L1 outside mask
metallic L1 outside mask
roughness L1 outside mask
alpha L1 outside mask
```

### 11.2 Edit 区域有效性

评估编辑是否成功：

```text
CLIP / image-text score on edited renders
DINO similarity with edited reference image
part-level semantic classifier score
human preference
```

### 11.3 Mask leakage

建议显式记录：

```text
mask_out_geometry_change
mask_out_material_change
mask_boundary_artifact_score
```

---

## 12. 实现优先级

### Phase 1：最小可跑通版本

目标：先跑通 TRELLIS.2 latent-space 局部编辑。

只支持：

```text
material editing
geometry modification without structure change
```

实现：

```text
asset → O-Voxel → encode
mask → material / geometry latent mask
invert material / geometry flow
masked inpainting
decode → asset
```

暂时不做 addition。

---

### Phase 2：支持 deletion / local modification

增加：

```text
structure mask
structure trajectory inversion
geometry boundary inpainting
mask dilation / erosion
```

支持：

```text
remove part
change local shape
```

---

### Phase 3：支持 addition

增加：

```text
candidate addition region construction
structure expansion
new active voxel generation
geometry + material generation for newly active region
```

addition 是最难的，因为需要在原 asset 外部开放生成空间。

---

### Phase 4：批量数据生产

增加：

```text
batch prompt generation
automatic validation
render comparison
metadata export
failed sample filtering
```

---

## 13. 主要风险与解决方案

### 风险 1：mask 投影不准导致边界 artifact

解决方案：

```text
geometry mask 使用 hard mask + 小 dilation
material mask 使用 soft mask + 大 dilation
边界区域增加 blending / partial preserve
```

### 风险 2：structure 改动污染 preserved region

解决方案：

```text
structure stage 使用更严格 preserve mask
对 preserved active voxels 强制回填原 trajectory
只允许 edit region 内新增 / 删除 active voxels
```

### 风险 3：material 与 geometry 不对齐

解决方案：

```text
material generation condition on edited geometry
material mask 从 edited geometry surface 重新投影
decode 后检查 PBR texture seam
```

### 风险 4：addition 没有生成空间

解决方案：

```text
基于 anchor part bbox 扩展 candidate region
用 edited 2D view / MLLM 估计新增部件位置
在 structure stage 开放 candidate region
```

### 风险 5：纯材质编辑引入几何漂移

解决方案：

```text
纯材质编辑完全跳过 structure / geometry sampling
只编辑 material latent
decode 时复用原 geometry latent
```

---

## 14. 核心伪代码

```python
def run_trellis2_edit(asset_path, edit_prompt, out_dir):
    # 1. Encode asset
    encoded = trellis2_encoder.encode(asset_path)
    z_shape_orig = encoded["z_shape"]
    z_mat_orig = encoded["z_material"]
    structure_orig = encoded["structure"]
    ovoxel_orig = encoded["ovoxel"]

    # 2. Plan edit
    prompt_info = planner.parse(
        asset=asset_path,
        edit_prompt=edit_prompt,
    )

    edit_type = prompt_info["edit_type"]
    route = stage_router.route(edit_type, prompt_info)

    # 3. Build masks
    edit_region = grounder.ground(
        asset=asset_path,
        part_names=prompt_info["edit_part_names"],
        edit_prompt=edit_prompt,
    )

    masks = mask_projector.build_trellis2_masks(
        edit_region=edit_region,
        ovoxel=ovoxel_orig,
        structure=structure_orig,
        z_shape=z_shape_orig,
        z_material=z_mat_orig,
    )

    # 4. Structure edit
    if route["edit_structure"]:
        struct_traj = inverter.invert_structure(
            structure_orig,
            cond_orig=prompt_info["original_description"],
        )

        structure_edit = masked_flow_edit(
            model=trellis2_structure_model,
            orig_traj=struct_traj,
            mask=masks.structure_mask,
            cond_edit=prompt_info["structure_prompt"],
        )
    else:
        structure_edit = structure_orig

    # 5. Geometry edit
    if route["edit_geometry"]:
        geo_traj = inverter.invert_geometry(
            z_shape_orig,
            structure_orig,
            cond_orig=prompt_info["original_description"],
        )

        z_shape_edit = masked_flow_edit(
            model=trellis2_geometry_model,
            orig_traj=geo_traj,
            mask=masks.geometry_mask,
            cond_edit=prompt_info["geometry_prompt"],
            extra_cond={"structure": structure_edit},
        )
    else:
        z_shape_edit = z_shape_orig

    # 6. Material edit
    if route["edit_material"]:
        mat_traj = inverter.invert_material(
            z_mat_orig,
            z_shape_orig,
            cond_orig=prompt_info["original_description"],
        )

        z_mat_edit = masked_flow_edit(
            model=trellis2_material_model,
            orig_traj=mat_traj,
            mask=masks.material_mask,
            cond_edit=prompt_info["material_prompt"],
            extra_cond={"geometry": z_shape_edit},
        )
    else:
        z_mat_edit = z_mat_orig

    # 7. Decode and export
    edited_asset = trellis2_decoder.decode(
        structure=structure_edit,
        z_shape=z_shape_edit,
        z_material=z_mat_edit,
    )

    exporter.save(
        out_dir=out_dir,
        original_asset=asset_path,
        edited_asset=edited_asset,
        prompt_info=prompt_info,
        masks=masks,
        latents={
            "z_shape_orig": z_shape_orig,
            "z_shape_edit": z_shape_edit,
            "z_material_orig": z_mat_orig,
            "z_material_edit": z_mat_edit,
        },
    )

    return edited_asset
```

---

## 15. 最终迁移结论

从 TRELLIS.1 到 TRELLIS.2 的迁移重点可以概括为一句话：

> 把 Vinedresser3D 的 SLAT-level masked flow inpainting，升级为 O-Voxel-derived structure / geometry / material 三层 latent 的 masked flow inpainting。

具体来说：

```text
TRELLIS.1:
  sparse coords + SLAT feats
  → one structure stage + one latent stage

TRELLIS.2:
  O-Voxel + SC-VAE
  → structure stage + geometry stage + material stage
```

保留的思想：

```text
MLLM planning
automatic grounding
editing mask
inversion trajectory
mask 外区域每步回填原 trajectory
mask 内区域 flow matching 编辑
```

必须替换的部分：

```text
SLAT encoder / decoder
coords-feats 数据格式
single feature mask
two-stage edit routing
```

替换为：

```text
O-Voxel converter
Sparse Compression VAE encoder / decoder
structure / geometry / material masks
three-stage edit routing
PBR-aware material editing
```

最推荐的开发策略是：

1. 先做 **material-only editing**，验证 TRELLIS.2 material latent 的局部可控性；
2. 再做 **geometry modification without structure change**；
3. 然后做 **deletion**；
4. 最后做最难的 **addition**。

这样迁移风险最低，也最能发挥 TRELLIS.2 的 native shape/material 解耦优势。
