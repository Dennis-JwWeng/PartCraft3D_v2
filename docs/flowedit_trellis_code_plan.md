# FlowEdit × TRELLIS 两阶段代码思路文档

> 目标：把 FlowEdit 的“source/target 双分支速度差”思想迁移到 TRELLIS 风格的 3D 生成流程中，重点记录 **Stage 1 sparse-structure / occupancy latent** 和 **Stage 2 geometry SLAT** 的代码组织方式。  
> 注意：这是代码设计与伪代码文档，不是可直接运行的完整实现；实际函数名需要替换成你使用的 TRELLIS repo/API。

---

## 1. 背景与核心假设

TRELLIS 类 3D 生成器通常可以抽象成三段：

1. **Stage 1：Sparse-Structure / Occupancy**  
   输入图像条件 `c`，通过 dense low-resolution latent `z_ss` 生成 occupancy grid / active coordinates。  
   典型 latent 形状可理解为：

   ```python
   x_ss: Tensor[B, C_ss, R, R, R]   # 例如 R=64
   ```

2. **Stage 2：Geometry SLAT**  
   在 Stage 1 得到的 active coordinates `V` 上运行 sparse DiT，生成几何结构 latent。  
   典型形状可理解为：

   ```python
   coords: Tensor[N, 3]
   x_geo: Tensor[B, N, C_geo]
   ```

3. **Stage 3：Material SLAT**  
   同样在 sparse coordinates 上生成材质 / PBR latent。本文档暂时不展开 Stage 3。

FlowEdit 的核心是：不把 source latent inversion 到 noise，而是从 clean source latent `x_src` 直接出发，用 source branch 和 target branch 的 **guided velocity difference** 推动 latent 走向编辑结果。

---

## 2. FlowEdit 核心公式转成代码接口

### 2.1 Rectified Flow 的速度预测

每个 TRELLIS DiT 都可以看作一个 velocity model：

```python
v = model(x_t, t, cond)
```

其中：

- `x_t`：当前时间步的 noisy latent / interpolated latent；
- `t`：flow 时间，通常从 `1 -> 0` 积分；
- `cond`：图像条件，例如 DINO feature / image embedding；
- `v`：velocity prediction。

### 2.2 CFG 封装

FlowEdit 每个 branch 都使用 classifier-free guidance：

```python
def guided_velocity(model, x_t, t, cond, null_cond, omega, cfg_active=True):
    """
    model: TRELLIS 某一阶段的 rectified-flow DiT
    x_t: 当前 latent
    t: 当前时间
    cond: source 或 target image condition
    null_cond: unconditional embedding / null token
    omega: CFG scale
    cfg_active: 是否启用 CFG
    """
    v_cond = model(x_t, t, cond)

    if not cfg_active or omega == 0:
        return v_cond

    v_null = model(x_t, t, null_cond)
    v_guided = (1.0 + omega) * v_cond - omega * v_null
    return v_guided
```

### 2.3 FlowEdit 双分支 coupling

核心 coupling：

```python
def make_flowedit_pair(x_src, z_edit, t, eps):
    """
    x_src: clean source latent
    z_edit: 当前编辑状态，初始为 x_src
    t: 当前 flow 时间
    eps: shared noise，与 x_src 同形状

    返回：
    z_src_t: source branch 的插值 latent
    z_tgt_t: target branch 的编辑 latent
    """
    z_src_t = (1.0 - t) * x_src + t * eps
    z_tgt_t = z_edit + (z_src_t - x_src)
    return z_src_t, z_tgt_t
```

这里的关键点是：

```text
z_tgt_t - z_src_t = z_edit - x_src
```

也就是说，target branch 和 source branch 的差值始终等于当前累计的编辑 offset。

### 2.4 velocity difference

```python
def flowedit_delta_velocity(
    model,
    x_src,
    z_edit,
    t,
    eps,
    c_src,
    c_tgt,
    null_cond,
    omega_src,
    omega_tgt,
    cfg_active=True,
):
    z_src_t, z_tgt_t = make_flowedit_pair(x_src, z_edit, t, eps)

    v_src = guided_velocity(
        model=model,
        x_t=z_src_t,
        t=t,
        cond=c_src,
        null_cond=null_cond,
        omega=omega_src,
        cfg_active=cfg_active,
    )

    v_tgt = guided_velocity(
        model=model,
        x_t=z_tgt_t,
        t=t,
        cond=c_tgt,
        null_cond=null_cond,
        omega=omega_tgt,
        cfg_active=cfg_active,
    )

    v_delta = v_tgt - v_src
    return v_delta
```

---

## 3. 总体工程结构建议

可以把实现拆成下面几个文件：

```text
flowedit_trellis/
├── configs.py                  # 超参数配置
├── conditions.py               # source/target image feature 编码
├── flowedit_core.py            # CFG、coupling、delta velocity
├── stage1_occupancy_edit.py    # 一阶段 FlowEdit
├── stage2_geometry_edit.py     # 二阶段 FlowEdit / FlowEdit-like
├── slat_alignment.py           # sparse coordinate 对齐工具
├── decode.py                   # occupancy / geometry decode
└── run_edit.py                 # 主入口
```

---

## 4. 配置结构

```python
from dataclasses import dataclass

@dataclass
class FlowEditConfig:
    # common
    num_steps: int = 25
    num_mc_samples: int = 5
    omega_src: float = 1.5
    omega_tgt: float = 9.0
    cfg_t_min: float = 0.6
    cfg_t_max: float = 1.0
    seed: int = 1234

    # active window，可只在部分 step 上编辑
    active_start_idx: int = 0
    active_end_idx: int = 12

    # Stage 1
    stage1_latent_resolution: int = 64

    # Stage 2
    allow_stage2_direct_flowedit: bool = True
    stage2_support_mode: str = "same_or_target_scaffold"
    # 可选："same_only" | "target_scaffold" | "intersection_only"

    # numerical safety
    delta_clip_norm: float | None = None
```

时间表：

```python
def build_time_schedule(num_steps: int):
    # t: 1 -> 0
    # 注意 dt 通常是负数：t_next - t_cur
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    return ts


def is_cfg_active(t, cfg_t_min, cfg_t_max):
    return cfg_t_min <= float(t) <= cfg_t_max
```

---

## 5. 条件输入：source condition 与 target condition

FlowEdit 需要两个条件：

```python
c_src = image_encoder(source_render_image)
c_tgt = image_encoder(target_edited_image)
```

其中：

- `source_render_image`：从原始 3D asset 渲染出来的参考图；
- `target_edited_image`：2D 编辑后的目标图，或者由文本编辑得到的目标视图；
- `image_encoder`：TRELLIS 使用的图像特征编码器，例如 DINO 类 feature encoder；
- `null_cond`：unconditional embedding / null image feature。

推荐封装：

```python
def prepare_conditions(trellis, source_image, target_image):
    c_src = trellis.encode_image_condition(source_image)
    c_tgt = trellis.encode_image_condition(target_image)
    null_cond = trellis.get_null_condition()
    return c_src, c_tgt, null_cond
```

---

## 6. Stage 1：Occupancy latent 上的 FlowEdit

### 6.1 输入输出

输入：

```python
source_asset      # 原始 3D asset / mesh / glb
source_image      # source render
edited_image      # target image condition
stage1_model      # TRELLIS stage-1 sparse-structure DiT
stage1_encoder    # source asset -> x_ss_src 的 encoder / voxelizer / latent encoder
stage1_decoder    # z_ss_edit -> occupancy coordinates
```

输出：

```python
z_ss_edit: Tensor[B, C_ss, R, R, R]
C_tgt: Tensor[N_tgt, 3]  # edited occupancy active coordinates
```

### 6.2 一阶段主流程

```python
@torch.no_grad()
def flowedit_stage1_occupancy(
    stage1_model,
    stage1_decoder,
    x_ss_src,
    c_src,
    c_tgt,
    null_cond,
    cfg: FlowEditConfig,
):
    """
    在 TRELLIS Stage 1 dense occupancy latent 上做 FlowEdit。
    """
    ts = build_time_schedule(cfg.num_steps).to(x_ss_src.device)
    z_edit = x_ss_src.clone()

    generator = torch.Generator(device=x_ss_src.device)
    generator.manual_seed(cfg.seed)

    for k in range(cfg.num_steps):
        t_cur = ts[k]
        t_next = ts[k + 1]
        dt = t_next - t_cur

        # 可选：只在 active window 做编辑，其余 step 可以跳过或走普通 target sampling
        if not (cfg.active_start_idx <= k < cfg.active_end_idx):
            continue

        cfg_active = is_cfg_active(t_cur, cfg.cfg_t_min, cfg.cfg_t_max)

        v_deltas = []
        for s in range(cfg.num_mc_samples):
            eps = torch.randn_like(x_ss_src, generator=generator)

            v_delta = flowedit_delta_velocity(
                model=stage1_model,
                x_src=x_ss_src,
                z_edit=z_edit,
                t=t_cur,
                eps=eps,
                c_src=c_src,
                c_tgt=c_tgt,
                null_cond=null_cond,
                omega_src=cfg.omega_src,
                omega_tgt=cfg.omega_tgt,
                cfg_active=cfg_active,
            )

            if cfg.delta_clip_norm is not None:
                v_delta = clip_by_global_norm(v_delta, cfg.delta_clip_norm)

            v_deltas.append(v_delta)

        v_delta_mean = torch.stack(v_deltas, dim=0).mean(dim=0)
        z_edit = z_edit + dt * v_delta_mean

    C_tgt = stage1_decoder.decode_to_coordinates(z_edit)
    return z_edit, C_tgt
```

### 6.3 Stage 1 需要重点检查的点

1. **`dt` 的符号**  
   如果时间表是 `1 -> 0`，那么 `dt = t_next - t_cur` 是负数。需要确认你使用的 TRELLIS sampler 的 velocity 方向是否与论文公式一致。

2. **`model(x_t, t, cond)` 的时间格式**  
   有的 repo 需要 `t` 是 `[B]` shape，有的需要 sigma / timestep index。

3. **source latent 的来源**  
   如果 TRELLIS 没有直接提供 `source asset -> stage1 clean latent` 的 encoder，需要通过 voxelize + latent encode 或复用 pipeline 中的中间结果。

4. **FlowEdit 本身不保证局部保持**  
   原始 FlowEdit 的 `v_delta` 在非编辑区域也可能非零。后续可以接 RASI / PMG，或者至少加 diagnostic logging。

### 6.4 Stage 1 诊断日志

建议每一步记录：

```python
log = {
    "step": k,
    "t": float(t_cur),
    "dt": float(dt),
    "v_delta_norm": v_delta_mean.norm().item(),
    "z_edit_norm": z_edit.norm().item(),
    "num_active_voxels_after_decode": int(C_tgt.shape[0]) if k == cfg.num_steps - 1 else None,
}
```

---

## 7. Stage 2：Geometry SLAT 上的 FlowEdit 思路

### 7.1 最重要的限制

严格的 FlowEdit coupling 要求 source branch 和 target branch 在同一个 latent/token space 里运行。

在 TRELLIS Stage 2，latent 是定义在 sparse coordinates 上的：

```python
x_geo_src: Tensor[B, N_src, C_geo]
C_src: Tensor[N_src, 3]
C_tgt: Tensor[N_tgt, 3]
```

如果是 add/remove 类编辑，Stage 1 之后的 `C_tgt` 很可能与 `C_src` 不一样。此时 **不能直接把 Stage 2 当作 dense latent 做标准 FlowEdit**，因为：

```text
N_src != N_tgt
source token i 和 target token i 不一定表示同一个空间位置
```

所以 Stage 2 推荐分两种模式：

| 模式 | 适用场景 | 是否严格 FlowEdit |
|---|---|---|
| `same_support` | replacement / deformation，`C_src == C_tgt` 或几乎一致 | 是 |
| `target_scaffold` | add/remove，`C_tgt != C_src` | FlowEdit-like，不是严格 FlowEdit |

---

## 8. Stage 2-A：same support 下的直接 FlowEdit

### 8.1 输入输出

输入：

```python
C_src == C_tgt
x_geo_src: Tensor[B, N, C_geo]
stage2_model: SparseDiT
c_src, c_tgt, null_cond
```

输出：

```python
z_geo_edit: Tensor[B, N, C_geo]
```

### 8.2 伪代码

```python
@torch.no_grad()
def flowedit_stage2_geometry_same_support(
    stage2_model,
    coords,
    x_geo_src,
    c_src,
    c_tgt,
    null_cond,
    cfg: FlowEditConfig,
):
    """
    当 Stage 2 的 source/target sparse coordinates 完全一致时，
    可以在 geometry SLAT 上直接套用 FlowEdit。
    """
    ts = build_time_schedule(cfg.num_steps).to(x_geo_src.device)
    z_edit = x_geo_src.clone()

    generator = torch.Generator(device=x_geo_src.device)
    generator.manual_seed(cfg.seed + 1000)

    for k in range(cfg.num_steps):
        t_cur = ts[k]
        t_next = ts[k + 1]
        dt = t_next - t_cur

        cfg_active = is_cfg_active(t_cur, cfg.cfg_t_min, cfg.cfg_t_max)

        v_deltas = []
        for s in range(cfg.num_mc_samples):
            eps = torch.randn_like(x_geo_src, generator=generator)

            # sparse model 通常需要 coords 作为额外输入
            v_delta = flowedit_delta_velocity_sparse(
                model=stage2_model,
                coords=coords,
                x_src=x_geo_src,
                z_edit=z_edit,
                t=t_cur,
                eps=eps,
                c_src=c_src,
                c_tgt=c_tgt,
                null_cond=null_cond,
                omega_src=cfg.omega_src,
                omega_tgt=cfg.omega_tgt,
                cfg_active=cfg_active,
            )

            v_deltas.append(v_delta)

        v_delta_mean = torch.stack(v_deltas, dim=0).mean(dim=0)
        z_edit = z_edit + dt * v_delta_mean

    return z_edit
```

`sparse` 版本的 delta velocity：

```python
def flowedit_delta_velocity_sparse(
    model,
    coords,
    x_src,
    z_edit,
    t,
    eps,
    c_src,
    c_tgt,
    null_cond,
    omega_src,
    omega_tgt,
    cfg_active=True,
):
    z_src_t, z_tgt_t = make_flowedit_pair(x_src, z_edit, t, eps)

    v_src = guided_velocity_sparse(
        model=model,
        coords=coords,
        x_t=z_src_t,
        t=t,
        cond=c_src,
        null_cond=null_cond,
        omega=omega_src,
        cfg_active=cfg_active,
    )

    v_tgt = guided_velocity_sparse(
        model=model,
        coords=coords,
        x_t=z_tgt_t,
        t=t,
        cond=c_tgt,
        null_cond=null_cond,
        omega=omega_tgt,
        cfg_active=cfg_active,
    )

    return v_tgt - v_src
```

```python
def guided_velocity_sparse(model, coords, x_t, t, cond, null_cond, omega, cfg_active=True):
    v_cond = model(x_t=x_t, coords=coords, t=t, cond=cond)

    if not cfg_active or omega == 0:
        return v_cond

    v_null = model(x_t=x_t, coords=coords, t=t, cond=null_cond)
    return (1.0 + omega) * v_cond - omega * v_null
```

---

## 9. Stage 2-B：target scaffold 下的 FlowEdit-like 方案

当 `C_tgt != C_src` 时，推荐把 Stage 2 固定在 `C_tgt` 上运行。也就是说：

```text
Stage 1 先输出 edited coordinates C_tgt
Stage 2 在 C_tgt 这个 scaffold 上生成 geometry SLAT
```

问题是：source geometry latent 原本定义在 `C_src` 上，不在 `C_tgt` 上。需要先构造一个 `x_geo_src_on_tgt`。

### 9.1 构造 source latent on target scaffold

```python
def build_source_geo_on_target_scaffold(
    C_src,
    C_tgt,
    x_geo_src,
    init_new_tokens="noise",
    noise_scale=1.0,
):
    """
    把 source geometry SLAT 映射到 target coordinates 上。

    对于交集 token：
        使用 source encoding。
    对于新增 token：
        用 noise / zero / target prior 初始化。
    对于被删除 token：
        不出现在 C_tgt 中，自然丢弃。
    """
    src_map = {tuple(coord.tolist()): i for i, coord in enumerate(C_src)}

    B, _, C = x_geo_src.shape
    N_tgt = C_tgt.shape[0]
    x_out = torch.zeros(B, N_tgt, C, device=x_geo_src.device, dtype=x_geo_src.dtype)
    keep_mask = torch.zeros(N_tgt, device=x_geo_src.device, dtype=torch.bool)

    for j, coord in enumerate(C_tgt):
        key = tuple(coord.tolist())
        if key in src_map:
            i = src_map[key]
            x_out[:, j] = x_geo_src[:, i]
            keep_mask[j] = True
        else:
            if init_new_tokens == "noise":
                x_out[:, j] = noise_scale * torch.randn(B, C, device=x_geo_src.device)
            elif init_new_tokens == "zero":
                x_out[:, j] = 0
            else:
                raise ValueError(f"Unknown init_new_tokens={init_new_tokens}")

    return x_out, keep_mask
```

### 9.2 在 target scaffold 上运行 FlowEdit-like

```python
@torch.no_grad()
def flowedit_stage2_geometry_target_scaffold(
    stage2_model,
    C_src,
    C_tgt,
    x_geo_src,
    c_src,
    c_tgt,
    null_cond,
    cfg: FlowEditConfig,
):
    """
    add/remove 场景下的 Stage 2 方案。
    注意：这不是严格 FlowEdit，而是把 source latent 投影到 C_tgt 后，
    在同一个 target scaffold 上做 FlowEdit-like 更新。
    """
    x_geo_src_on_tgt, keep_mask = build_source_geo_on_target_scaffold(
        C_src=C_src,
        C_tgt=C_tgt,
        x_geo_src=x_geo_src,
        init_new_tokens="noise",
    )

    z_geo_edit = flowedit_stage2_geometry_same_support(
        stage2_model=stage2_model,
        coords=C_tgt,
        x_geo_src=x_geo_src_on_tgt,
        c_src=c_src,
        c_tgt=c_tgt,
        null_cond=null_cond,
        cfg=cfg,
    )

    return z_geo_edit, keep_mask
```

### 9.3 更稳的替代：Stage 2 采用 twin agreement / residual injection

如果目标是复现 VS3D 论文里的稳定效果，Stage 2 更推荐采用 twin agreement，而不是强行做 FlowEdit。原因是 Stage 2/3 的 sparse token space 在 add/remove 后经常不匹配。

伪代码：

```python
@torch.no_grad()
def stage2_geometry_twin_agreement(
    stage2_sampler,
    geo_encoder,
    C_src,
    C_tgt,
    source_asset,
    c_src,
    c_tgt,
    cfg,
    lambda_residual=0.5,
    threshold=0.7,
    clip_tau=10.0,
):
    # 1. 在同一个 C_tgt scaffold 上跑 target branch
    z_tgt = stage2_sampler.sample(coords=C_tgt, cond=c_tgt, seed=cfg.seed)

    # 2. 同样的 noise / seed / schedule，只换成 source condition
    z_src_twin = stage2_sampler.sample(coords=C_tgt, cond=c_src, seed=cfg.seed)

    # 3. disagreement 越小，越应该保留 source
    d = torch.linalg.norm(z_tgt - z_src_twin, dim=-1)  # [B, N] or [N]
    p_keep = robust_quantile_keep_score(d, q_low=0.05, q_high=0.95)

    # 4. source asset 编码到 C_src，再映射到 C_tgt 的交集
    z_src_enc = geo_encoder.encode(source_asset, coords=C_src)
    z_src_on_tgt, intersection_mask = build_source_geo_on_target_scaffold(
        C_src=C_src,
        C_tgt=C_tgt,
        x_geo_src=z_src_enc,
        init_new_tokens="zero",
    )

    # 5. residual injection
    residual = z_src_on_tgt - z_tgt
    residual = clip_by_token_norm(residual, clip_tau)

    inject_mask = intersection_mask[None, :, None] & (p_keep[..., None] >= threshold)
    z_edit = torch.where(
        inject_mask,
        z_tgt + lambda_residual * p_keep[..., None] * residual,
        z_tgt,
    )

    return z_edit, p_keep
```

这个版本更接近 VS3D 的 Stage 2 思路：让 sampler 自己比较 source condition 和 target condition 在同一 scaffold 上的输出差异，再决定哪些 token 该保留。

---

## 10. 主入口流程

```python
def run_trellis_flowedit_two_stage(
    trellis,
    source_asset,
    source_image,
    target_image,
    cfg: FlowEditConfig,
):
    # 1. 准备条件
    c_src, c_tgt, null_cond = prepare_conditions(
        trellis=trellis,
        source_image=source_image,
        target_image=target_image,
    )

    # 2. 编码 source asset 到 Stage 1 latent
    x_ss_src = trellis.stage1_encode_source_asset(source_asset)
    C_src = trellis.stage1_decode_to_coordinates(x_ss_src)

    # 3. Stage 1 FlowEdit：得到 edited occupancy
    z_ss_edit, C_tgt = flowedit_stage1_occupancy(
        stage1_model=trellis.stage1_model,
        stage1_decoder=trellis.stage1_decoder,
        x_ss_src=x_ss_src,
        c_src=c_src,
        c_tgt=c_tgt,
        null_cond=null_cond,
        cfg=cfg,
    )

    # 4. 编码 source geometry SLAT
    x_geo_src = trellis.geo_encoder.encode(source_asset, coords=C_src)

    # 5. Stage 2 Geometry
    if same_coordinates(C_src, C_tgt):
        z_geo_edit = flowedit_stage2_geometry_same_support(
            stage2_model=trellis.stage2_model,
            coords=C_tgt,
            x_geo_src=x_geo_src,
            c_src=c_src,
            c_tgt=c_tgt,
            null_cond=null_cond,
            cfg=cfg,
        )
        p_keep = None
    else:
        # 更推荐 twin agreement；也可以换成 flowedit_stage2_geometry_target_scaffold
        z_geo_edit, p_keep = stage2_geometry_twin_agreement(
            stage2_sampler=trellis.stage2_sampler,
            geo_encoder=trellis.geo_encoder,
            C_src=C_src,
            C_tgt=C_tgt,
            source_asset=source_asset,
            c_src=c_src,
            c_tgt=c_tgt,
            cfg=cfg,
        )

    # 6. Decode geometry mesh
    mesh = trellis.decode_geometry(coords=C_tgt, z_geo=z_geo_edit)

    return {
        "z_ss_edit": z_ss_edit,
        "C_src": C_src,
        "C_tgt": C_tgt,
        "z_geo_edit": z_geo_edit,
        "p_keep": p_keep,
        "mesh": mesh,
    }
```

---

## 11. 可选增强：把 VS3D 的 RASI / PMG 挂到 Stage 1

如果只实现原始 FlowEdit，很容易出现非编辑区域漂移。可以预留两个 hook：

```python
class Stage1FlowEditHooks:
    def get_null_cond_for_step(self, t, default_null_cond):
        return default_null_cond

    def postprocess_v_deltas(self, v_deltas):
        return torch.stack(v_deltas, dim=0).mean(dim=0)
```

### 11.1 RASI hook 思路

- 每个 active timestep 优化一个 `phi_t`；
- 优化目标是：把 target branch 的条件临时换成 `c_src`，看一步 Euler 后能否回到 `x_src`；
- 之后真实编辑时，把 `null_cond` 替换成缓存的 `phi_t`。

伪接口：

```python
class RASIHook(Stage1FlowEditHooks):
    def __init__(self):
        self.phi_cache = {}

    def calibrate_phi_for_step(self, model, x_src, z_edit, t, c_src, null_cond, cfg):
        phi_t = null_cond.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([phi_t], lr=1e-5)

        for _ in range(3):
            eps = torch.randn_like(x_src)
            z_src_t, z_tgt_t = make_flowedit_pair(x_src, z_edit, t, eps)

            v_src = guided_velocity(model, z_src_t, t, c_src, phi_t, cfg.omega_src)
            v_tgt_as_src = guided_velocity(model, z_tgt_t, t, c_src, phi_t, cfg.omega_tgt)
            v_delta = v_tgt_as_src - v_src

            z_recon = z_edit + get_dt(t) * v_delta
            loss = torch.mean((z_recon - x_src) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        self.phi_cache[float(t)] = phi_t.detach()

    def get_null_cond_for_step(self, t, default_null_cond):
        return self.phi_cache.get(float(t), default_null_cond)
```

### 11.2 PMG hook 思路

- 每个 timestep 采样 `S` 个 `v_delta`；
- 全样本均值：`mu_S = mean(v_deltas[:S])`；
- 部分样本均值：`mu_L = mean(v_deltas[:L])`；
- 更新方向：`mu_S + w * (mu_S - mu_L)`。

```python
class PMGHook(Stage1FlowEditHooks):
    def __init__(self, w=1.2, L=2):
        self.w = w
        self.L = L

    def postprocess_v_deltas(self, v_deltas):
        stack = torch.stack(v_deltas, dim=0)
        mu_S = stack.mean(dim=0)
        mu_L = stack[:self.L].mean(dim=0)
        return mu_S + self.w * (mu_S - mu_L)
```

---

## 12. 推荐调试顺序

1. **先跑 Stage 1 原始 FlowEdit**  
   验证 `z_ss_edit -> C_tgt` 是否能产生大致的编辑区域变化。

2. **固定 `c_tgt = c_src` 做 identity test**  
   如果理论上没有编辑，但 occupancy 明显漂移，说明 CFG / condition residual 在影响非编辑区域。

3. **加入 RASI / 或降低 `omega_tgt`**  
   先解决 identity drift，再谈编辑强度。

4. **加入 PMG**  
   如果编辑区域弱、加不出来、替换不明显，再放大稳定的 `v_delta`。

5. **Stage 2 先用 same support case 测试**  
   找一个 replacement 类型编辑，尽量让 `C_src` 和 `C_tgt` 接近，验证 sparse FlowEdit 是否跑通。

6. **add/remove 再切 target scaffold / twin agreement**  
   不要一开始就在 support mismatch 上硬套 FlowEdit，容易定位不到 bug。

---

## 13. 常见坑

### 13.1 时间方向反了

如果结果完全发散，第一件事检查：

```python
z_edit = z_edit + dt * v_delta
```

中的 `dt` 是否和 sampler 的 velocity 定义一致。不同 repo 可能把时间、sigma、scheduler index 封装过。

### 13.2 CFG scale 太强

`omega_tgt` 太大时，编辑会明显，但 source identity 容易被拖走。建议先从小 scale 跑通：

```python
omega_src = 1.0 ~ 2.0
omega_tgt = 4.0 ~ 9.0
```

### 13.3 Stage 2 sparse coordinates 没对齐

`sparse token index` 不等于空间位置。对齐必须基于 `coord tuple`，不能直接按 index 对应。

### 13.4 新增 token 的初始化影响很大

`C_tgt \ C_src` 的新增 geometry token 如果用纯 noise 初始化，可能更自由但更不稳定；如果用 zero 初始化，可能更稳但细节弱。可以比较：

```python
init_new_tokens = "noise" | "zero" | "target_prior"
```

### 13.5 误把 Stage 2 FlowEdit 当成无条件可用

Stage 2 的严格 FlowEdit 只适合 shared token space。对于 add/remove，推荐把 Stage 2 写成 FlowEdit-like 或 TAR-like。

---

## 14. 最小实现 checklist

- [ ] 能从 source asset 得到 `x_ss_src` 和 `C_src`
- [ ] 能从 source / target image 得到 `c_src`、`c_tgt`
- [ ] 能调用 Stage 1 DiT：`v = stage1_model(x_t, t, cond)`
- [ ] 实现 `guided_velocity`
- [ ] 实现 `make_flowedit_pair`
- [ ] 实现 Stage 1 Monte-Carlo `v_delta` 平均和 Euler update
- [ ] 能把 `z_ss_edit` decode 成 `C_tgt`
- [ ] 能从 source asset 得到 `x_geo_src`
- [ ] 实现 Stage 2 same-support FlowEdit
- [ ] 实现 coordinate-based SLAT alignment
- [ ] 对 support mismatch 场景实现 target-scaffold 或 twin-agreement fallback
- [ ] 记录每步 norm、active voxel 数量、最终 mesh 可视化

---

## 15. 一句话总结

Stage 1 可以比较自然地套用 FlowEdit，因为 occupancy latent 是 dense token space；Stage 2 只有在 source/target sparse coordinates 一致时才适合严格 FlowEdit。对于 add/remove 造成的 coordinate mismatch，更稳的工程实现是：先用 Stage 1 得到 `C_tgt`，再在 `C_tgt` 上用 target scaffold + source projection，或者直接采用 twin agreement / residual injection 的方式保护非编辑区域。
