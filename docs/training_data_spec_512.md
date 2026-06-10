# Pxform_v2 编辑训练数据规格(512 edit-res base)

每条训练样本 = **一次编辑(edit)**。三阶段 latent 各有 before/after,外加一份 mask。

- **before** = 原始物体 encode(每个 object 一份,**该 object 的所有编辑共享**)。
- **after** = 这一次编辑的结果(**每条编辑一份**)。
- **mask** = 这一次编辑的区域掩码(**每条编辑一份**)。

分辨率 base = **512 edit-res**。后果:S2 SLat 在 **32³**;S1/SS 永远 64³ 占据 → **16³** 稠密 latent
(SS latent 不随 512/1024 变,固定 16³)。**所有 SLat 必须用 32³ 表示**——before 取 `_e512`
sidecar(32³),**不是** 64³ 主体。

---

## 0. 分辨率对照(512 base)

| 阶段 | 表示 | 网格 | 形状 |
|------|------|------|------|
| SS(结构)latent | **稠密** | 16³ | `[1, 8, 16, 16, 16]` |
| shape SLat | **稀疏** | 32³ (0..31) | feats `[N, 32]` + coords `[N, 3]` |
| tex SLat | **稀疏** | 32³ (0..31) | feats `[N, 32]` + coords `[N, 3]`(与 shape **共享 coords**) |
| SS mask | 稠密 | 16³ | `[16, 16, 16]` |
| SLat mask | 逐稀疏体素 | — | `[N]`(对齐 shape/tex coords) |

`N` 在 before / after 不同(例:before 2675、after 3275);**同一状态内 shape 与 tex 的 N 和 coords 完全相同**。

---

## 1. SS 阶段 latent(稠密 16³)

SS latent = `ss_enc(占据@64³)`,一次 SS-VAE 前向(无 flow、无 flux)。

| | 文件 | key | 形状 | dtype | 来源 |
|--|------|-----|------|-------|------|
| **before** | `p1_encode/ss.npz` | `ss` | `[1,8,16,16,16]` | float32 | `ss_enc(原始mesh 64³占据)`(已存在) |
| **after** | `edits_3d/<eid>/latents/ss_latent.npz` | `ss` | `[1,8,16,16,16]` | float32 | `ss_enc(after 64³占据)` ← **已实现** |

> **after SS latent 实现**(`trellis2_3d.py`,2026-06-10 落地 + 冒烟验证):
> - **forward-save**:normal 路径在 `_to_s2` 前 `ss_enc(coords_new@64³)` → `_save_edit_latents`
>   写 `ss_latent.npz`。以后所有新跑的 mod/scale 自带,与 shape/tex 同次产出 → **保证一致**。
> - **mod/scale 回填**:`trellis2_ss_latent_only: true` → `_build_p4_mesh` 只跑 S1,
>   `ss_enc` 后早返回(跳过 S2/decode/render,~20s/edit),写 `ss_latent.npz`+`mask.npz`。
>   一致性 `downsample2(重跑 coords_new@64³) == 盘上 coords_new@32³`:冒烟 3/3 **IoU=1.0000**。
> - **del/add**:after 是真 mesh → 体素化 64³ → `ss_enc`(干净,与 before 同源)。← **待加到 del_add_reencode**。
>
> 注:`edits_3d/<eid>/latents/ss.npz`(不带 `_latent`)是**坐标/区域包,不是 latent**——见 §4,mask 由它派生。命名上 `ss_latent.npz` 才是稠密 SS latent。

---

## 2. shape SLat(稀疏 32³)

| | 文件 | keys | 形状 | dtype | 网格 |
|--|------|------|------|-------|------|
| **before** | `p1_encode/shape_slat_e512.npz` | `feats`,`coords` | `[N_b,32]`,`[N_b,3]` | float32,int32 | 32³(0..31)✅ |
| **after** | `edits_3d/<eid>/latents/shape_slat.npz` | `feats`,`coords` | `[N_a,32]`,`[N_a,3]` | float32,int16 | 32³(0..31) |

> ⚠️ before 用 **`shape_slat_e512.npz`**(32³),**不要**用 `shape_slat.npz`(那是 64³,coords 0..63,与 512 base 不对齐)。
> before 的 `coords` == 编辑包里的 `coords0`(同为 `N_b` 个 32³ 体素)。

---

## 3. tex SLat(稀疏 32³,与 shape 共享 coords)

| | 文件 | keys | 形状 | dtype | 网格 |
|--|------|------|------|-------|------|
| **before** | `p1_encode/tex_slat_e512.npz` | `feats`,`coords` | `[N_b,32]`,`[N_b,3]` | float32,int32 | 32³ ✅ |
| **after** | `edits_3d/<eid>/latents/tex_slat.npz` | `feats`,`coords` | `[N_a,32]`,`[N_a,3]` | float32,int16 | 32³ |

> tex 的 `coords` 与同状态 shape 的 `coords` **逐字节相同**(一次体素化共用)。tex feats 是 6 通道 PBR
> 属性(base_color3+metallic1+roughness1+alpha1)经 VAE 编码后的 32 维 latent。

---

## 4. mask(每条编辑一份;覆盖三阶段分辨率)

mask 是编辑区掩码,**KEEP 极性**:`1 = 保留原结构`,`0 = 编辑/重采样区`(编辑区 = 补集)。
全部是 `edits_3d/<eid>/latents/ss.npz`(坐标/区域包)的纯函数,`mask_from_ss(ss)` 派生。

| key | 形状 | dtype | 对齐 | 含义 |
|-----|------|-------|------|------|
| `mask_keep_ss` | `[16,16,16]` | uint8 | SS latent 16³ | SS 阶段保留掩码(= 区域包的 `keep16`) |
| `mask_keep_slat` | `[N_a]` | uint8 | **after** shape/tex coords | after 每个 32³ 稀疏体素:在 `edit_grid` 内→0(编辑),外→1(保留) |
| `mask_keep_slat_before` | `[N_b]` | uint8 | **before** shape/tex coords | before 每个 32³ 稀疏体素的同义掩码 |
| `selected_part_ids` | `[P]` | int32 | — | 这次编辑涉及的 part id |

> SS mask 是稠密 16³(配 §1 的稠密 SS latent);SLat mask 是逐稀疏体素 1D(配 §2/§3 的稀疏 coords,
> before/after 各一份,因 N 不同)。

落盘:`edits_3d/<eid>/latents/mask.npz`(开 `del_add_write_mask` / 在 `_save_edit_latents` 写;
mod/scale 已有数据用 CPU 回填)。

---

## 5. 编辑区域包(provenance,mask 的源)—— `edits_3d/<eid>/latents/ss.npz`

非训练直喂,是 mask 的来源 + 溯源。已存在,schema 不动:

| key | 形状 | 含义 |
|-----|------|------|
| `coords0` | `[N_b,3]` int16 0..31 | before 32³ 占据(== before SLat coords) |
| `coords_new` | `[N_a,3]` int16 0..31 | after 32³ 占据(== after SLat coords) |
| `edit_grid` | `[M,3]` int16 | 32³ 编辑区坐标 |
| `keep16` | `[16,16,16]` bool | 16³ 保留掩码(= `mask_keep_ss`) |
| `parts` | `[P]` int32 | selected_part_ids |
| `edit_type` | scalar str | modification/scale/deletion/addition |
| `s1_pad`,`s1_thresh` | scalar | 派生 keep16/edit_grid 的参数(可复现) |

---

## 6. 最终每条样本的物理布局(prod 直读,不导出)

```
<obj>/p1_encode/                              # before,per-object 共享
    ss.npz                  ss[1,8,16,16,16]              ← SS before latent
    shape_slat_e512.npz     feats[N_b,32] coords[N_b,3]   ← shape before @32³
    tex_slat_e512.npz       feats[N_b,32] coords[N_b,3]   ← tex   before @32³

<obj>/edits_3d/<eid>/latents/                 # after + mask,per-edit
    ss_latent.npz           ss[1,8,16,16,16]              ← SS after latent   ★新增
    shape_slat.npz          feats[N_a,32] coords[N_a,3]   ← shape after @32³
    tex_slat.npz            feats[N_a,32] coords[N_a,3]   ← tex   after @32³
    mask.npz                mask_keep_ss[16³] + mask_keep_slat[N_a]
                            + mask_keep_slat_before[N_b] + selected_part_ids[P]  ★新增
    ss.npz                  坐标/区域包(mask 的源,§5)
```

一条训练样本 = 上面 before 三件 + after 三件(含新增 `ss_latent.npz`)+ `mask.npz`。

---

## 7. 不变量(每条样本必须满足)

- shape 与 tex 的 `coords` **逐字节相同**(同状态);before 用 `_e512`,coords.max() ≤ 31。
- before `coords` == 区域包 `coords0`;after SLat `coords` == 区域包 `coords_new`。
- `mask_keep_slat` 长度 == after `N_a`;`mask_keep_slat_before` 长度 == before `N_b`;`mask_keep_ss` == `[16,16,16]`。
- SS latent(before & after)== `[1,8,16,16,16]` float32。
- del/add 的 after SS latent 与 before **同源**(真占据 `ss_enc`);mod/scale 的 after SS latent 标注来源((a) upsample / (b) S1-rerun)。

---

## 8. 待落地(本规格驱动的改动)

1. **新增 after SS latent `ss_latent.npz`**:
   - del/add → `del_add_reencode.py`:after mesh 体素化 64³ → `ss_enc` → 写 `ss_latent.npz`。
   - mod/scale → 先跑 §1 的 (a) 保真探针(`ss_enc(upsample(coords0))` vs `p1 ss.npz` 的 decode IoU);
     IoU 高走 (a) 廉价 encode pass,否则走 (b) S1-only。
2. **mask 落盘**:`del_add_write_mask=true` + `_save_edit_latents` 写 `mask.npz` + mod/scale CPU 回填。
3. 全程**不重跑 flux / 全 pipeline**。
