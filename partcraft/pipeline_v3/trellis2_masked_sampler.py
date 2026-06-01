"""P2 — masked-sampling sampler for TRELLIS.2 flow models.

This module subclasses TRELLIS.2's `FlowEulerSampler` family **without
modifying the upstream codebase**. The subclasses add two knobs needed
by Vinedresser3D-style inpainting:

1. ``x_init``  — start the trajectory from a user-supplied tensor
   (e.g. an inverted clean SLat) instead of pure Gaussian noise.

2. ``step_callback(sample, t, t_prev, idx, total) -> sample`` — fires
   AFTER each Euler step's ``pred_x_prev`` has been computed.  The
   callback may return a modified ``sample`` (e.g. with R_pres tokens
   overwritten by the noised clean reference).  Returning ``None`` is
   equivalent to leaving ``sample`` unchanged.

The callback is the hook P3 will use to implement masked replacement:
    sample.feats[R_pres] = noised_clean.feats[R_pres]   at step t_prev

Helper :func:`forward_diffuse_flow` produces the correctly-noised
clean reference for a given t under flow matching:

    x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * eps

so the callback can be written in one line.

Usage (drop-in replacement for `FlowEulerGuidanceIntervalSampler`):

    sampler = MaskedFlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    out = sampler.sample(
        model, noise, cond=cond, neg_cond=neg_cond,
        steps=50, guidance_strength=3.0, guidance_interval=(0., 1.),
        x_init=inverted_x0,                  # optional
        step_callback=my_mask_callback,      # optional
    )
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

# Resolve TRELLIS.2 codebase on sys.path (machine-env can override).
_T2 = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if _T2 not in sys.path:
    sys.path.insert(0, _T2)

from trellis2.pipelines.samplers.flow_euler import (   # noqa: E402
    FlowEulerSampler,
)
from trellis2.pipelines.samplers.classifier_free_guidance_mixin import (  # noqa: E402
    ClassifierFreeGuidanceSamplerMixin,
)
from trellis2.pipelines.samplers.guidance_interval_mixin import (  # noqa: E402
    GuidanceIntervalSamplerMixin,
)


# ── helper: noised version of a clean x_0 at step t (flow matching) ──

def forward_diffuse_flow(
    x_0: Any,
    t: float,
    sigma_min: float,
    eps: Optional[Any] = None,
) -> Any:
    """Forward-diffuse a clean sample to step ``t`` under flow matching.

    Mirrors the analytic forward process used by ``FlowEulerSampler``:

        x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * eps

    Works for both dense tensors and ``SparseTensor`` (whose arithmetic
    overloads dispatch to ``feats``).

    Args:
        x_0:   clean reference sample (Tensor or SparseTensor)
        t:     target timestep in [0, 1]; 0 = clean, 1 = pure noise
        sigma_min: sampler ``sigma_min`` (must match the sampler used)
        eps:   user-supplied noise.  If None, draws standard normal of
               the same shape/structure as ``x_0``.

    Returns:
        x_t of the same type/shape as ``x_0``.
    """
    if eps is None:
        if hasattr(x_0, "feats"):                  # SparseTensor
            eps_feats = torch.randn_like(x_0.feats)
            eps = x_0.replace(eps_feats)
        else:
            eps = torch.randn_like(x_0)
    sig = sigma_min + (1 - sigma_min) * t
    return (1 - t) * x_0 + sig * eps


# ── core: callback-hook sampler ──────────────────────────────────────

class MaskedFlowEulerSampler(FlowEulerSampler):
    """``FlowEulerSampler`` with ``x_init`` and per-step ``step_callback``.

    Also exposes RF-Solver (2nd-order Heun-style RK) inversion utilities
    used by Vinedresser3D-style edit pipelines: see :meth:`rf_step` and
    :meth:`invert_clean`.
    """

    # ─────────────── RF-Solver (2nd-order RK) ────────────────────────
    # Ported from Vinedresser3D/interweave_Trellis.py::RF_sample_once.
    # The SAME formula works for forward (t_curr > t_prev) and inverse
    # (t_curr < t_prev); the caller just supplies appropriately ordered
    # (t_curr, t_prev) pairs.  CFG / interval / mixin behaviour is honoured
    # automatically because we route through ``self._inference_model``.

    @torch.no_grad()
    def rf_step(self, model, x_t, t_curr: float, t_prev: float, **kwargs):
        """One 2nd-order RF-Solver step.

        ``x(t_prev) = x(t_curr) + dt * v(t_curr) + (dt^2 / 2) * dv/dt``
        with ``dv/dt`` approximated by a midpoint sample. The result is
        deterministic given the model + cond + t pair.
        """
        pred_v0 = self._inference_model(model, x_t, t_curr, **kwargs)
        sample_mid = x_t + (t_prev - t_curr) / 2 * pred_v0
        t_mid = (t_curr + t_prev) / 2
        pred_v_mid = self._inference_model(model, sample_mid, t_mid, **kwargs)
        first_order = (pred_v_mid - pred_v0) / ((t_prev - t_curr) / 2)
        dt = t_prev - t_curr
        return x_t + dt * pred_v0 + 0.5 * dt * dt * first_order

    @torch.no_grad()
    def invert_clean(
        self,
        model,
        clean_x0,
        cond: Optional[Any] = None,
        steps: int = 12,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "RF inversion",
        **kwargs,
    ) -> dict:
        """Run RF-Solver in inverse direction (t: 0 → 1).

        Returns a dict ``{t_value: x_t}`` keyed by rounded float t (6 decimals).
        Includes the input ``clean_x0`` under key 0.0 and the final
        inverted state under key 1.0 — pass that as ``x_init`` to forward
        sampling and consult intermediate entries from a step callback
        (see :func:`make_inverse_anchored_callback`).

        ``kwargs`` are forwarded to ``self._inference_model`` (e.g.
        ``neg_cond``, ``guidance_strength``, ``guidance_interval``).
        Vinedresser3D sets ``guidance_strength=0`` during inversion.
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq[::-1].tolist()              # 0 → 1
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        traj: dict = {round(float(t_seq[0]), 6): clean_x0}
        sample = clean_x0
        for t_curr, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            sample = self.rf_step(model, sample, float(t_curr), float(t_prev),
                                  cond=cond, **kwargs)
            traj[round(float(t_prev), 6)] = sample
        return traj

    # ─────────────── forward sample (with callback / x_init) ─────────

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        *,
        x_init: Optional[Any] = None,
        step_callback: Optional[
            Callable[[Any, float, float, int, int], Optional[Any]]
        ] = None,
        **kwargs,
    ):
        """Generate samples; same return shape as the parent class.

        New kwargs:
            x_init:        start tensor (overrides ``noise`` for the
                           trajectory's initial value).  Useful when
                           bootstrapping from an inverted clean x_0.
            step_callback: ``fn(sample, t, t_prev, idx, total) -> sample``.
                           Called AFTER each Euler step.  Return value
                           replaces ``sample``; return ``None`` to keep.
        """
        sample = x_init if x_init is not None else noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        total = len(t_pairs)
        for idx, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)
        ):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            if step_callback is not None:
                new_sample = step_callback(sample, t, t_prev, idx, total)
                if new_sample is not None:
                    sample = new_sample
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class MaskedFlowEulerCfgSampler(
    ClassifierFreeGuidanceSamplerMixin, MaskedFlowEulerSampler
):
    """CFG variant; same kwargs as ``FlowEulerCfgSampler`` + masked-sampler extras."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        *,
        x_init: Optional[Any] = None,
        step_callback: Optional[Callable] = None,
        **kwargs,
    ):
        return super().sample(
            model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            x_init=x_init, step_callback=step_callback,
            **kwargs,
        )


class MaskedFlowEulerGuidanceIntervalSampler(
    GuidanceIntervalSamplerMixin,
    ClassifierFreeGuidanceSamplerMixin,
    MaskedFlowEulerSampler,
):
    """CFG + interval variant — drop-in for ``FlowEulerGuidanceIntervalSampler``."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        *,
        x_init: Optional[Any] = None,
        step_callback: Optional[Callable] = None,
        **kwargs,
    ):
        return super().sample(
            model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            x_init=x_init, step_callback=step_callback,
            **kwargs,
        )


# ── convenience: a callback factory for "preserve a fixed mask" ──────

def make_mask_preserve_callback(
    clean_x0: Any,
    keep_mask: torch.Tensor,
    sigma_min: float,
    *,
    schedule: str = "all",
    cutoff_t: float = 0.0,
    seed: Optional[int] = None,
):
    """Return a ``step_callback`` that re-injects clean tokens at every step.

    At step ``t_prev``, the callback replaces ``sample`` values where
    ``keep_mask`` is True with the noised version of ``clean_x0`` at the
    same timestep (analytic flow-matching forward process).  Anywhere
    ``keep_mask`` is False, the sample is left alone (= "edit region").

    Args:
        clean_x0: clean reference (SparseTensor or dense Tensor).  Must
                  match the structure/shape of ``sample``.
        keep_mask: bool tensor selecting which positions to preserve.
                   * SparseTensor case: shape ``[N_tokens]`` aligned with
                     ``clean_x0.feats``.
                   * Dense case: broadcastable to ``clean_x0`` (e.g.
                     ``[B, 1, D, H, W]``).
        sigma_min: must match the sampler.
        schedule: ``"all"`` = inject at every step; ``"early"`` = only
                  when ``t_prev >= cutoff_t`` (RePaint-style, lets the
                  edit region settle near the end).
        cutoff_t: threshold for the ``early`` schedule.
        seed:    optional torch seed for the per-step noise (reproducible).

    The callback re-samples noise at each step (matching RePaint's
    independent-noise rule), so the preserved region remains on the
    correct marginal for ``t_prev``.
    """
    gen = torch.Generator(device=keep_mask.device) if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)

    is_sparse = hasattr(clean_x0, "feats")

    def _cb(sample, t, t_prev, idx, total):
        if schedule == "early" and t_prev < cutoff_t:
            return None
        # Build noise matched to the sample's structure.
        if is_sparse:
            eps_feats = torch.empty_like(clean_x0.feats)
            if gen is not None:
                eps_feats.normal_(generator=gen)
            else:
                eps_feats.normal_()
            eps = clean_x0.replace(eps_feats)
        else:
            eps = torch.empty_like(clean_x0)
            if gen is not None:
                eps.normal_(generator=gen)
            else:
                eps.normal_()
        noised_clean = forward_diffuse_flow(clean_x0, t_prev, sigma_min, eps=eps)

        if is_sparse:
            # In-place overwrite preserved tokens.  sample.feats is a torch
            # tensor of shape [N_tokens, C]; keep_mask is [N_tokens].
            new_feats = sample.feats.clone()
            new_feats[keep_mask] = noised_clean.feats[keep_mask]
            return sample.replace(new_feats)
        else:
            return torch.where(keep_mask, noised_clean, sample)

    return _cb


def make_inverse_anchored_callback(
    inverse_traj: dict,
    keep_mask: torch.Tensor,
    *,
    schedule: str = "all",
    cutoff_t: float = 0.0,
    tol: float = 1e-4,
):
    """Replace ``sample[keep_mask]`` at each step with the inversion trajectory
    sampled at ``t_prev``.

    Unlike :func:`make_mask_preserve_callback` (which re-noises ``clean_x0``
    with fresh Gaussian at every step), this callback uses the deterministic
    state produced by RF-Solver inversion — closer to in-distribution for
    the model because the velocity field itself put the clean sample there.

    Args:
        inverse_traj: dict ``{t: x_t}`` from :meth:`MaskedFlowEulerSampler.invert_clean`.
        keep_mask:    bool tensor selecting positions to preserve.  Same
                      conventions as :func:`make_mask_preserve_callback`.
        schedule:     ``"all"`` (default) or ``"early"`` — ``"early"`` only
                      replaces while ``t_prev >= cutoff_t`` (skips the last
                      few steps so the edit region can settle).
        cutoff_t:     threshold used by the ``"early"`` schedule.
        tol:          float tolerance for matching a forward ``t_prev`` to
                      a key in ``inverse_traj`` (since rounding may diverge
                      across the two passes).
    """
    is_sparse_ref = next(iter(inverse_traj.values()))
    is_sparse = hasattr(is_sparse_ref, "feats")
    keys_sorted = sorted(inverse_traj.keys())

    def _lookup(t_prev: float):
        # First try exact (rounded) hit, then nearest within tol.
        key = round(float(t_prev), 6)
        if key in inverse_traj:
            return inverse_traj[key]
        # Nearest neighbour.
        best_k = min(keys_sorted, key=lambda k: abs(k - t_prev))
        return inverse_traj[best_k] if abs(best_k - t_prev) < tol else None

    def _cb(sample, t, t_prev, idx, total):
        if schedule == "early" and t_prev < cutoff_t:
            return None
        ref = _lookup(t_prev)
        if ref is None:
            return None
        if is_sparse:
            new_feats = sample.feats.clone()
            new_feats[keep_mask] = ref.feats[keep_mask]
            return sample.replace(new_feats)
        else:
            if keep_mask.dtype == torch.bool:
                return torch.where(keep_mask, ref, sample)
            # soft keep weight in [0,1]: blend instead of hard-replace so the
            # body↔edit boundary transitions smoothly (no torn occupancy seam).
            return sample + keep_mask * (ref - sample)

    return _cb


def make_bridged_anchor_callback(
    inverse_traj: dict,
    preserved_mask: torch.Tensor,
    src_index: torch.Tensor,
    *,
    schedule: str = "all",
    cutoff_t: float = 0.0,
    tol: float = 1e-4,
):
    """Coord-bridged variant of :func:`make_inverse_anchored_callback`.

    Used when the forward sample lives on a DIFFERENT sparse coord set than the
    inversion trajectory — i.e. the *structure* (active-voxel set) was edited,
    so ``coords_new != coords_orig``.  Token rows therefore no longer line up
    1:1 and we must gather by an explicit index map.

    Args:
        inverse_traj:   dict ``{t: x_t}`` from
                        :meth:`MaskedFlowEulerSampler.invert_clean`, on the
                        ORIGINAL coords.
        preserved_mask: bool ``[N_new]`` selecting which forward-sample tokens
                        (in coords_new order) to anchor to the original.
        src_index:      long ``[P]`` where ``P == preserved_mask.sum()``; row in
                        the trajectory tensors (coords_orig order) feeding each
                        preserved token, in coords_new order.

    At step ``t_prev``:
        ``sample.feats[preserved_mask] = inverse_traj[t_prev].feats[src_index]``
    """
    keys_sorted = sorted(inverse_traj.keys())

    def _lookup(t_prev: float):
        key = round(float(t_prev), 6)
        if key in inverse_traj:
            return inverse_traj[key]
        best_k = min(keys_sorted, key=lambda k: abs(k - t_prev))
        return inverse_traj[best_k] if abs(best_k - t_prev) < tol else None

    def _cb(sample, t, t_prev, idx, total):
        if schedule == "early" and t_prev < cutoff_t:
            return None
        ref = _lookup(t_prev)
        if ref is None:
            return None
        new_feats = sample.feats.clone()
        new_feats[preserved_mask] = ref.feats[src_index]
        return sample.replace(new_feats)

    return _cb


__all__ = [
    "MaskedFlowEulerSampler",
    "MaskedFlowEulerCfgSampler",
    "MaskedFlowEulerGuidanceIntervalSampler",
    "forward_diffuse_flow",
    "make_mask_preserve_callback",
    "make_inverse_anchored_callback",
    "make_bridged_anchor_callback",
]
