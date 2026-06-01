"""Compatibility shims for the TRELLIS.2 codebase against newer dependencies.

Kept in our repo (not the external TRELLIS.2 install) so the pipeline runs
without patching a third-party checkout.
"""
from __future__ import annotations


def patch_dinov3_extractor() -> None:
    """Make ``DinoV3FeatureExtractor`` work across transformers versions.

    transformers 5.x nests the DINOv3 ViT encoder, moving the transformer
    blocks from ``model.layer`` to ``model.model.layer``.  TRELLIS.2's shipped
    extractor hardcodes ``self.model.layer``, so ``pipeline.get_cond`` raises
    ``'DINOv3ViTModel' object has no attribute 'layer'`` on transformers 5.x.
    This patches ``extract_features`` to locate the blocks either way.
    Idempotent; safe to call before every pipeline load.
    """
    try:
        from trellis2.modules import image_feature_extractor as ife
    except Exception:
        return
    cls = getattr(ife, "DinoV3FeatureExtractor", None)
    if cls is None or getattr(cls, "_pcv2_patched", False):
        return

    import torch.nn.functional as F

    def extract_features(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:                       # transformers 5.x nesting
            layers = self.model.model.layer
        for layer_module in layers:
            hidden_states = layer_module(
                hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    cls.extract_features = extract_features
    cls._pcv2_patched = True


__all__ = ["patch_dinov3_extractor"]
