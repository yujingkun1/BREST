"""H0-mini image encoder wrapper (Bioptimus H0-mini, HuggingFace bioptimus/H0-mini).

timm `vit_base_patch14_reg4_dinov2` + 4 register tokens + SwiGLU MLP, 768-d tokens.
Weights are downloaded separately (see ``docs/preprocessing.md``); set
H0MINI_CHECKPOINT or pass --h0-checkpoint.
"""

from __future__ import annotations

import os

import torch
from torch import nn
import timm

# Download bioptimus/H0-mini and point here (env var overrides).
H0MINI_CHECKPOINT = os.environ.get("H0MINI_CHECKPOINT", "H0-mini/pytorch_model.bin")
# H0-mini ImageNet-style normalization (from the model's pretrained_cfg).
H0MINI_MEAN = (0.707223, 0.578729, 0.703617)
H0MINI_STD = (0.211883, 0.230117, 0.177517)


class H0MiniEncoder(nn.Module):
    """Return CLS and patch tokens from local H0-mini weights."""

    output_dim = 768

    def __init__(
        self,
        *,
        checkpoint_path: str = H0MINI_CHECKPOINT,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.trainable = bool(trainable)
        self.model = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=224,
            reg_tokens=4,
            dynamic_img_size=True,
            mlp_ratio=5.33334,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable

    @property
    def num_prefix_tokens(self) -> int:
        return int(self.model.num_prefix_tokens)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.trainable:
            tokens = self.model.forward_features(image)
        else:
            self.model.eval()
            with torch.no_grad():
                tokens = self.model.forward_features(image)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, self.num_prefix_tokens :],
            "tokens": tokens,
        }
