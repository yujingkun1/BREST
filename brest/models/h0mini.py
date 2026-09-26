"""Trainable H0-mini image encoder used by end-to-end Visium training."""

from __future__ import annotations

import torch
from torch import nn

H0MINI_MEAN = (0.707223, 0.578729, 0.703617)
H0MINI_STD = (0.211883, 0.230117, 0.177517)


class H0MiniEncoder(nn.Module):
    """Return H0-mini CLS and patch tokens from a local checkpoint."""

    output_dim = 768

    def __init__(self, checkpoint_path: str, *, trainable: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise ImportError(
                "H0-mini fine-tuning requires timm; install BREST with "
                "`pip install -e '.[preprocessing]'`."
            ) from exc
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
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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


class H0FineTunedExpressionModel(nn.Module):
    """Compose trainable H0-mini with the unchanged BREST expression model."""

    def __init__(
        self,
        expression_model: nn.Module,
        checkpoint_path: str,
        *,
        encoder_batch: int = 192,
    ) -> None:
        super().__init__()
        self.image_encoder = H0MiniEncoder(checkpoint_path, trainable=True)
        self.expression_model = expression_model
        self.encoder_batch = int(encoder_batch)
        if self.encoder_batch <= 0:
            raise ValueError("encoder_batch must be positive")
        self.register_buffer("image_mean", torch.tensor(H0MINI_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(H0MINI_STD).view(1, 3, 1, 1))
        setter = getattr(self.image_encoder.model, "set_grad_checkpointing", None)
        if setter is not None:
            setter(True)

    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode uint8 NHWC/NCHW crops as CLS+mean-patch 1,536-d features."""
        if crops.ndim != 4:
            raise ValueError(f"expected 4-D crops, got shape {tuple(crops.shape)}")
        if crops.shape[-1] == 3:
            crops = crops.permute(0, 3, 1, 2)
        elif crops.shape[1] != 3:
            raise ValueError(f"expected RGB crops, got shape {tuple(crops.shape)}")
        outputs = []
        for start in range(0, crops.shape[0], self.encoder_batch):
            image = crops[start : start + self.encoder_batch].float().div_(255.0)
            image = (image - self.image_mean) / self.image_std
            with torch.autocast(
                device_type=image.device.type,
                dtype=torch.bfloat16,
                enabled=image.device.type == "cuda",
            ):
                tokens = self.image_encoder(image)
            outputs.append(
                torch.cat([tokens["cls"], tokens["patch"].mean(dim=1)], dim=1).float()
            )
        return torch.cat(outputs, dim=0)

    def forward(self, data) -> dict[str, torch.Tensor]:
        features = self.encode_crops(data.x)
        return self.expression_model.forward_features(
            features,
            data.edge_index,
            getattr(data, "pos", None),
            getattr(data, "batch", None),
        )
