"""Encoder registry for LVCG linear probing."""

from .base import BaseEncoder
from .lvcg_encoder import LVCGEncoder

ENCODER_REGISTRY = {
    "lvcg": LVCGEncoder,
}

_RESERVED_KEYS = {"type", "enabled", "tag", "source"}


def create_encoder(model_config: dict, device: str = "cpu") -> BaseEncoder:
    model_type = model_config["type"]
    if model_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {model_type}. Available: {list(ENCODER_REGISTRY)}"
        )
    encoder_cls = ENCODER_REGISTRY[model_type]
    kwargs = {k: v for k, v in model_config.items() if k not in _RESERVED_KEYS}
    encoder = encoder_cls(**kwargs)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


__all__ = ["BaseEncoder", "ENCODER_REGISTRY", "create_encoder", "LVCGEncoder"]
