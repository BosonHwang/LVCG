"""
Model registry for config-based instantiation.

Supports dynamic model loading via config:
    model:
      type: lvcg
      ...
"""

from typing import Dict, Type, Any
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg: Any) -> nn.Module:
    """
    Build model from config using registry.
    
    Args:
        cfg: Config object with cfg.model.type specifying model name
    
    Returns:
        Instantiated model
    
    Example:
        cfg.model.type = "lvcg"
        model = build_model(cfg)
    """
    model_type = cfg.model.get("type", "lvcg")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_cls = MODEL_REGISTRY[model_type]
    return model_cls.from_config(cfg)

