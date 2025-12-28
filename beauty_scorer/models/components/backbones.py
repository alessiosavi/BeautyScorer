"""
Backbone feature extractors registry and factory.

Provides a unified interface for loading various backbone networks
including MobileNet, EfficientNet, ViT, and DINOv2.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackboneInfo:
    """Information about a backbone network."""

    name: str
    feature_dim: int
    input_size: int
    family: str
    description: str


class BackboneRegistry:
    """Registry for backbone networks."""

    _backbones: dict[str, Callable] = {}
    _info: dict[str, BackboneInfo] = {}

    @classmethod
    def register(
        cls,
        name: str,
        feature_dim: int,
        input_size: int = 224,
        family: str = "unknown",
        description: str = "",
    ):
        """
        Decorator to register a backbone.

        Args:
            name: Backbone name.
            feature_dim: Output feature dimension.
            input_size: Expected input size.
            family: Model family (e.g., "mobilenet", "efficientnet").
            description: Description of the backbone.
        """

        def decorator(func: Callable):
            cls._backbones[name] = func
            cls._info[name] = BackboneInfo(
                name=name,
                feature_dim=feature_dim,
                input_size=input_size,
                family=family,
                description=description,
            )
            return func

        return decorator

    @classmethod
    def get(
        cls,
        name: str,
        pretrained: bool = True,
        **kwargs,
    ) -> tuple[nn.Module, int]:
        """
        Get a backbone by name.

        Args:
            name: Backbone name.
            pretrained: Whether to load pretrained weights.
            **kwargs: Additional arguments for backbone creation.

        Returns:
            Tuple of (backbone_module, feature_dim).
        """
        if name not in cls._backbones:
            available = list(cls._backbones.keys())
            raise ValueError(f"Unknown backbone: {name}. Available: {available}")

        backbone = cls._backbones[name](pretrained=pretrained, **kwargs)
        feature_dim = cls._info[name].feature_dim

        return backbone, feature_dim

    @classmethod
    def list_available(cls) -> list[str]:
        """List available backbone names."""
        return list(cls._backbones.keys())

    @classmethod
    def get_info(cls, name: str) -> BackboneInfo:
        """Get information about a backbone."""
        if name not in cls._info:
            raise ValueError(f"Unknown backbone: {name}")
        return cls._info[name]


# Register MobileNet backbones
@BackboneRegistry.register(
    "mobilenet_v3_large",
    feature_dim=960,
    input_size=224,
    family="mobilenet",
    description="MobileNetV3-Large - good balance of accuracy and speed",
)
def _mobilenet_v3_large(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    return model.features


@BackboneRegistry.register(
    "mobilenet_v3_small",
    feature_dim=576,
    input_size=224,
    family="mobilenet",
    description="MobileNetV3-Small - faster but less accurate",
)
def _mobilenet_v3_small(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    return model.features


@BackboneRegistry.register(
    "mobilenet_v2",
    feature_dim=1280,
    input_size=224,
    family="mobilenet",
    description="MobileNetV2 - CPU-friendly without SE blocks",
)
def _mobilenet_v2(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    return model.features


# Register EfficientNet backbones
@BackboneRegistry.register(
    "efficientnet_b0",
    feature_dim=1280,
    input_size=224,
    family="efficientnet",
    description="EfficientNet-B0 - efficient and accurate",
)
def _efficientnet_b0(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    return model.features


@BackboneRegistry.register(
    "efficientnet_lite0",
    feature_dim=320,  # EfficientNet-Lite0 outputs 320 features (not 1280 like regular B0)
    input_size=224,
    family="efficientnet",
    description="EfficientNet-Lite0 - optimized for CPU/mobile",
)
def _efficientnet_lite0(pretrained: bool = True, **kwargs) -> nn.Module:
    try:
        import timm

        model = timm.create_model(
            "efficientnet_lite0",
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )
        return model
    except ImportError:
        logger.warning("timm not installed, falling back to efficientnet_b0")
        return _efficientnet_b0(pretrained=pretrained)


# Register ResNet backbones
@BackboneRegistry.register(
    "resnet50",
    feature_dim=2048,
    input_size=224,
    family="resnet",
    description="ResNet-50 - classic architecture",
)
def _resnet50(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    # Remove avgpool and fc
    return nn.Sequential(*list(model.children())[:-2])


# Register ViT backbones
@BackboneRegistry.register(
    "vit_b_16",
    feature_dim=768,
    input_size=224,
    family="vit",
    description="Vision Transformer Base with 16x16 patches",
)
def _vit_b_16(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    # Return encoder only (without classification head)
    return _ViTFeatureExtractor(model)


@BackboneRegistry.register(
    "vit_b_32",
    feature_dim=768,
    input_size=224,
    family="vit",
    description="Vision Transformer Base with 32x32 patches (faster)",
)
def _vit_b_32(pretrained: bool = True, **kwargs) -> nn.Module:
    from torchvision.models import ViT_B_32_Weights, vit_b_32

    weights = ViT_B_32_Weights.DEFAULT if pretrained else None
    model = vit_b_32(weights=weights)
    return _ViTFeatureExtractor(model)


# Register DINOv2 backbones
@BackboneRegistry.register(
    "dinov2_vits14",
    feature_dim=384,
    input_size=224,
    family="dinov2",
    description="DINOv2 ViT-S/14 - excellent self-supervised features",
)
def _dinov2_vits14(pretrained: bool = True, **kwargs) -> nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        return _DINOv2FeatureExtractor(model)
    except Exception as e:
        logger.warning(f"Failed to load DINOv2: {e}. Falling back to ViT.")
        return _vit_b_16(pretrained=pretrained)


@BackboneRegistry.register(
    "dinov2_vitb14",
    feature_dim=768,
    input_size=224,
    family="dinov2",
    description="DINOv2 ViT-B/14 - larger DINOv2 model",
)
def _dinov2_vitb14(pretrained: bool = True, **kwargs) -> nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained)
        return _DINOv2FeatureExtractor(model)
    except Exception as e:
        logger.warning(f"Failed to load DINOv2: {e}. Falling back to ViT.")
        return _vit_b_16(pretrained=pretrained)


class _ViTFeatureExtractor(nn.Module):
    """Wrapper to extract features from ViT models."""

    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model
        # Remove the classification head
        self.model.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _DINOv2FeatureExtractor(nn.Module):
    """Wrapper to extract features from DINOv2 models."""

    def __init__(self, dino_model):
        super().__init__()
        self.model = dino_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 returns CLS token features
        return self.model(x)


class CNNFeatureExtractor(nn.Module):
    """
    Generic CNN feature extractor wrapper.

    Handles pooling for CNN backbones that output spatial features.
    """

    def __init__(self, backbone: nn.Module, pool: bool = True):
        """
        Initialize wrapper.

        Args:
            backbone: CNN backbone module.
            pool: Whether to apply global average pooling.
        """
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.

        Args:
            x: Input images (batch, C, H, W).

        Returns:
            Features (batch, feature_dim).
        """
        features = self.backbone(x)

        # Handle timm models that return list
        if isinstance(features, list):
            features = features[-1]

        if self.pool is not None and features.dim() == 4:
            features = self.pool(features).flatten(1)

        return features


def get_backbone(
    name: str,
    pretrained: bool = True,
    pool: bool = True,
    **kwargs,
) -> tuple[nn.Module, int]:
    """
    Get a backbone feature extractor.

    Args:
        name: Backbone name.
        pretrained: Whether to load pretrained weights.
        pool: Whether to apply global pooling for CNN backbones.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (backbone_module, feature_dim).
    """
    backbone, feature_dim = BackboneRegistry.get(name, pretrained=pretrained, **kwargs)

    info = BackboneRegistry.get_info(name)

    # Wrap CNN backbones with pooling
    if info.family in ["mobilenet", "efficientnet", "resnet"]:
        backbone = CNNFeatureExtractor(backbone, pool=pool)

    return backbone, feature_dim


def list_available_backbones() -> list[dict]:
    """
    List all available backbones with their info.

    Returns:
        List of backbone info dictionaries.
    """
    result = []
    for name in BackboneRegistry.list_available():
        info = BackboneRegistry.get_info(name)
        result.append(
            {
                "name": info.name,
                "feature_dim": info.feature_dim,
                "input_size": info.input_size,
                "family": info.family,
                "description": info.description,
            }
        )
    return result


def freeze_backbone(
    backbone: nn.Module,
    num_unfrozen_layers: int = 0,
) -> None:
    """
    Freeze backbone parameters.

    Args:
        backbone: Backbone module.
        num_unfrozen_layers: Number of layers to keep unfrozen from the end.
    """
    # Freeze all parameters first
    for param in backbone.parameters():
        param.requires_grad = False

    if num_unfrozen_layers > 0:
        # Get all modules that have parameters
        modules_with_params = []
        for module in backbone.modules():
            if list(module.parameters(recurse=False)):
                modules_with_params.append(module)

        # Unfreeze last N modules
        for module in modules_with_params[-num_unfrozen_layers:]:
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_backbone(backbone: nn.Module) -> None:
    """
    Unfreeze all backbone parameters.

    Args:
        backbone: Backbone module.
    """
    for param in backbone.parameters():
        param.requires_grad = True
