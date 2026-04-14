"""Transfer learning models for aerial classification.

Supported backbones: ResNet50, MobileNetV2, EfficientNet-B0.
All use ImageNet-pretrained weights with a replaced classification head.
"""

import torch.nn as nn
from torchvision import models


def create_transfer_model(
    backbone: str = "resnet50",
    num_classes: int = 2,
    freeze_ratio: float = 0.8,
) -> nn.Module:
    """Create a transfer-learning model with partially frozen backbone.

    Parameters
    ----------
    backbone : str
        One of ``"resnet50"``, ``"mobilenet_v2"``, ``"efficientnet_b0"``.
    num_classes : int
        Number of output classes.
    freeze_ratio : float
        Fraction of backbone parameters to freeze (0.0 = none, 1.0 = all).
    """
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    else:
        raise ValueError(f"Unsupported backbone: {backbone}. "
                         "Choose from resnet50, mobilenet_v2, efficientnet_b0.")

    # Freeze early layers
    params = list(model.parameters())
    freeze_count = int(len(params) * freeze_ratio)
    for param in params[:freeze_count]:
        param.requires_grad = False

    return model
