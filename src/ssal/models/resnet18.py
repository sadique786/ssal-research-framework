from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ssal.models.base import BaseModel


class ResNet18Classifier(BaseModel):
    """ResNet-18 classifier adapted for small images such as CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes)

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = resnet18(weights=weights)

        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification layer."""

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        return torch.flatten(x, 1)
