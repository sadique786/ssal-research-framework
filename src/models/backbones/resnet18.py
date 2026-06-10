# src/models/backbones/resnet18.py

import torch
import torch.nn as nn

from torchvision.models import resnet18


class ResNet18Classifier(nn.Module):
    """
    ResNet18 classifier wrapper.
    """

    def __init__(self, num_classes: int):

        super().__init__()

        self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):

        return self.backbone(x)

    def extract_features(self, x):

        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        features = self.backbone.avgpool(features)

        return torch.flatten(features, 1)
