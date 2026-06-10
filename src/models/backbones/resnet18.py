import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Classifier(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.backbone = resnet18(weights=None)

        feature_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):

        modules = list(self.backbone.children())[:-1]

        feature_extractor = nn.Sequential(*modules)

        features = feature_extractor(x)

        return torch.flatten(features, 1)