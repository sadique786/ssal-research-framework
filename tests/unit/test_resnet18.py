import pytest
import torch

from ssal.models import ResNet18Classifier


def test_resnet18_output_shape() -> None:
    model = ResNet18Classifier(
        num_classes=10,
        pretrained=False,
    )

    inputs = torch.randn(4, 3, 32, 32)

    outputs = model(inputs)

    assert list(outputs.shape) == [4, 10]


def test_resnet18_feature_shape() -> None:
    model = ResNet18Classifier(
        num_classes=10,
        pretrained=False,
    )

    inputs = torch.randn(4, 3, 32, 32)

    features = model.extract_features(inputs)

    assert features.shape[0] == 4
    assert features.shape[1] == 512


def test_resnet18_invalid_num_classes() -> None:
    with pytest.raises(ValueError):
        ResNet18Classifier(num_classes=0)
