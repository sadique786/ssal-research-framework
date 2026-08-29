import pytest

from ssal.utils import get_device


def test_auto_device_returns_supported_device() -> None:
    device = get_device("auto")

    assert device.type in {"cpu", "cuda", "mps"}


def test_cpu_device() -> None:
    device = get_device("cpu")

    assert device.type == "cpu"


def test_invalid_device_raises_error() -> None:
    with pytest.raises(ValueError):
        get_device("invalid")
