"""Regression tests for portable classifier input handling.

The module is loaded directly because importing ``medrax.tools`` eagerly loads
the optional heavyweight tools, which are intentionally outside this focused
classifier test environment.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torchxrayvision as xrv


MODULE_PATH = Path(__file__).parents[1] / "medrax" / "tools" / "classification.py"
SPEC = importlib.util.spec_from_file_location("classification_under_test", MODULE_PATH)
classification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(classification)


def test_uint8_path_preserves_previous_normalization():
    image = np.array([[0, 64], [128, 255]], dtype=np.uint8)

    patched = xrv.datasets.normalize(
        image, classification.normalization_maxval(image)
    )
    previous = xrv.datasets.normalize(image, 255)

    np.testing.assert_allclose(patched, previous)


def test_full_range_uint16_matches_equivalent_uint8_values():
    image_8bit = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_16bit = image_8bit.astype(np.uint16) * 257

    normalized_8bit = xrv.datasets.normalize(
        image_8bit, classification.normalization_maxval(image_8bit)
    )
    normalized_16bit = xrv.datasets.normalize(
        image_16bit, classification.normalization_maxval(image_16bit, 65535)
    )

    np.testing.assert_allclose(normalized_8bit, normalized_16bit)


def test_12bit_uint16_requires_and_honors_explicit_scale():
    image_12bit = np.array([[0, 1024], [2048, 4095]], dtype=np.uint16)

    with pytest.raises(ValueError, match="explicit input_maxval"):
        classification.normalization_maxval(image_12bit)

    normalized = xrv.datasets.normalize(
        image_12bit, classification.normalization_maxval(image_12bit, 4095)
    )
    np.testing.assert_allclose(
        normalized,
        [[-1024.0, -511.8748], [0.2501, 1024.0]],
        atol=1e-3,
    )


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.array([-1, 0], dtype=np.int16), "Unsupported CXR image dtype"),
        (np.array([0.0, np.nan], dtype=np.float32), "Floating-point"),
        (np.array([0.0, 1.1], dtype=np.float32), "Floating-point"),
    ],
)
def test_invalid_unscaled_inputs_fail_loudly(image, message):
    with pytest.raises((TypeError, ValueError), match=message):
        classification.normalization_maxval(image)


def test_auto_device_resolver_returns_available_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: False)

    assert classification.resolve_classifier_device("auto").type == "cpu"


def test_unavailable_explicit_cuda_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        classification.resolve_classifier_device("cuda")
