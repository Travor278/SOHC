from __future__ import annotations

import pytest

from craic_pipeline.train_sac import _resolve_device


def test_resolve_device_accepts_cpu():
    """SAC CLI device resolver keeps explicit CPU runs deterministic."""
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_unknown_name():
    """SAC CLI fails early for misspelled device names."""
    with pytest.raises(ValueError):
        _resolve_device("tpu")
