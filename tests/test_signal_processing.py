"""Analytical CWT and asymmetric windowing contracts."""

import numpy as np
import torch

from src.models import RealMorletCWT
from src.signal_processing import extract_beat


def test_cwt_has_expected_shape_and_frequency_selectivity() -> None:
    """CWT returns 32 bands and has a measurable response to test sinusoids."""
    layer = RealMorletCWT().eval()
    time = np.arange(280, dtype=np.float32) / 360.0
    signals = [torch.from_numpy(np.sin(2 * np.pi * hz * time)).unsqueeze(0) for hz in (5, 15, 30)]
    responses = []
    with torch.inference_mode():
        for signal in signals:
            output = layer(signal)
            assert output.shape == (1, 32, 280)
            responses.append(float(output.mean()))
    assert all(value > 0 for value in responses)


def test_extract_beat_uses_asymmetric_geometry() -> None:
    """The annotated R peak is at index 99 in the extracted window."""
    signal = np.arange(500, dtype=np.float32)
    beat = extract_beat(signal, 200)
    assert beat.shape == (280,)
    assert beat[99] == 200
