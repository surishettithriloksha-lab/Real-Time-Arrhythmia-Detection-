"""CWNN architecture and CPU inference contracts."""

import time

import torch

from src.models import CWNN


def test_parameter_count_and_output_shape() -> None:
    """The model must remain compact and produce five logits."""
    model = CWNN().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) < 200_000
    with torch.inference_mode():
        assert model(torch.randn(1, 280)).shape == (1, 5)


def test_single_beat_execution_is_bounded() -> None:
    """A warm single-beat execution must satisfy the CI safety threshold."""
    torch.set_num_threads(1)
    model = CWNN().eval()
    beat = torch.randn(1, 280)
    with torch.inference_mode():
        model(beat)
        start = time.perf_counter()
        model(beat)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 15.0
