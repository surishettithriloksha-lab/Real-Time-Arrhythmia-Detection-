"""CPU inference latency benchmark for one 280-sample beat."""

import argparse
import statistics
import time

import torch

from .models import CWNN


def median_latency(iterations: int = 1000) -> float:
    """Return median single-beat inference latency in milliseconds."""
    torch.set_num_threads(1)
    model = CWNN().eval()
    beat = torch.randn(1, 280)
    samples = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = time.perf_counter()
            model(beat)
            samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    """Print the P50 latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    print(f"P50 latency: {median_latency(args.iterations):.3f} ms")


if __name__ == "__main__":
    main()
