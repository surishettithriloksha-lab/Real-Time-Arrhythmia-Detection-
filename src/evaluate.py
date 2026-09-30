"""Evaluate a checkpoint on patient-independent DS2 and robustness perturbations."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .config import SETTINGS
from .data import load_dataset
from .metrics import classification_metrics
from .models import CWNN
from .signal_processing import add_baseline_drift, add_gaussian_noise, normalize_signal, shift_beat


def predict(model: CWNN, beats: np.ndarray) -> np.ndarray:
    """Run batched inference on normalized beats."""
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(np.stack([normalize_signal(b) for b in beats]))
        return model(inputs).argmax(dim=1).numpy()


def evaluate_variant(
    model: CWNN,
    beats: np.ndarray,
    labels: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Apply a perturbation and return its Macro-F1."""
    changed = np.stack([transform(beat) for beat in beats])
    return float(classification_metrics(labels, predict(model, changed))["macro_f1"])


def main() -> None:
    """Print a JSON evaluation report without failing on robustness threshold misses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SETTINGS.checkpoint)
    parser.add_argument("--run-robustness", action="store_true")
    args = parser.parse_args()
    model = CWNN()
    if args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    dataset = load_dataset("test")
    baseline_predictions = predict(model, dataset.beats)
    baseline = float(classification_metrics(dataset.labels, baseline_predictions)["macro_f1"])
    rng = np.random.default_rng(SETTINGS.seed)
    report: dict[str, object] = {
        "watermark": SETTINGS.synthetic_watermark,
        "baseline_macro_f1": baseline,
    }
    if args.run_robustness:
        variants = {
            "gaussian_noise_20db": lambda beat: add_gaussian_noise(beat, 20.0, rng),
            "baseline_drift_0.2hz": add_baseline_drift,
            "sampling_jitter": lambda beat: shift_beat(beat, 3),
        }
        report["robustness"] = {
            name: evaluate_variant(model, dataset.beats, dataset.labels, fn)
            for name, fn in variants.items()
        }
    print(SETTINGS.synthetic_watermark)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
