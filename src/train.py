"""Training entry point with augmentation and domain-adversarial objective."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .config import SETTINGS
from .data import BeatDataset, load_dataset
from .models import CWNN
from .signal_processing import augment_training_beat, normalize_signal


def focal_loss(logits: Tensor, labels: Tensor, gamma: float = 2.0) -> Tensor:
    """Compute multi-class focal loss."""
    cross_entropy = nn.functional.cross_entropy(logits, labels, reduction="none")
    return ((1.0 - torch.exp(-cross_entropy)) ** gamma * cross_entropy).mean()


def train_epoch(model: CWNN, dataset: BeatDataset, optimizer: torch.optim.Optimizer) -> float:
    """Train one deterministic full-batch epoch with DANN loss."""
    rng = np.random.default_rng(SETTINGS.seed)
    beats = np.stack([augment_training_beat(normalize_signal(beat), rng) for beat in dataset.beats])
    inputs = torch.from_numpy(beats)
    labels = torch.from_numpy(dataset.labels)
    patients = torch.from_numpy(dataset.patients % 17)
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs, return_domain=True)
    arrhythmia = outputs["arrhythmia"]
    patient = outputs["patient"]
    class_loss = focal_loss(arrhythmia, labels)
    counts = torch.bincount(patients, minlength=17).float().clamp_min(1.0)
    weights = len(patients) / (17.0 * counts)
    domain_loss = nn.functional.cross_entropy(patient, patients, weight=weights)
    total = class_loss + 0.1 * domain_loss
    total.backward()
    optimizer.step()
    return float(total.detach())


def main() -> None:
    """Train a compact checkpoint from synthetic fallback data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=SETTINGS.checkpoint)
    args = parser.parse_args()
    torch.manual_seed(SETTINGS.seed)
    dataset = load_dataset("train")
    model = CWNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = [train_epoch(model, dataset, optimizer) for _ in range(args.epochs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "synthetic": dataset.synthetic}, args.output)
    print(
        json.dumps(
            {"checkpoint": str(args.output), "losses": losses, "synthetic": dataset.synthetic}
        )
    )


if __name__ == "__main__":
    main()
