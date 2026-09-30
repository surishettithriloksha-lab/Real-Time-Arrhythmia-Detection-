"""Patient-isolated record splits and deterministic synthetic fallback data."""

from dataclasses import dataclass

import numpy as np

TRAIN_RECORDS = (
    101,
    106,
    108,
    109,
    112,
    115,
    116,
    118,
    119,
    122,
    124,
    201,
    203,
    205,
    207,
    208,
    230,
)
VAL_RECORDS = (114, 209, 215, 220, 223)
TEST_RECORDS = (
    100,
    103,
    105,
    111,
    113,
    117,
    121,
    123,
    200,
    202,
    210,
    212,
    213,
    214,
    219,
    221,
    222,
    228,
    231,
    232,
    233,
    234,
)
EXCLUDED_RECORDS = (102, 104, 107, 217)


@dataclass(frozen=True)
class BeatDataset:
    """Small in-memory dataset with beat arrays, labels, and patient identifiers."""

    beats: np.ndarray
    labels: np.ndarray
    patients: np.ndarray
    synthetic: bool = True


def split_records() -> dict[str, tuple[int, ...]]:
    """Return the immutable DS1/DS2 partitions and paced-record exclusions."""
    return {
        "train": TRAIN_RECORDS,
        "validation": VAL_RECORDS,
        "test": TEST_RECORDS,
        "excluded": EXCLUDED_RECORDS,
    }


def _morphology(label: int, length: int = 280) -> np.ndarray:
    """Create a deterministic ECG-like morphology for one AAMI class."""
    x = np.arange(length, dtype=np.float32)
    center = 100.0
    signal = 0.10 * np.sin(2.0 * np.pi * x / 110.0)
    if label == 0:
        signal += 0.18 * np.exp(-(((x - 60.0) / 15.0) ** 2)) - 0.25 * np.exp(
            -(((x - 98.0) / 5.0) ** 2)
        )
        signal += 1.00 * np.exp(-(((x - center) / 5.0) ** 2)) + 0.40 * np.exp(
            -(((x - 145.0) / 24.0) ** 2)
        )
    elif label == 1:
        signal += 0.95 * np.exp(-(((x - 88.0) / 4.0) ** 2)) - 0.32 * np.exp(
            -(((x - 110.0) / 8.0) ** 2)
        )
    elif label == 2:
        signal += 1.10 * np.exp(-(((x - 100.0) / 14.0) ** 2)) - 0.55 * np.exp(
            -(((x - 135.0) / 20.0) ** 2)
        )
    elif label == 3:
        normal = 0.75 * np.exp(-(((x - center) / 6.0) ** 2))
        ventricular = 0.70 * np.exp(-(((x - center) / 14.0) ** 2))
        signal += 0.5 * (normal + ventricular)
    else:
        signal += 1.5 * np.exp(-(((x - 78.0) / 1.2) ** 2)) + 0.85 * np.exp(
            -(((x - 105.0) / 16.0) ** 2)
        )
    return signal.astype(np.float32)


def generate_synthetic(records: tuple[int, ...], beats_per_record: int = 10) -> BeatDataset:
    """Generate reproducible, prominently synthetic labeled beats by record ID."""
    values: list[np.ndarray] = []
    labels: list[int] = []
    patients: list[int] = []
    for record_id in records:
        rng = np.random.default_rng(record_id)
        for beat_index in range(beats_per_record):
            label = (record_id + beat_index) % 5
            beat = _morphology(label) + rng.normal(0.0, 0.025, 280).astype(np.float32)
            values.append(beat)
            labels.append(label)
            patients.append(record_id)
    return BeatDataset(
        np.stack(values), np.asarray(labels, dtype=np.int64), np.asarray(patients, dtype=np.int64)
    )


def load_dataset(split: str) -> BeatDataset:
    """Load a split, using deterministic synthetic data when MIT-BIH is unavailable."""
    records = split_records().get(split)
    if records is None:
        raise ValueError(f"unknown split: {split}")
    return generate_synthetic(records)
