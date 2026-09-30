"""Configuration values for training, inference, and data handling."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    sample_rate: int = int(os.getenv("CWNN_SAMPLE_RATE", "360"))
    beat_length: int = int(os.getenv("CWNN_BEAT_LENGTH", "280"))
    num_scales: int = int(os.getenv("CWNN_NUM_SCALES", "32"))
    num_classes: int = 5
    seed: int = int(os.getenv("CWNN_SEED", "2024"))
    data_dir: Path = Path(os.getenv("CWNN_DATA_DIR", "data"))
    checkpoint: Path = Path(os.getenv("CWNN_CHECKPOINT", "checkpoints/cwnn.pt"))
    synthetic_watermark: str = "[SYNTHETIC TEST DATA - NOT MIT-BIH BENCHMARK]"


SETTINGS = Settings()
AAMI_CLASSES = ("N", "S", "V", "F", "Q")
