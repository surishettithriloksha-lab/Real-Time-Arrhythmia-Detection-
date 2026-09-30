"""Signal normalization, beat windowing, and robustness transformations."""

import numpy as np


def extract_beat(signal: np.ndarray, r_peak: int, pre: int = 99, post: int = 180) -> np.ndarray:
    """Extract the asymmetric 280-sample beat window, padding at boundaries."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if pre + post + 1 != 280:
        raise ValueError("window geometry must contain exactly 280 samples")
    start, stop = r_peak - pre, r_peak + post + 1
    result = np.zeros(280, dtype=np.float32)
    left, right = max(0, start), min(signal.size, stop)
    result[left - start:right - start] = signal[left:right]
    return result


def normalize_signal(beat: np.ndarray) -> np.ndarray:
    """Standardize a beat while safely handling constant signals."""
    values = np.asarray(beat, dtype=np.float32)
    scale = float(values.std())
    return (values - float(values.mean())) / max(scale, 1e-6)


def add_gaussian_noise(beat: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise at the requested signal-to-noise ratio."""
    power = float(np.mean(np.square(beat)))
    noise_power = power / max(10.0 ** (snr_db / 10.0), 1e-12)
    return np.asarray(beat) + rng.normal(0.0, np.sqrt(noise_power), beat.shape)


def add_baseline_drift(
    beat: np.ndarray, sample_rate: int = 360, amplitude: float = 0.5
) -> np.ndarray:
    """Add a 0.2-Hz sinusoidal baseline drift in normalized signal units."""
    time = np.arange(beat.size, dtype=np.float32) / sample_rate
    return np.asarray(beat) + amplitude * np.sin(2.0 * np.pi * 0.2 * time)


def shift_beat(beat: np.ndarray, shift: int) -> np.ndarray:
    """Shift a beat by samples with zero padding rather than circular wraparound."""
    result = np.zeros_like(beat)
    if shift >= 0:
        result[shift:] = beat[: beat.size - shift] if shift < beat.size else 0
    else:
        amount = abs(shift)
        result[: beat.size - amount] = beat[amount:] if amount < beat.size else 0
    return result


def augment_training_beat(beat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply training-only timing jitter and baseline drift augmentation."""
    jittered = shift_beat(beat, int(rng.integers(-3, 4)))
    return add_baseline_drift(jittered, amplitude=0.15)


def select_mlii(signal: np.ndarray, sig_name: tuple[str, ...]) -> np.ndarray:
    """Select MLII dynamically from record metadata, never assuming channel zero."""
    try:
        index = sig_name.index("MLII")
    except ValueError as exc:
        raise ValueError("record does not contain an MLII lead") from exc
    if signal.ndim != 2 or index >= signal.shape[1]:
        raise ValueError("signal shape is incompatible with sig_name")
    return signal[:, index]
