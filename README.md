# Patient-Independent Real-Time Arrhythmia Detection Using CWNN

> **[SYNTHETIC TEST DATA - NOT MIT-BIH BENCHMARK]**

This repository implements a compact, real-valued Morlet Continuous Wavelet Neural Network (CWNN) for five-class AAMI beat classification. MIT-BIH ingestion is intentionally optional: the runnable default is a deterministic synthetic fallback, prominently marked everywhere.

## Architecture

```text
ECG record -> dynamic MLII selection -> [R-99 : R+180] (280 samples)
                                      |
                                      v
                         Hann + L1 normalized Morlet CWT
                                      |
                                      v
                 3x depthwise-separable Conv2D + SE blocks
                                      |
                         +------------+-------------+
                         |                          |
                    AAMI classifier          GRL (lambda=.1)
                    N/S/V/F/Q logits         17-class patient head
```

## Data partitions

| Partition | Records | Patient overlap |
|---|---|---|
| DS1 train | 101, 106, 108, 109, 112, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 230 | None |
| DS1 validation | 114, 209, 215, 220, 223 | None |
| DS2 unseen test | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | None |
| Excluded paced | 102, 104, 107, 217 | Rejected |

The `MLII` lead is resolved from `record.sig_name`; no channel position is assumed. Missing MIT-BIH data falls back to seeded morphologies for N, S, V, F, and Q.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff flake8
pytest tests/
python -m src.train --epochs 2
python -m src.evaluate --run-robustness
python -m src.benchmark --iterations 1000
streamlit run app/dashboard.py
```

## Design notes

- The analytical Morlet kernel uses `omega_0=5`, 32 logarithmic scales from 1–45 Hz, a 65-tap Hann taper, and strict per-scale L1 normalization.
- The full model is below 200,000 parameters and includes a DANN patient discriminator with inverse-frequency weighting.
- Robustness misses are reported as JSON warnings/results rather than uncaught exceptions.
- No secrets or absolute data paths are required. Environment variables can override sample rate, seed, data directory, and checkpoint path.
