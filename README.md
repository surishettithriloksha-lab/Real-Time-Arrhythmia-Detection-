# Patient-Independent Real-Time Arrhythmia Detection Using CWNN

> ⚠️ **Important:** The default version of this project uses **synthetic ECG data** for testing. It is **not MIT-BIH benchmark data** unless the MIT-BIH dataset is explicitly provided.

## 📌 About the Project

This project detects **five types of ECG heartbeats** using a lightweight **Continuous Wavelet Neural Network (CWNN)**.

The system converts ECG signals into a time-frequency representation using a **Morlet Continuous Wavelet Transform (CWT)** and then uses a lightweight CNN to classify each heartbeat.

The model is designed to work across **different patients**, rather than learning patterns specific to only one patient.

### Classes

* **N** — Normal
* **S** — Supraventricular ectopic
* **V** — Ventricular ectopic
* **F** — Fusion
* **Q** — Unknown / Other

---

## 🧠 How It Works

```text
ECG Signal
    ↓
R-Peak Detection
    ↓
Beat Extraction
    ↓
Morlet CWT
    ↓
Time-Frequency Image
    ↓
Lightweight CNN
    ↓
Arrhythmia Classification
    ↓
N / S / V / F / Q
```

### Patient-Independent Learning

The model also uses a **Gradient Reversal Layer (GRL)** with a patient classifier.

```text
                    ┌──→ Arrhythmia Classifier
                    │
ECG → CNN Features ─┤
                    │
                    └──→ GRL → Patient Classifier
```

This encourages the model to learn ECG features that are useful for arrhythmia detection while reducing dependence on patient-specific characteristics.

---

## ⚙️ Model Architecture

The main pipeline is:

```text
ECG Record
    ↓
Dynamic MLII Lead Selection
    ↓
Beat Window [R-99 : R+180]
    ↓
Hann Window
    ↓
Morlet CWT
    ↓
3 × Depthwise-Separable Conv2D
    ↓
SE Blocks
    ↓
Feature Representation
    ├──→ 5-Class AAMI Classifier
    └──→ GRL → 17-Class Patient Classifier
```

### CWT Configuration

* Morlet wavelet
* `ω₀ = 5`
* 32 logarithmic scales
* Frequency range: **1–45 Hz**
* 65-tap Hann window
* L1 normalization for each scale

The complete model contains **less than 200,000 parameters**, making it suitable for lightweight and real-time applications.

---

## 📊 Dataset Setup

When MIT-BIH data is available, the project uses patient-independent record splits.

| Dataset    | Records                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------ |
| Training   | 101, 106, 108, 109, 112, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 230                          |
| Validation | 114, 209, 215, 220, 223                                                                                      |
| Test       | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |
| Excluded   | 102, 104, 107, 217                                                                                           |

There is **no patient overlap** between training, validation, and testing.

### Synthetic Data

If MIT-BIH data is not available, the project automatically uses deterministic synthetic ECG signals representing:

* Normal beats
* Supraventricular beats
* Ventricular beats
* Fusion beats
* Unknown beats

> **Synthetic results are only for testing and validating the pipeline. They must not be reported as MIT-BIH benchmark results.**

---

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY>
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -e .
pip install pytest ruff flake8
```

---

## ▶️ Run the Project

### 1. Run Tests

```bash
pytest tests/
```

### 2. Train the Model

For a quick test:

```bash
python -m src.train --epochs 2
```

### 3. Evaluate the Model

```bash
python -m src.evaluate
```

Run robustness testing:

```bash
python -m src.evaluate --run-robustness
```

### 4. Run Benchmark

```bash
python -m src.benchmark --iterations 1000
```

### 5. Start the Dashboard

```bash
streamlit run app/dashboard.py
```

---

## 📁 Project Workflow

```text
Raw ECG
   ↓
Preprocessing
   ↓
R-Peak Detection
   ↓
Beat Segmentation
   ↓
Morlet CWT
   ↓
CWNN
   ↓
Prediction
   ↓
Arrhythmia Class + Confidence
   ↓
Dashboard
```

---

## 🔬 Why This Approach?

### Continuous Wavelet Transform

CWT helps represent ECG morphology in both **time and frequency**, making it useful for detecting short-duration changes in heartbeat patterns.

### Lightweight CNN

Depthwise-separable convolutions reduce the number of parameters and computational cost.

### SE Blocks

Squeeze-and-Excitation blocks help the network focus on important learned features.

### Patient-Independent Learning

The GRL-based patient classifier helps reduce patient-specific bias and encourages more generalizable ECG representations.

---

## 🛡️ Robustness

The evaluation pipeline includes robustness testing.

Failures are reported as **warnings/results in JSON format** instead of causing the entire evaluation process to crash.

---

## 🔧 Configuration

The project does not require hard-coded absolute paths or secrets.

The following settings can be overridden using environment variables:

* Sample rate
* Random seed
* Data directory
* Model checkpoint path

---

## 📌 Current Status

**Core model and evaluation pipeline implemented.**

The project currently supports:

* ECG preprocessing
* Dynamic MLII lead selection
* Beat extraction
* Morlet CWT
* Lightweight CWNN
* Five-class arrhythmia classification
* Patient-adversarial learning
* Robustness evaluation
* Benchmarking
* Streamlit dashboard
* Deterministic synthetic-data fallback

### Important

The synthetic fallback allows the complete system to be tested without downloading MIT-BIH data.

For **research-grade benchmark results**, provide the actual MIT-BIH records and report those results separately from synthetic-data results.

---

## 📜 Disclaimer

This project is a **research and educational prototype**.

It is not a medical device and should not be used to make clinical decisions or diagnose patients.
