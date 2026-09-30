"""Streamlit dashboard for sequential real-time beat inference."""

import numpy as np
import streamlit as st
import torch

from src.config import AAMI_CLASSES, SETTINGS
from src.data import load_dataset
from src.models import CWNN
from src.signal_processing import normalize_signal


@st.cache_resource
def load_model() -> CWNN:
    """Load a checkpoint when available, otherwise use an untrained model."""
    model = CWNN().eval()
    if SETTINGS.checkpoint.exists():
        model.load_state_dict(torch.load(SETTINGS.checkpoint, map_location="cpu")["model"])
    return model


st.set_page_config(page_title="CWNN Arrhythmia Monitor", layout="wide")
st.warning(SETTINGS.synthetic_watermark)
st.title("Patient-Independent Real-Time Arrhythmia Detection")
st.caption("Continuous Wavelet Neural Network • 280-sample asymmetric R-peak windows")

model = load_model()
dataset = load_dataset("test")
index = st.slider("Sequential heartbeat", 0, len(dataset.beats) - 1, 0)
beat = normalize_signal(dataset.beats[index])
with torch.inference_mode():
    logits = model(torch.from_numpy(beat).unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1).numpy()[0]

left, right = st.columns(2)
with left:
    st.subheader("Live beat signal")
    st.line_chart(beat)
with right:
    st.subheader("CWT class probabilities")
    st.bar_chart(dict(zip(AAMI_CLASSES, probabilities)))
st.metric("Predicted class", AAMI_CLASSES[int(np.argmax(probabilities))])
st.caption(
    f"Record/patient: {int(dataset.patients[index])} • "
    f"Ground truth: {AAMI_CLASSES[int(dataset.labels[index])]}"
)
