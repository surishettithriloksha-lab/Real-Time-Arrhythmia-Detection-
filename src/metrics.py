"""Evaluation metrics for five-class AAMI arrhythmia classification."""

from collections.abc import Iterable

import numpy as np

from .config import AAMI_CLASSES


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, object]:
    """Calculate sensitivity, specificity, precision, F1, and present-class Macro-F1."""
    truth, pred = np.asarray(list(y_true)), np.asarray(list(y_pred))
    per_class: dict[str, dict[str, float]] = {}
    f1_values = []
    for index, name in enumerate(AAMI_CLASSES):
        tp = float(np.sum((truth == index) & (pred == index)))
        fn = float(np.sum((truth == index) & (pred != index)))
        fp = float(np.sum((truth != index) & (pred == index)))
        tn = float(np.sum((truth != index) & (pred != index)))
        support = tp + fn
        sensitivity = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        per_class[name] = {
            "Se": sensitivity,
            "Sp": specificity,
            "+P": precision,
            "F1": f1,
            "support": support,
        }
        if support > 0:
            f1_values.append(f1)
    return {"per_class": per_class, "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0}
