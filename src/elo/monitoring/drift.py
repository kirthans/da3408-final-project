"""Population stability using the training bins, including a separate missing bin."""

import warnings

import numpy as np
import pandas as pd

from elo.models.aggregate import FEATURE_COLUMNS


def psi(reference_entry, values):
    values = pd.Series(values)
    if reference_entry["type"] == "numeric":
        counts = np.bincount(
            np.searchsorted(reference_entry["edges"], values.dropna().astype(float), side="left"),
            minlength=len(reference_entry["edges"]) + 1,
        )
        counts = np.append(counts, values.isna().sum())
    else:
        counts = np.array(
            [values.eq(c).sum() for c in reference_entry["categories"][:-1]] + [values.isna().sum()]
        )
    actual = np.maximum(counts / max(len(values), 1), 1e-4)
    expected = np.maximum(reference_entry["proportions"], 1e-4)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


_warned = set()


def psi_report(reference, frame):
    result = {}
    for name in FEATURE_COLUMNS:
        entry = reference[name]
        if pd.api.types.is_bool_dtype(frame[name].dtype) and entry["type"] == "numeric":
            if name not in _warned:
                warnings.warn(f"Skipping legacy numeric boolean reference: {name}", stacklevel=2)
                _warned.add(name)
            continue
        result[name] = psi(entry, frame[name])
    return result
