"""Population stability using the training bins, including a separate missing bin."""

import warnings

import numpy as np
import pandas as pd

from elo.models.aggregate import FEATURE_COLUMNS


def psi(reference_entry, values):
    """Measure distribution shift using fixed training bins (DESIGN D17).

    Args:
        reference_entry: Numeric edges or categories and training proportions.
        values: Current feature values, including missing observations.

    Returns:
        PSI with an explicit missing bin and a 1e-4 floor for both populations."""
    values = pd.Series(values)
    if reference_entry["type"] == "numeric":
        # Reuse training edges and the same right-closed bins; refitting would hide drift.
        counts = np.bincount(
            np.searchsorted(reference_entry["edges"], values.dropna().astype(float), side="left"),
            minlength=len(reference_entry["edges"]) + 1,
        )
        counts = np.append(counts, values.isna().sum())
    else:
        counts = np.array(
            [values.eq(c).sum() for c in reference_entry["categories"][:-1]] + [values.isna().sum()]
        )
    # A shared epsilon keeps empty bins from causing division by zero or log(0) (DESIGN D17).
    actual = np.maximum(counts / max(len(values), 1), 1e-4)
    expected = np.maximum(reference_entry["proportions"], 1e-4)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


_warned = set()


def psi_report(reference, frame):
    """Compute feature PSI while warning once about legacy numeric boolean bins.

    Args:
        reference: Training distribution entry for each model feature.
        frame: Current perspective rows with the model feature columns.

    Returns:
        PSI by feature, excluding incompatible legacy boolean references."""
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
