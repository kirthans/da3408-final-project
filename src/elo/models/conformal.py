"""Finite-sample Mondrian split conformal intervals on untouched calibration residuals."""

import math

import numpy as np


def quantile(residuals, alpha):
    """The ceil((n+1)(1-alpha))-th order statistic, capped at n."""
    residuals = np.asarray(residuals)
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - alpha)))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def calibrate(residuals, classes, alpha=0.2):
    """Use the global quantile for classes with fewer than 50 observations."""
    residuals, classes = np.asarray(residuals), np.asarray(classes)
    global_q = quantile(residuals, alpha)
    counts = {str(c): int((classes == c).sum()) for c in np.unique(classes)}
    return dict(
        alpha=alpha,
        global_q=global_q,
        n=counts,
        q={
            c: quantile(residuals[classes == c], alpha) if n >= 50 else global_q
            for c, n in counts.items()
        },
    )


def radii(frame, calibration):
    """Return one radius per row, including a global fallback for unseen classes."""
    return np.array([calibration["q"].get(str(c), calibration["global_q"]) for c in frame.tc_class])
