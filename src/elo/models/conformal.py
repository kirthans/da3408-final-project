"""Finite-sample Mondrian split conformal intervals on untouched calibration residuals."""

import math

import numpy as np


def quantile(residuals, alpha):
    """Select the finite-sample corrected residual order statistic (DESIGN D7).

    Args:
        residuals: Nonempty absolute errors on untouched calibration rows.
        alpha: Miscoverage fraction.

    Returns:
        The ceil((n+1)(1-alpha))-th residual, capped at the largest residual."""
    residuals = np.asarray(residuals)
    # The n+1 correction supplies finite-sample coverage; cap the rank for small samples.
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - alpha)))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def calibrate(residuals, classes, alpha=0.2):
    """Fit Mondrian interval radii by time-control class (DESIGN D7).

    Args:
        residuals: Absolute calibration errors, aligned with classes.
        classes: Time-control labels for the calibration rows.
        alpha: Target miscoverage fraction.

    Returns:
        Global and per-class radii, counts and alpha. Classes with fewer than
        50 observations use the global radius."""
    residuals, classes = np.asarray(residuals), np.asarray(classes)
    global_q = quantile(residuals, alpha)
    # Mondrian groups are time controls; sparse classes borrow the global radius (DESIGN D7).
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
