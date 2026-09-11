"""Symbolic-regression surrogates for local Nusselt curves.
"""

from __future__ import annotations

import warnings
import numpy as np


THETA_DEG_DEFAULT = np.arange(360, dtype=np.float64)
THETA_RAD_DEFAULT = np.deg2rad(THETA_DEG_DEFAULT)

SR_RE_RANGE = (10.0, 200.0)
SR_AOA_RANGE = (0.0, 90.0)
SR_AR_RANGE = (1.2, 5.0)
SR_REDUCED_ACCURACY_AR = 2.0
SR_REDUCED_ACCURACY_AOA = 40.0


def _validate_inputs(re, aoa_deg, ar):
    re_v = np.asarray(re, dtype=np.float64)
    aoa_v = np.asarray(aoa_deg, dtype=np.float64)
    ar_v = np.asarray(ar, dtype=np.float64)

    if np.any(re_v < 0.0):
        raise ValueError("Re must be non-negative")
    if np.any(ar_v <= 0.0):
        raise ValueError("AR must be positive")
    return re_v, np.deg2rad(aoa_v), ar_v


def _warn_if_out_of_range(re_v, aoa_deg_v, ar_v):
    out_of_range = (
        (re_v < SR_RE_RANGE[0]) | (re_v > SR_RE_RANGE[1]) |
        (aoa_deg_v < SR_AOA_RANGE[0]) | (aoa_deg_v > SR_AOA_RANGE[1]) |
        (ar_v < SR_AR_RANGE[0]) | (ar_v > SR_AR_RANGE[1])
    )
    if np.any(out_of_range):
        warnings.warn(
            "SR surrogate is being evaluated outside its recommended range: "
            f"Re in [{SR_RE_RANGE[0]}, {SR_RE_RANGE[1]}], "
            f"angle of attack in [{SR_AOA_RANGE[0]}, {SR_AOA_RANGE[1]}] deg, "
            f"AR in [{SR_AR_RANGE[0]}, {SR_AR_RANGE[1]}].",
            UserWarning,
            stacklevel=2,
        )


def _warn_reduced_accuracy(aoa_deg_v, ar_v):
    reduced = (ar_v > SR_REDUCED_ACCURACY_AR) & (aoa_deg_v > SR_REDUCED_ACCURACY_AOA)
    if np.any(reduced):
        warnings.warn(
            "SR accuracy is reduced for AR > 2 and angle of attack > 40 deg; "
            "the GP surrogate is preferred in this regime.",
            UserWarning,
            stacklevel=2,
        )


def _theta_to_rad(theta_deg=None):
    if theta_deg is None:
        return THETA_RAD_DEFAULT
    return np.deg2rad(np.asarray(theta_deg, dtype=np.float64))


def predict_sr_theta_value(re: float, aoa_deg: float, ar: float, theta_deg: float, complexity: int = 30) -> float:
    """Predict SR value at one theta angle (in degrees)."""
    y = predict_sr_theta_curve(re, aoa_deg, ar, theta_deg=np.array([theta_deg], dtype=np.float64), complexity=complexity)
    return float(y[0])


def predict_sr_theta_curve(re: float, aoa_deg: float, ar: float, theta_deg=None, complexity: int = 30):
    """Predict SR theta-dependent distribution for one case.

    Parameters
    ----------
    re, aoa_deg, ar:
        Physical inputs.
    theta_deg:
        Theta angle(s) in degrees. If None, uses 0..359.
    complexity:
        18 or 30.
    """
    re_v, alpha, ar_v = _validate_inputs(re, aoa_deg, ar)
    aoa_deg_v = np.asarray(aoa_deg, dtype=np.float64)
    _warn_if_out_of_range(re_v, aoa_deg_v, ar_v)
    _warn_reduced_accuracy(aoa_deg_v, ar_v)
    theta = _theta_to_rad(theta_deg)

    s = np.sin(theta)
    c = np.cos(theta)

    if complexity == 18:
        # (sqrt(Re) * exp((c * (((c * sqrt(AR)) - -3.189618) - alpha)) / 3.6124845)) / 2.1695602
        out = (
            np.sqrt(re_v)
            * np.exp((c * (((c * np.sqrt(ar_v)) - -3.189618) - alpha)) / 3.6124845)
        ) / 2.1695602
        return np.asarray(out, dtype=np.float64)

    if complexity == 30:
        # (sqrt(Re) / 2.516005) * ((alpha * (exp(s) * 0.19265835)) + exp((c / 2.8050222) * (((((-1.8071486 / AR) - -2.0788376) * c) - -3.9651434) - (alpha - -1.008985))))
        out = (
            np.sqrt(re_v) / 2.516005
        ) * (
            (alpha * (np.exp(s) * 0.19265835))
            + np.exp(
                (c / 2.8050222)
                * (((((-1.8071486 / ar_v) - -2.0788376) * c) - -3.9651434) - (alpha - -1.008985))
            )
        )
        return np.asarray(out, dtype=np.float64)

    raise ValueError("complexity must be 18 or 30")
