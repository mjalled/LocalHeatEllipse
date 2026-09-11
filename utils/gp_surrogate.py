"""Gaussian Process surrogate for local Nusselt curves.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np


GP_RE_RANGE = (10.0, 200.0)
GP_AOA_RANGE = (0.0, 90.0)
GP_AR_RANGE = (1.2, 5.0)

_MODEL = None


def _model_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "model_data" / "gp_model_data.npz"


def _load_model() -> dict:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    data = np.load(_model_data_path())
    model = {
        "X_train_raw": data["X_train_raw"],
        "X_train_scaled": data["X_train_scaled"],
        "alpha": data["alpha"],
        "constant_value": float(data["constant_value"]),
        "length_scale": data["length_scale"],
        "noise_level": float(data["noise_level"]),
        "x_min": data["x_min"],
        "x_max": data["x_max"],
        "y_train_mean": data["y_train_mean"],
        "y_train_std": data["y_train_std"],
    }

    # Precompute K^-1 once for fast predictive variance.
    Xtr = model["X_train_scaled"]
    c = model["constant_value"]
    ls = model["length_scale"]
    noise = model["noise_level"]

    diff = (Xtr[:, None, :] - Xtr[None, :, :]) / ls
    sqdist = np.sum(diff * diff, axis=2)
    K = c * np.exp(-0.5 * sqdist)
    K[np.diag_indices_from(K)] += noise
    K += 1e-12 * np.eye(K.shape[0])
    model["K_inv"] = np.linalg.inv(K)

    _MODEL = model
    return _MODEL


def _ensure_2d_inputs(re, aoa_deg, ar):
    re_arr = np.atleast_1d(np.asarray(re, dtype=np.float64))
    aoa_arr = np.atleast_1d(np.asarray(aoa_deg, dtype=np.float64))
    ar_arr = np.atleast_1d(np.asarray(ar, dtype=np.float64))
    if not (re_arr.shape == aoa_arr.shape == ar_arr.shape):
        raise ValueError("re, aoa_deg, and ar must have the same shape")
    X = np.column_stack([re_arr, aoa_arr, ar_arr])
    return X, re_arr.shape


def _scale_inputs(X_raw: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    denom = x_max - x_min
    if np.any(denom <= 0.0):
        raise ValueError("Invalid scaling range in model data")
    return (X_raw - x_min) / denom


def _cross_kernel(X_scaled: np.ndarray, X_train_scaled: np.ndarray, constant_value: float, length_scale: np.ndarray) -> np.ndarray:
    diff = (X_scaled[:, None, :] - X_train_scaled[None, :, :]) / length_scale
    sqdist = np.sum(diff * diff, axis=2)
    return constant_value * np.exp(-0.5 * sqdist)


def _warn_if_out_of_range(X_raw: np.ndarray) -> None:
    re = X_raw[:, 0]
    aoa = X_raw[:, 1]
    ar = X_raw[:, 2]

    out_of_range = (
        (re < GP_RE_RANGE[0]) | (re > GP_RE_RANGE[1]) |
        (aoa < GP_AOA_RANGE[0]) | (aoa > GP_AOA_RANGE[1]) |
        (ar < GP_AR_RANGE[0]) | (ar > GP_AR_RANGE[1])
    )
    if np.any(out_of_range):
        warnings.warn(
            "GP surrogate is being evaluated outside its recommended range: "
            f"Re in [{GP_RE_RANGE[0]}, {GP_RE_RANGE[1]}], "
            f"angle of attack in [{GP_AOA_RANGE[0]}, {GP_AOA_RANGE[1]}] deg, "
            f"AR in [{GP_AR_RANGE[0]}, {GP_AR_RANGE[1]}].",
            UserWarning,
            stacklevel=2,
        )


def predict_gp_curves(re, aoa_deg, ar, return_std: bool = False):
    """Predict full 360-point GP curves for one or many cases.

    Parameters
    ----------
    re, aoa_deg, ar:
        Scalars or arrays with matching shape.
    return_std:
        If True, also return per-theta predictive standard deviation.

    Returns
    -------
    mean : ndarray
        Shape (360,) for scalar input, else (n_cases, 360).
    std : ndarray, optional
        Same shape as mean, when return_std=True.
    """
    m = _load_model()
    X_raw, original_shape = _ensure_2d_inputs(re, aoa_deg, ar)
    _warn_if_out_of_range(X_raw)
    X_scaled = _scale_inputs(X_raw, m["x_min"], m["x_max"])

    K_trans = _cross_kernel(
        X_scaled,
        m["X_train_scaled"],
        m["constant_value"],
        m["length_scale"],
    )

    mean_norm = K_trans @ m["alpha"]
    mean = mean_norm * m["y_train_std"][None, :] + m["y_train_mean"][None, :]

    if original_shape == () or np.asarray(re).ndim == 0:
        mean_out = mean[0]
    else:
        mean_out = mean

    if not return_std:
        return mean_out

    quad = np.sum((K_trans @ m["K_inv"]) * K_trans, axis=1)
    var_base = (m["constant_value"] + m["noise_level"]) - quad
    var_base = np.maximum(var_base, 0.0)

    std = np.sqrt(var_base)[:, None] * np.abs(m["y_train_std"])[None, :]
    if original_shape == () or np.asarray(re).ndim == 0:
        std_out = std[0]
    else:
        std_out = std

    return mean_out, std_out


def predict_gp_curve(re: float, aoa_deg: float, ar: float, return_std: bool = False):
    """Convenience wrapper for one case: (Re, angle attack [deg], AR)."""
    return predict_gp_curves(re, aoa_deg, ar, return_std=return_std)


def get_gp_training_inputs_raw() -> np.ndarray:
    """Return stored GP training input matrix [Re, angle_attack_deg, AR]."""
    return _load_model()["X_train_raw"].copy()
