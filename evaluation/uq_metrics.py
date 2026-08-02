"""BPM2025 / UQ4PPM metric quartet for probabilistic remaining-time prediction.

Faithful port of the evaluation in Amiri Elyasi, van der Aa & Stuckenschmidt,
"A Simple and Calibrated Approach for Uncertainty-Aware Remaining Time
Prediction" (BPM 2025) — github.com/keyvan-amiri/UQ4PPM,
calibrated_regression.py:

    MAE   mean |pred_mean - y|                                 (days, lower)
    MA    uct.metrics_calibration.miscalibration_area(mu, sigma, y):
          area between the calibration curve PICP(p) and the parity line,
          with Gaussian central intervals mu +/- z((1+p)/2) sigma over a
          uniform grid of expected proportions (100 bins). In [0, 0.5].
    MPIW  the paper's "MPIW" column is uct.sharpness(sigma) =
          sqrt(mean(sigma^2)) (RMS of the predictive stds, days, lower).
    AURG  sparsification: sort by sigma descending, remove 1% at a time,
          MAE of the retained set vs fraction removed ("UQ curve"); AURG =
          trapz(random curve) - trapz(UQ curve), random removal with
          np.random.seed(42). Higher is better; ~0 means uninformative
          uncertainty; negative means misleading uncertainty.

All inputs are 1-D numpy arrays in days. We additionally provide a
distribution-free MA computed from empirical sample quantiles (`ma_sample`)
for models whose predictive law is given by Monte-Carlo samples rather than
a Gaussian (mu, sigma) — the Gaussian MA is still reported for benchmark
comparability.
"""

import numpy as np
from scipy.stats import norm

# np.trapz was renamed np.trapezoid in numpy 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def mae(pred_mean, y_true):
    return float(np.mean(np.abs(pred_mean - y_true)))


def miscalibration_area(pred_mean, pred_std, y_true, n_bins=100):
    """Gaussian-interval MA, replicating uncertainty_toolbox (interval type)."""
    exp_p = np.linspace(0.0, 1.0, n_bins)
    resid = (y_true - pred_mean) / np.clip(pred_std, 1e-12, None)
    lower = norm.ppf(0.5 - exp_p / 2.0)            # [n_bins]
    upper = norm.ppf(0.5 + exp_p / 2.0)
    within = (resid[None, :] >= lower[:, None]) & (resid[None, :] <= upper[:, None])
    obs_p = within.mean(axis=1)
    return float(_trapz(np.abs(obs_p - exp_p), exp_p))


def sharpness_rms(pred_std):
    """uct.sharpness: RMS of predictive stds = the paper's 'MPIW' column."""
    return float(np.sqrt(np.mean(np.asarray(pred_std, dtype=float) ** 2)))


def aurg(pred_mean, y_true, pred_std, n_steps=100, seed=42):
    """Area Under the Random Gain curve (UQ4PPM get_sparsification, verbatim
    logic incl. the fixed random seed). Returns (ause, aurg)."""
    pred_mean = np.asarray(pred_mean, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)
    n_samples = len(pred_mean)
    step_size = int(n_samples / n_steps)
    if step_size == 0:
        return float("nan"), float("nan")
    rng_state = np.random.get_state()
    np.random.seed(seed)
    err = np.abs(pred_mean - y_true)

    def curve(order):
        out = []
        for i in range(n_steps):
            idx = order[i * step_size:]
            out.append(float(np.mean(err[idx])))
        return out

    mae_oracle = curve(np.argsort(err)[::-1])
    mae_random = []
    for i in range(n_steps):
        idx = np.random.choice(n_samples, n_samples - i * step_size, replace=False)
        mae_random.append(float(np.mean(err[idx])))
    mae_uq = curve(np.argsort(pred_std)[::-1])
    np.random.set_state(rng_state)

    x = np.linspace(0, 1, n_steps)
    ause = float(_trapz(mae_uq, x) - _trapz(mae_oracle, x))
    aurg_ = float(_trapz(mae_random, x) - _trapz(mae_uq, x))
    return ause, aurg_


def ma_sample(rt_samples, y_true, p_grid=None):
    """Distribution-free MA from per-instance MC samples.

    rt_samples: [N, S] array; central p-intervals from empirical quantiles.
    """
    if p_grid is None:
        p_grid = np.linspace(0.05, 0.95, 19)
    qs_lo = np.quantile(rt_samples, (1 - p_grid) / 2, axis=1)     # [P, N]
    qs_hi = np.quantile(rt_samples, (1 + p_grid) / 2, axis=1)
    obs = ((y_true[None, :] >= qs_lo) & (y_true[None, :] <= qs_hi)).mean(axis=1)
    return float(_trapz(np.abs(obs - p_grid), p_grid)), {
        "p": p_grid.tolist(), "picp": obs.tolist()}


def quartet(pred_mean, pred_std, y_true):
    """The four headline numbers as a dict."""
    _, g = aurg(pred_mean, y_true, pred_std)
    return {
        "mae": mae(pred_mean, y_true),
        "ma": miscalibration_area(pred_mean, pred_std, y_true),
        "mpiw_rms_sigma": sharpness_rms(pred_std),
        "aurg": g,
        "n": int(len(y_true)),
    }
