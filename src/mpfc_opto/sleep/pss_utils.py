"""Spectral-slope (PSS) computation, independent of DataJoint.

Separated from `pss` so these can be imported, and tested, without a
database: importing a module that declares tables opens a connection.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def fit_pss_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_range: tuple[float, float] = (4.0, 90.0),
) -> tuple[float, float, float]:
    """Fit log10(PSD) vs log10(freq) and return the inverted slope.

    Returns
    -------
    pss : float
        Negative slope so that larger values correspond to steeper 1/f falloff.
    intercept : float
        Intercept in log10 space.
    slope : float
        Raw slope (before sign inversion).
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    mask = (
        (freqs >= f_range[0])
        & (freqs <= f_range[1])
        & (freqs > 0)
        & np.isfinite(freqs)
        & np.isfinite(psd)
        & (psd > 0)
    )
    if mask.sum() < 5:
        raise ValueError("Not enough valid frequency bins to fit PSS.")

    x = np.log10(freqs[mask])
    y = np.log10(psd[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return -float(slope), float(intercept), float(slope)


def compute_pss_windows(
    x: np.ndarray,
    fs: float,
    window_s: float = 2.0,
    step_s: float = 1.0,
    f_range: tuple[float, float] = (4.0, 90.0),
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Compute sliding-window PSS from a wideband LFP trace.

    Returns relative window centers in seconds from the start of x.
    For absolute timestamps, prefer compute_pss_windows_with_timestamps().
    """
    x = np.asarray(x, dtype=float)
    nperseg = int(round(window_s * fs))
    step = int(round(step_s * fs))

    if nperseg < 16:
        raise ValueError("window_s is too small for stable PSS estimation.")
    if step <= 0:
        raise ValueError("step_s must be positive.")
    if len(x) < nperseg:
        raise ValueError("Signal shorter than one window.")

    starts = np.arange(0, len(x) - nperseg + 1, step)
    t_centers = np.empty(len(starts), dtype=float)
    pss_vals = np.empty(len(starts), dtype=float)
    slopes = np.empty(len(starts), dtype=float)
    intercepts = np.empty(len(starts), dtype=float)
    psd_rows = []
    freqs_out = None

    for i, start in enumerate(starts):
        seg = x[start : start + nperseg]
        freqs, psd = welch(
            seg,
            fs=fs,
            nperseg=nperseg,
            noverlap=0,
            detrend="constant",
            scaling="density",
        )
        if freqs_out is None:
            freqs_out = freqs
        pss, intercept, slope = fit_pss_from_psd(freqs, psd, f_range=f_range)
        pss_vals[i] = pss
        slopes[i] = slope
        intercepts[i] = intercept
        t_centers[i] = (start + nperseg / 2) / fs
        psd_rows.append(psd)

    return (
        t_centers,
        pss_vals,
        slopes,
        intercepts,
        freqs_out,
        np.asarray(psd_rows),
    )


def compute_pss_windows_with_timestamps(
    x: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    window_s: float = 2.0,
    step_s: float = 1.0,
    f_range: tuple[float, float] = (4.0, 90.0),
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Compute sliding-window PSS and return absolute window-center timestamps.

    This is the preferred helper when the LFP dataframe index is already in
    absolute time (e.g., Unix seconds), because the returned timestamps will
    match the sleep scoring timebase.
    """
    x = np.asarray(x, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if len(x) != len(timestamps):
        raise ValueError("x and timestamps must have the same length.")

    nperseg = int(round(window_s * fs))
    step = int(round(step_s * fs))

    if nperseg < 16:
        raise ValueError("window_s is too small for stable PSS estimation.")
    if step <= 0:
        raise ValueError("step_s must be positive.")
    if len(x) < nperseg:
        raise ValueError("Signal shorter than one window.")

    starts = np.arange(0, len(x) - nperseg + 1, step)
    t_centers = np.empty(len(starts), dtype=float)
    pss_vals = np.empty(len(starts), dtype=float)
    slopes = np.empty(len(starts), dtype=float)
    intercepts = np.empty(len(starts), dtype=float)
    psd_rows = []
    freqs_out = None

    for i, start in enumerate(starts):
        seg = x[start : start + nperseg]
        freqs, psd = welch(
            seg,
            fs=fs,
            nperseg=nperseg,
            noverlap=0,
            detrend="constant",
            scaling="density",
        )
        if freqs_out is None:
            freqs_out = freqs
        pss, intercept, slope = fit_pss_from_psd(freqs, psd, f_range=f_range)
        pss_vals[i] = pss
        slopes[i] = slope
        intercepts[i] = intercept
        # Center timestamp from the original absolute timestamps
        t_centers[i] = float(timestamps[start + nperseg // 2])
        psd_rows.append(psd)

    return (
        t_centers,
        pss_vals,
        slopes,
        intercepts,
        freqs_out,
        np.asarray(psd_rows),
    )
