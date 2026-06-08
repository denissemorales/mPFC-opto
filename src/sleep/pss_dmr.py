"""
Standalone PSS pipeline for Spyglass / DataJoint.

This module defines:
- PSSParams: parameters for PSS computation
- PSSSelection: what data to compute PSS from
- SleepPSS: computed PSS trace (timestamps + values)
- helper utilities for interpolation and PSD slope fitting

How to use in your sleep scoring pipeline:
1) Populate SleepPSS for the same recording / LFP source you use for scoring.
2) In SleepScoring._fetch_data(), fetch SleepPSS and interpolate onto your scoring timestamps.
3) Include pss as a feature when params.use_pss is True.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import datajoint as dj
import numpy as np
from scipy.signal import welch

from spyglass.lfp.analysis.v1 import lfp_band
from spyglass.utils import SpyglassMixin

schema = dj.schema("denissemorales_pss")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def fit_pss_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_range: Tuple[float, float] = (4.0, 90.0),
) -> Tuple[float, float, float]:
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
    f_range: Tuple[float, float] = (4.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    f_range: Tuple[float, float] = (4.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

@schema
class PSSParams(SpyglassMixin, dj.Lookup):
    """Parameter sets for PSS computation."""

    definition = """
    pss_params_name: varchar(64)
    ---
    window_s: float          # Sliding PSD window length in seconds
    step_s: float            # Step between windows in seconds
    fmin: float              # Lower bound of slope fit (Hz)
    fmax: float              # Upper bound of slope fit (Hz)
    channel_aggregation: varchar(16)  # mean | median
    """

    contents = [
        {
            "pss_params_name": "default_4_90",
            "window_s": 2.0,
            "step_s": 1.0,
            "fmin": 4.0,
            "fmax": 90.0,
            "channel_aggregation": "mean",
        }
    ]


@schema
class PSSSelection(SpyglassMixin, dj.Manual):
    """What to compute PSS from."""

    definition = """
    -> PSSParams
    nwb_file_name: varchar(64)
    lfp_merge_id: uuid
    pss_source_filter_name: varchar(64)   # wideband source to compute PSS from
    filter_sampling_rate: float
    ---
    target_interval_list_name='': varchar(64)
    pss_notes='': varchar(255)
    """


@schema
class SleepPSS(SpyglassMixin, dj.Computed):
    """Computed PSS trace."""

    definition = """
    -> PSSSelection
    ---
    pss_timestamps: longblob
    pss_values: longblob
    pss_slopes: longblob
    pss_intercepts: longblob
    pss_freqs: longblob
    pss_psd: longblob
    """

    def make(self, key):
        sel = (PSSSelection & key).fetch1()
        params = (PSSParams & key).fetch1()

        lfp_df = self._fetch_wideband_lfp(sel)
        if lfp_df.empty:
            raise ValueError("Wideband LFP dataframe is empty.")

        if params["channel_aggregation"] == "mean":
            x = lfp_df.mean(axis=1).values
        elif params["channel_aggregation"] == "median":
            x = lfp_df.median(axis=1).values
        else:
            raise ValueError(
                f"Unsupported channel_aggregation={params['channel_aggregation']!r}; use mean or median."
            )

        fs = float(sel["filter_sampling_rate"])
        lfp_timestamps = np.asarray(lfp_df.index.values, dtype=float)
        if len(lfp_timestamps) != len(x):
            raise ValueError("LFP timestamps and signal length do not match.")

        t_pss, pss_vals, slopes, intercepts, freqs, psd = (
            compute_pss_windows_with_timestamps(
                x,
                timestamps=lfp_timestamps,
                fs=fs,
                window_s=float(params["window_s"]),
                step_s=float(params["step_s"]),
                f_range=(float(params["fmin"]), float(params["fmax"])),
            )
        )

        self.insert1(
            {
                **key,
                "pss_timestamps": t_pss,
                "pss_values": pss_vals,
                "pss_slopes": slopes,
                "pss_intercepts": intercepts,
                "pss_freqs": freqs,
                "pss_psd": psd,
            }
        )

    def _fetch_wideband_lfp(self, sel):
        """Fetch a wideband LFP dataframe from Spyglass.

        Adjust this if your project stores the source differently.
        """
        try:
            return (
                lfp_band.LFPBandV1
                & {
                    "lfp_merge_id": sel["lfp_merge_id"],
                    "filter_name": sel["pss_source_filter_name"],
                }
            ).fetch1_dataframe()
        except Exception as exc:
            raise RuntimeError(
                "Could not fetch wideband LFP from LFPBandV1. "
                "Check lfp_merge_id and pss_source_filter_name."
            ) from exc


# -----------------------------------------------------------------------------
# Convenience helper for SleepScoring
# -----------------------------------------------------------------------------

def fetch_pss_series(
    pss_key: dict,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Fetch a computed PSS trace and interpolate onto target timestamps."""
    pss_row = (SleepPSS & pss_key).fetch1()
    return np.interp(
        target_timestamps,
        np.asarray(pss_row["pss_timestamps"]),
        np.asarray(pss_row["pss_values"]),
        left=np.nan,
        right=np.nan,
    )
