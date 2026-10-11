"""
Standalone PSS pipeline for Spyglass / DataJoint.

This module defines:
- PSSParams: parameters for PSS computation
- PSSSelection: what data to compute PSS from
- SleepPSS: computed PSS trace (timestamps + values)
- fetch_pss_series: interpolate a stored trace onto target timestamps

The PSD slope fitting itself lives in `pss_utils`, which has no DataJoint
dependency.

How to use in your sleep scoring pipeline:
1) Populate SleepPSS for the same recording / LFP source you use for scoring.
2) In SleepScoring.make_fetch(), fetch SleepPSS and interpolate onto your scoring timestamps.
3) Include pss as a feature when params.use_pss is True.
"""

from __future__ import annotations


import datajoint as dj
import numpy as np

from spyglass.lfp.analysis.v1 import lfp_band
from spyglass.utils import SpyglassMixin

from mpfc_opto.sleep.pss_utils import (
    compute_pss_windows_with_timestamps,
)

schema = dj.schema("denissemorales_pss")


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
