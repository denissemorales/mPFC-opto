"""
Spyglass Up/Down State Detection Pipeline

Detects UP and DOWN states during NREM sleep using LFP slow oscillations
and multi-unit activity (MUA). Depends on SleepScoring for NREM intervals.

Author: DMR+Claude
Date: Mar 2026
"""

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from spyglass.common import IntervalList
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.lfp.analysis.v1 import lfp_band
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import SpyglassMixin

from sleep_scoring import SleepScoring, SleepScoringSelection

schema = dj.schema("denissemorales_updownstates")

VALID_METHODS = {"threshold", "gmm", "hilbert"}


@schema
class UpDownStateParams(SpyglassMixin, dj.Lookup):
    definition = """
    up_down_params_name: varchar(64)
    ---
    # Detection method
    method: varchar(32)              # Detection method (threshold | gmm | hilbert)

    # LFP slow oscillation parameters
    so_filter_name: varchar(64)      # Filter name for slow oscillation band (~0.5-4 Hz)
    so_smoothing: float              # Gaussian smoothing sigma for LFP (seconds)

    # MUA parameters
    mua_smoothing: float             # Gaussian smoothing sigma for MUA (seconds)
    mua_bin_size: float              # Bin size for spike rate estimation (seconds)

    # Threshold parameters (used when method='threshold')
    lfp_down_percentile: float       # LFP amplitude percentile defining DOWN states
    mua_down_percentile: float       # MUA rate percentile defining DOWN states

    # Minimum duration parameters
    min_down_duration: float         # Minimum DOWN state duration (seconds)
    min_up_duration: float           # Minimum UP state duration (seconds)

    # Feature weighting
    lfp_weight: float                # Weight for LFP signal in combined score (0-1)
    mua_weight: float                # Weight for MUA signal in combined score (0-1)
    """

    contents = [
        {
            "up_down_params_name": "default",
            "method": "threshold",
            "so_filter_name": "slow_oscillation 0.5-4 Hz",
            "so_smoothing": 0.1,
            "mua_smoothing": 0.05,
            "mua_bin_size": 0.01,
            "lfp_down_percentile": 25.0,
            "mua_down_percentile": 25.0,
            "min_down_duration": 0.1,
            "min_up_duration": 0.1,
            "lfp_weight": 0.5,
            "mua_weight": 0.5,
        }
    ]

    def insert1(self, row, **kwargs):
        if row.get("method") not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{row['method']}'. Must be one of: {VALID_METHODS}"
            )
        if abs(row.get("lfp_weight", 0) + row.get("mua_weight", 0) - 1.0) > 1e-6:
            raise ValueError("lfp_weight + mua_weight must equal 1.0")
        super().insert1(row, **kwargs)


@schema
class UpDownStateSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> UpDownStateParams
    -> SleepScoring                  # Provides NREM intervals + nwb_file_name
    -> lfp_band.LFPBandV1.proj(so_lfp_merge_id='lfp_merge_id',
                                so_filter_name='filter_name')
    -> SpikeSortingOutput.proj(spike_sorting_merge_id='merge_id')
    ---
    nwb_file_name: varchar(64)
    """


@schema
class UpDownStates(SpyglassMixin, dj.Computed):
    definition = """
    -> UpDownStateSelection
    ---
    state_labels: blob               # 0=DOWN, 1=UP, per time bin
    timestamps: blob                 # Timestamps for state labels
    down_duration: float             # Total DOWN state duration (seconds)
    up_duration: float               # Total UP state duration (seconds)
    down_percentage: float           # Percentage of NREM time in DOWN states
    up_percentage: float             # Percentage of NREM time in UP states
    n_down_states: int               # Number of detected DOWN state bouts
    n_up_states: int                 # Number of detected UP state bouts
    mean_down_duration: float        # Mean DOWN state duration (seconds)
    mean_up_duration: float          # Mean UP state duration (seconds)
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make(self, key):
        fetch_dict = self._fetch_data(key)
        result = self._compute_states(key, fetch_dict)
        self._store_results(key, result, fetch_dict["nwb_file_name"])

    # ==================== Tri-part helpers ====================

    def _fetch_data(self, key):
        """Fetch all upstream data needed for up/down detection."""
        params = (UpDownStateParams & key).fetch1()
        sel = (UpDownStateSelection & key).fetch1()
        nwb_file_name = sel["nwb_file_name"]

        # --- NREM intervals from SleepScoring ---
        sleep_key = {
            k: key[k]
            for k in SleepScoring.primary_key
            if k in key
        }
        state_labels, timestamps = (SleepScoring & sleep_key).fetch1(
            "state_labels", "timestamps"
        )
        nrem_mask = state_labels == 0
        nrem_intervals = _states_to_intervals(state_labels, timestamps, state=0)

        # --- LFP slow oscillation ---
        so_df = (
            lfp_band.LFPBandV1
            & {
                "lfp_merge_id": sel["so_lfp_merge_id"],
                "filter_name": sel["so_filter_name"],
            }
        ).fetch1_dataframe()
        so_signal = so_df.mean(axis=1).values
        so_timestamps = so_df.index.values

        # --- MUA: spike times → rate ---
        spike_df = (
            SpikeSortingOutput & {"merge_id": sel["spike_sorting_merge_id"]}
        ).fetch1_dataframe()
        mua_rate, mua_timestamps = _compute_mua_rate(
            spike_df, so_timestamps, bin_size=params["mua_bin_size"]
        )

        return {
            "params": params,
            "nwb_file_name": nwb_file_name,
            "so_signal": so_signal,
            "so_timestamps": so_timestamps,
            "mua_rate": mua_rate,
            "mua_timestamps": mua_timestamps,
            "nrem_intervals": nrem_intervals,
            "nrem_mask": nrem_mask,
            "sleep_timestamps": timestamps,
        }

    def _compute_states(self, key, fetch_dict):
        """Detect UP/DOWN states within NREM intervals; no DB access."""
        params = fetch_dict["params"]
        so_signal = fetch_dict["so_signal"]
        so_timestamps = fetch_dict["so_timestamps"]
        mua_rate = fetch_dict["mua_rate"]
        nrem_intervals = fetch_dict["nrem_intervals"]

        fs = 1.0 / np.median(np.diff(so_timestamps))

        # Smooth signals
        if params["so_smoothing"] > 0:
            so_signal = gaussian_filter1d(so_signal, sigma=params["so_smoothing"] * fs)
        if params["mua_smoothing"] > 0:
            mua_rate = gaussian_filter1d(mua_rate, sigma=params["mua_smoothing"] * fs)

        # Restrict to NREM only
        nrem_mask = _build_nrem_mask(so_timestamps, nrem_intervals)

        # Detect states
        method = params["method"]
        if method == "threshold":
            states = self._threshold_detection(
                so_signal, mua_rate, nrem_mask, params
            )
        elif method == "gmm":
            states = self._gmm_detection(
                so_signal, mua_rate, nrem_mask, params
            )
        elif method == "hilbert":
            states = self._hilbert_detection(
                so_signal, mua_rate, nrem_mask, params
            )

        # Enforce minimum durations
        states = _enforce_min_duration(
            states,
            min_down=params["min_down_duration"],
            min_up=params["min_up_duration"],
            window_size=np.median(np.diff(so_timestamps)),
        )

        # Summary statistics
        down_intervals = _states_to_intervals(states, so_timestamps, state=0)
        up_intervals = _states_to_intervals(states, so_timestamps, state=1)

        nrem_time = float(np.sum([e - s for s, e in nrem_intervals]))
        down_duration = float(np.sum([e - s for s, e in down_intervals]))
        up_duration = float(np.sum([e - s for s, e in up_intervals]))

        down_durations = [e - s for s, e in down_intervals]
        up_durations = [e - s for s, e in up_intervals]

        return {
            "states": states,
            "timestamps": so_timestamps,
            "down_intervals": down_intervals,
            "up_intervals": up_intervals,
            "down_duration": down_duration,
            "up_duration": up_duration,
            "down_percentage": 100 * down_duration / nrem_time if nrem_time > 0 else 0.0,
            "up_percentage": 100 * up_duration / nrem_time if nrem_time > 0 else 0.0,
            "n_down_states": len(down_intervals),
            "n_up_states": len(up_intervals),
            "mean_down_duration": float(np.mean(down_durations)) if down_durations else 0.0,
            "mean_up_duration": float(np.mean(up_durations)) if up_durations else 0.0,
        }

    def _store_results(self, key, result, nwb_file_name):
        """Write results to NWB and insert DB row."""
        nwb_obj = {
            "state_labels": result["states"],
            "timestamps": result["timestamps"],
            "down_intervals": result["down_intervals"],
            "up_intervals": result["up_intervals"],
        }

        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(nwb_obj)
            analysis_file = builder.analysis_file_name

        self.insert1(
            {
                **key,
                "state_labels": result["states"],
                "timestamps": result["timestamps"],
                "down_duration": result["down_duration"],
                "up_duration": result["up_duration"],
                "down_percentage": result["down_percentage"],
                "up_percentage": result["up_percentage"],
                "n_down_states": result["n_down_states"],
                "n_up_states": result["n_up_states"],
                "mean_down_duration": result["mean_down_duration"],
                "mean_up_duration": result["mean_up_duration"],
                "analysis_file_name": analysis_file,
                "trial_object_id": obj_id,
            }
        )

    # ==================== Detection Methods ====================

    def _threshold_detection(self, so_signal, mua_rate, nrem_mask, params):
        """
        Classify DOWN states as epochs where both LFP amplitude AND MUA rate
        are below their respective percentile thresholds (within NREM only).
        """
        n = len(so_signal)
        states = np.full(n, -1)  # -1 = outside NREM

        nrem_so = so_signal[nrem_mask]
        nrem_mua = mua_rate[nrem_mask]

        lfp_thresh = np.percentile(np.abs(nrem_so), params["lfp_down_percentile"])
        mua_thresh = np.percentile(nrem_mua, params["mua_down_percentile"])

        # Weighted combined score: low = DOWN, high = UP
        lfp_score = np.abs(so_signal) / (lfp_thresh + 1e-10)
        mua_score = mua_rate / (mua_thresh + 1e-10)
        combined = params["lfp_weight"] * lfp_score + params["mua_weight"] * mua_score

        states[nrem_mask] = (combined[nrem_mask] >= 1.0).astype(int)  # 0=DOWN, 1=UP
        return states

    def _gmm_detection(self, so_signal, mua_rate, nrem_mask, params):
        """
        Fit a 2-component GMM to (LFP amplitude, MUA rate) within NREM;
        assign DOWN to the low-activity component.
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        n = len(so_signal)
        states = np.full(n, -1)

        nrem_indices = np.where(nrem_mask)[0]
        x = np.column_stack([
            np.abs(so_signal[nrem_mask]),
            mua_rate[nrem_mask],
        ])
        x_scaled = StandardScaler().fit_transform(x)

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        labels = gmm.fit_predict(x_scaled)

        # DOWN = cluster with lower mean MUA
        mua_means = [np.mean(mua_rate[nrem_mask][labels == i]) for i in (0, 1)]
        down_cluster = np.argmin(mua_means)
        nrem_states = np.where(labels == down_cluster, 0, 1)

        states[nrem_indices] = nrem_states
        return states

    def _hilbert_detection(self, so_signal, mua_rate, nrem_mask, params):
        """
        Use the phase of the slow oscillation (via Hilbert transform) to
        assign UP (ascending/peak phase) and DOWN (trough phase) states.
        DOWN states are near the trough (phase ≈ ±π); UP states near 0.
        MUA is used to confirm: DOWN epochs must also have low MUA.
        """
        from scipy.signal import hilbert

        n = len(so_signal)
        states = np.full(n, -1)

        phase = np.angle(hilbert(so_signal))

        nrem_mua = mua_rate[nrem_mask]
        mua_thresh = np.percentile(nrem_mua, params["mua_down_percentile"])

        # DOWN: near trough (|phase| > π/2) AND low MUA
        near_trough = np.abs(phase) > (np.pi / 2)
        low_mua = mua_rate < mua_thresh

        nrem_states = np.where(near_trough[nrem_mask] & low_mua[nrem_mask], 0, 1)
        states[np.where(nrem_mask)[0]] = nrem_states
        return states

    # ==================== Visualisation ====================

    def plot_up_down_states(self, t_start=None, t_stop=None, figsize=(15, 6)):
        """
        Plot LFP slow oscillation and MUA with UP/DOWN state shading
        for a given time window.
        """
        states = self.fetch1("state_labels")
        timestamps = self.fetch1("timestamps")

        if t_start is None:
            t_start = timestamps[0]
        if t_stop is None:
            t_stop = min(timestamps[0] + 60, timestamps[-1])  # default: first 60s

        win = (timestamps >= t_start) & (timestamps <= t_stop)
        t = timestamps[win]
        s = states[win]

        fig, ax = plt.subplots(figsize=figsize)

        # Shade DOWN states
        in_down = False
        for i, (ts, st) in enumerate(zip(t, s)):
            if st == 0 and not in_down:
                down_start = ts
                in_down = True
            elif st != 0 and in_down:
                ax.axvspan(down_start, t[i - 1], alpha=0.3, color="steelblue", label="DOWN")
                in_down = False
        if in_down:
            ax.axvspan(down_start, t[-1], alpha=0.3, color="steelblue")

        ax.set_xlabel("Time (s)")
        ax.set_title("UP / DOWN States during NREM")

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        n_down = self.fetch1("n_down_states")
        n_up = self.fetch1("n_up_states")
        mean_down = self.fetch1("mean_down_duration")
        mean_up = self.fetch1("mean_up_duration")

        stats_text = (
            f"DOWN: n={n_down}, mean={mean_down:.2f}s\n"
            f"UP:   n={n_up}, mean={mean_up:.2f}s"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        plt.show()
        return fig


# ==================== Module-level helpers ====================

def _compute_mua_rate(spike_df, reference_timestamps, bin_size):
    """
    Bin all spike times across units into a population MUA rate (spikes/s),
    aligned to reference_timestamps.
    """
    all_spike_times = np.concatenate(spike_df["spike_times"].values)
    t_start = reference_timestamps[0]
    t_stop = reference_timestamps[-1]

    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    counts, _ = np.histogram(all_spike_times, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rate = counts / bin_size  # spikes/s

    # Interpolate onto the LFP timestamps
    mua_interp = np.interp(reference_timestamps, bin_centers, rate)
    return mua_interp, reference_timestamps


def _build_nrem_mask(timestamps, nrem_intervals):
    """Return a boolean mask over timestamps that are within any NREM interval."""
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, stop in nrem_intervals:
        mask |= (timestamps >= start) & (timestamps <= stop)
    return mask


def _states_to_intervals(states, timestamps, state):
    """Convert a state label array to [[start, stop], ...] intervals."""
    intervals = []
    in_state = False
    start_time = None

    for i, (s, t) in enumerate(zip(states, timestamps)):
        if s == state and not in_state:
            start_time = t
            in_state = True
        elif s != state and in_state:
            intervals.append([start_time, timestamps[i - 1]])
            in_state = False

    if in_state:
        intervals.append([start_time, timestamps[-1]])

    return np.array(intervals) if intervals else np.array([]).reshape(0, 2)


def _enforce_min_duration(states, min_down, min_up, window_size):
    """Remove bouts shorter than the minimum duration for each state."""
    smoothed = states.copy()

    i = 0
    while i < len(smoothed):
        current = smoothed[i]
        if current == -1:  # outside NREM — skip
            i += 1
            continue
        j = i
        while j < len(smoothed) and smoothed[j] == current:
            j += 1

        duration = (j - i) * window_size
        min_dur = min_down if current == 0 else min_up
        if duration < min_dur and i > 0 and smoothed[i - 1] != -1:
            smoothed[i:j] = smoothed[i - 1]

        i = j

    return smoothed