"""
Spyglass Sleep Scoring Pipeline

These Spyglass tables are for automated sleep state classification (NREM, REM, WAKE).
It uses unsupervised ML methods (GMM, K-means) for state classification.

Author: DMR
Date: March 2026
"""

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import spyglass.lfp as lfp
from spyglass.common import IntervalList
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.lfp.analysis.v1 import lfp_band
from spyglass.position.position_merge import PositionOutput
from spyglass.utils import SpyglassMixin
from sleep.pss_dmr import PSSParams, PSSSelection, SleepPSS

schema = dj.schema("denissemorales_sleepscoring")
VALID_METHODS = {"gmm", "kmeans", "hierarchical"}


@schema
class SleepScoringParams(SpyglassMixin, dj.Lookup):
    definition = """
    sleep_scoring_params_name: varchar(64)
    ---
    # Classification parameters
    method: varchar(32)              # Classification method (gmm | kmeans | hierarchical)
    use_hierarchical: bool           # Use two-stage classification
    use_pss: bool                    # Include power spectrum slope if available

    # Smoothing parameters
    power_smoothing: float           # Gaussian smoothing sigma for power (seconds)
    speed_smoothing: float           # Gaussian smoothing sigma for speed (seconds)

    # Constraint parameters
    apply_constraints: bool          # Apply physiological constraints
    rem_cannot_follow_wake: bool     # REM cannot directly follow WAKE
    constraint_max_iterations: int    # Max iterations for constraint enforcement

    # State duration parameters
    min_duration: float              # Minimum state bout duration (seconds)

    # Wake detection parameters
    speed_threshold: float           # Speed threshold for wake detection (cm/s)
    use_speed_for_wake: bool         # Use speed instead of EMG for wake detection
    """

    contents = [
        {
            "sleep_scoring_params_name": "hierarchical",
            "method": "hierarchical",
            "use_hierarchical": True,
            "use_pss": False,            "power_smoothing": 0.5,
            "speed_smoothing": 0.5,
            "apply_constraints": True,
            "rem_cannot_follow_wake": True,
            "constraint_max_iterations": 15,
            "min_duration": 5.0,
            "speed_threshold": 3.0,
            "use_speed_for_wake": True,
        }
    ]

    def insert1(self, row, **kwargs):
        method = row.get("method")
        if method not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {VALID_METHODS}"
            )
        super().insert1(row, **kwargs)


@schema
class SleepScoringSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SleepScoringParams
    target_interval_list_name: varchar(64)
    nwb_file_name: varchar(64)
    theta_filter_name='': varchar(64)
    delta_filter_name='': varchar(64)
    lfp_merge_id: uuid               # Shared LFP merge ID (theta and delta use different filter_names on the same merge)
    ---
    emg_merge_id=null: uuid          # Optional EMG band power merge ID
    pss_merge_id=null: uuid          # Optional PSS merge ID

    # Position merge
    pos_merge_id: uuid               # PositionOutput merge ID

    # Sampling rate for power features
    filter_sampling_rate: float

    emg_filter_name='': varchar(64)
    pss_filter_name='': varchar(64)
    """


@schema
class SleepScoring(SpyglassMixin, dj.Computed):
    definition = """
    -> SleepScoringSelection
    ---
    state_labels: longblob               # Array of state labels (0=NREM, 1=REM, 2=WAKE)
    timestamps: longblob                 # Timestamps for state labels
    nrem_duration: float                 # Total NREM duration (seconds)
    rem_duration: float                  # Total REM duration (seconds)
    wake_duration: float                 # Total WAKE duration (seconds)
    nrem_percentage: float               # Percentage of time in NREM
    rem_percentage: float                # Percentage of time in REM
    wake_percentage: float               # Percentage of time in WAKE
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make(self, key):
        fetch_dict = self._fetch_data(key)
        result = self._compute_states(key, fetch_dict)
        self._store_results(key, result, fetch_dict["nwb_file_name"])

    # ==================== Tri-part helpers ====================

    def _fetch_data(self, key):
        """Fetch all upstream data needed for classification."""
        params = (SleepScoringParams & key).fetch1()
        sel = (SleepScoringSelection & key).fetch1()
        nwb_file_name = sel["nwb_file_name"]

        # Theta power
        theta_df = (
            lfp_band.LFPBandV1
            & {
                "lfp_merge_id": sel["lfp_merge_id"],
                "filter_name": sel["theta_filter_name"],
            }
        ).fetch1_dataframe()
        theta_power = theta_df.mean(axis=1).values
        theta_timestamps = theta_df.index.values

        # Delta power
        delta_df = (
            lfp_band.LFPBandV1
            & {
                "lfp_merge_id": sel["lfp_merge_id"],
                "filter_name": sel["delta_filter_name"],
            }
        ).fetch1_dataframe()
        delta_power = delta_df.mean(axis=1).values

        # Head speed
        head_speed = None
        if sel.get("pos_merge_id") is not None:
            pos_df = (PositionOutput & {"merge_id": sel["pos_merge_id"]}).fetch1_dataframe()
            head_speed = np.interp(
                theta_timestamps, pos_df.index.values, pos_df["speed"].values
            )

        # EMG (optional)
        emg_power = None
        if sel.get("emg_merge_id") is not None and sel.get("emg_filter_name"):
            emg_df = (
                lfp.LFPOutput
                & {"merge_id": sel["emg_merge_id"]}
            ).fetch1_dataframe()
            print(f"Fetched EMG dataframe with shape {emg_df.shape}")
            emg_timestamps, emg_corr = self.emg_from_lfp_corr(emg_df, output_fs=2.0)
            emg_power = np.interp(theta_timestamps, emg_timestamps, emg_corr)

        # PSS (optional)
        pss_data = None
        if params["use_pss"]:
            pss_timestamps = (
                SleepPSS
                & {
                    "nwb_file_name": sel["nwb_file_name"],
                    "lfp_merge_id": sel["lfp_merge_id"],
                }
            ).fetch("pss_timestamps")[0]
            pss_values = (
                SleepPSS
                & {
                    "nwb_file_name": sel["nwb_file_name"],
                    "lfp_merge_id": sel["lfp_merge_id"],
                }
            ).fetch("pss_values")[0]
            print(pss_timestamps.shape, pss_values.shape, theta_timestamps.shape)
            pss_data = np.interp(theta_timestamps, pss_timestamps, pss_values)

        return {
            "params": params,
            "nwb_file_name": nwb_file_name,
            "theta_power": theta_power,
            "delta_power": delta_power,
            "theta_timestamps": theta_timestamps,
            "head_speed": head_speed,
            "emg_power": emg_power,
            "pss_data": pss_data,
        }

    def _compute_states(self, key, fetch_dict):
        """Run the full classification pipeline; no DB access here."""
        params = fetch_dict["params"]
        theta_timestamps = fetch_dict["theta_timestamps"]

        features = self._prepare_features(
            theta_power=fetch_dict["theta_power"],
            delta_power=fetch_dict["delta_power"],
            emg_data=fetch_dict["emg_power"],
            headspeed_data=fetch_dict["head_speed"],
            pss_data=fetch_dict["pss_data"],
            timestamps=theta_timestamps,
            params=params,
        )

        states = self._hierarchical_classification(features, params)

        if params["apply_constraints"]:
            states = self._apply_constraints(states, params)

        states = self._smooth_states(
            states,
            min_duration=params["min_duration"],
            window_size=np.median(np.diff(theta_timestamps)),
        )

        # REM fallback
        rem_mask = states == 1
        if np.mean(rem_mask) < 0.02 and np.any(states != 2):
            sleep_mask = states != 2
            theta = features["theta_power"]
            delta = features["delta_power"]
            rem_mask = (
                sleep_mask
                & (theta > np.percentile(theta[sleep_mask], params.get("rem_percentile", 70)))
                & (delta < np.percentile(delta[sleep_mask], 30))
            )
            states[rem_mask] = 1
            print(f"REM fallback applied: {np.sum(rem_mask)} epochs set to REM")

        # Derive intervals & durations
        nrem_intervals = self._states_to_intervals(states, theta_timestamps, state=0)
        rem_intervals = self._states_to_intervals(states, theta_timestamps, state=1)
        wake_intervals = self._states_to_intervals(states, theta_timestamps, state=2)

        total_time = theta_timestamps[-1] - theta_timestamps[0]
        nrem_duration = float(np.sum([e - s for s, e in nrem_intervals]))
        rem_duration = float(np.sum([e - s for s, e in rem_intervals]))
        wake_duration = float(np.sum([e - s for s, e in wake_intervals]))

        return {
            "states": states,
            "timestamps": theta_timestamps,
            "intervals": {"nrem": nrem_intervals, "rem": rem_intervals, "wake": wake_intervals},
            "nrem_duration": nrem_duration,
            "rem_duration": rem_duration,
            "wake_duration": wake_duration,
            "nrem_percentage": 100 * nrem_duration / total_time,
            "rem_percentage": 100 * rem_duration / total_time,
            "wake_percentage": 100 * wake_duration / total_time,
            "params_name": key["sleep_scoring_params_name"],
        }

    def _store_results(self, key, result, nwb_file_name):
        """Write results to the NWB file and insert the DB row.

        Uses the context manager pattern so AnalysisNwbfile registration
        (and the FK ref) is handled automatically. Pass arrays directly
        rather than a dict since add_nwb_object requires an np.ndarray.
        """
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(
                result["states"].astype(np.int32), table_name="state_labels"
            )
            builder.add_nwb_object(result["timestamps"], table_name="timestamps")
            analysis_file_name = builder.analysis_file_name

        self.insert1(
            {
                **key,
                "state_labels": result["states"],
                "timestamps": result["timestamps"],
                "nrem_duration": result["nrem_duration"],
                "rem_duration": result["rem_duration"],
                "wake_duration": result["wake_duration"],
                "nrem_percentage": result["nrem_percentage"],
                "rem_percentage": result["rem_percentage"],
                "wake_percentage": result["wake_percentage"],
                "analysis_file_name": analysis_file_name,
                "trial_object_id": obj_id,
            }
        )

    # ==================== Feature Preparation ====================

    @staticmethod
    def emg_from_lfp_corr(
        lfp_df: pd.DataFrame,
        output_fs: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute pseudo-EMG from pre-filtered LFP as summed pairwise zero-lag
        correlations, following Swanson et al. 2025 / Watson et al. 2016 /
        Schomburg et al. 2014.

        Assumes input has ALREADY been:
        - Bandpass filtered at 300–600 Hz
        - Restricted to non-neighboring channels (>200 µm separation)

        Args:
            lfp_df:    DataFrame of shape (n_samples, n_channels),
                    index = timestamps in seconds.
                    Expected input fs ~2000 Hz from Spyglass.
            output_fs: Desired output rate in Hz (default 2 Hz).

        Returns:
            out_timestamps: 1-D array at output_fs.
            emg_corr:       1-D array of summed pairwise correlations.
        """
        if not isinstance(lfp_df, pd.DataFrame):
            lfp_df = pd.DataFrame(lfp_df)

        timestamps = lfp_df.index.to_numpy(dtype=float)
        x = lfp_df.to_numpy(dtype=float)  # (n_samples, n_channels)
        n_samples, n_channels = x.shape

        if n_samples < 2:
            raise ValueError("Need at least 2 samples.")
        if n_channels < 2:
            raise ValueError("Need at least 2 channels to compute pairwise EMG.")

        fs = 1.0 / float(np.median(np.diff(timestamps)))

        # Window sizing matches original MATLAB logic:
        #   binScootSamps      = round(Fs / output_fs)   -> step between frames
        #   xcorr_window_samps = round(Fs / output_fs)   -> half-width of window
        step_samps = int(round(fs / output_fs))
        half_win = step_samps

        center_inds = np.arange(half_win, n_samples - half_win, step_samps)
        if len(center_inds) == 0:
            raise ValueError(
                f"Signal too short for output_fs={output_fs} Hz "
                f"(need at least {2 * half_win + 1} samples, got {n_samples})."
            )

        win_offsets = np.arange(-half_win, half_win + 1)  # (2*half_win+1,)

        # All pairs — channel selection already done upstream
        pairs = [(j, k) for j in range(n_channels) for k in range(j + 1, n_channels)]

        # Summed zero-lag Pearson correlation across all pairs
        emg_corr = np.zeros(len(center_inds), dtype=float)

        for j, k in pairs:
            pair_trace = np.empty(len(center_inds), dtype=float)
            for idx, ci in enumerate(center_inds):
                seg_j = x[ci + win_offsets, j]
                seg_k = x[ci + win_offsets, k]
                std_j = np.std(seg_j)
                std_k = np.std(seg_k)
                if std_j == 0.0 or std_k == 0.0:
                    pair_trace[idx] = np.nan
                else:
                    pair_trace[idx] = np.corrcoef(seg_j, seg_k)[0, 1]

            emg_corr += np.nan_to_num(pair_trace, nan=0.0)

        out_timestamps = timestamps[center_inds]
        return out_timestamps, emg_corr

    def _prepare_features(
        self,
        theta_power,
        delta_power,
        emg_data,
        headspeed_data,
        pss_data,
        timestamps,
        params,
    ):
        """Prepare and smooth features for sleep scoring."""
        n_samples = len(theta_power)
        fs = 1.0 / np.median(np.diff(timestamps))

        if params.get("power_smoothing", 0) > 0:
            sigma = params["power_smoothing"] * fs
            delta_power = gaussian_filter1d(delta_power, sigma=sigma)
            theta_power = gaussian_filter1d(theta_power, sigma=sigma)
            if emg_data is not None:
                emg_data = gaussian_filter1d(emg_data, sigma=sigma)
            if pss_data is not None:
                pss_data = gaussian_filter1d(pss_data, sigma=sigma)

        if headspeed_data is not None and params.get("speed_smoothing", 0) > 0:
            sigma = params["speed_smoothing"] * fs
            headspeed_data = gaussian_filter1d(headspeed_data, sigma=sigma)

        if headspeed_data is not None:
            speed_threshold = params.get("speed_threshold", 5.0)
            speed_wake = (headspeed_data > speed_threshold).astype(int)
        else:
            headspeed_data = np.zeros(n_samples)
            speed_wake = np.zeros(n_samples, dtype=int)

        return {
            "time": timestamps,
            "delta_power": delta_power,
            "theta_power": theta_power,
            "delta_theta_ratio": delta_power / (theta_power + 1e-10),
            "emg_power": emg_data,
            "headspeed": headspeed_data,
            "speed_wake": speed_wake,
            "pss": pss_data if pss_data is not None else np.zeros(n_samples),
        }

    # ==================== Classification Methods ====================

    def _score_states(self, features, params):
        """Single-stage classification using GMM or K-means."""
        feature_list = [
            np.log(features["delta_power"] + 1e-10),
            np.log(features["theta_power"] + 1e-10),
        ]

        if features.get("emg_power") is not None:
            feature_list.append(np.log(features["emg_power"] + 1e-10))

        if params["use_pss"] and np.any(features["pss"] != 0):
            feature_list.append(features["pss"])

        feature_matrix = np.column_stack(feature_list)
        feature_matrix_scaled = StandardScaler().fit_transform(feature_matrix)

        if params["method"] == "gmm":
            model = GaussianMixture(n_components=3, random_state=42, covariance_type="full")
        else:
            model = KMeans(n_clusters=3, random_state=42, n_init=10)

        cluster_labels = model.fit_predict(feature_matrix_scaled)
        return self._map_clusters_to_states(features, cluster_labels)

    def _hierarchical_classification(self, features, params):
        """
        Two-stage sleep scoring.

        Stage 1: Wake vs Sleep (EMG or head speed).
        Stage 2: NREM vs REM within sleep (delta/theta ratio).
        """
        n = len(features["time"])
        states = np.full(n, 2)  # default all WAKE

        use_speed = params.get("use_speed_for_wake", True)

        emg_available = (
            features.get("emg_power") is not None
            and not np.allclose(features["emg_power"], 0)
        )

        if emg_available:
            print("Wake detection using EMG")
            emg_log = np.log(features["emg_power"] + 1e-10)
            emg_z = (emg_log - np.mean(emg_log)) / np.std(emg_log)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            emg_labels = kmeans.fit_predict(emg_z.reshape(-1, 1))

            if np.mean(emg_z[emg_labels == 0]) > np.mean(emg_z[emg_labels == 1]):
                emg_labels = 1 - emg_labels

            wake_mask = emg_labels.astype(bool)

        elif use_speed and features.get("speed_wake") is not None:
            wake_mask = features["speed_wake"].astype(bool)
            print(f"Wake detection using head speed: {np.sum(wake_mask)} epochs WAKE")

        else:
            wake_mask = np.zeros(n, dtype=bool)
            print("No EMG or head speed: assuming all sleep")

        sleep_mask = ~wake_mask
        states[wake_mask] = 2

        # Stage 2: NREM vs REM
        sleep_indices = np.where(sleep_mask)[0]
        min_sleep_epochs = params.get("min_sleep_epochs", 20)

        if len(sleep_indices) >= min_sleep_epochs:
            dt_ratio = features["delta_theta_ratio"][sleep_mask]
            valid = np.isfinite(dt_ratio)
            sleep_indices_valid = sleep_indices[valid]
            dt_valid = dt_ratio[valid]

            if len(dt_valid) >= min_sleep_epochs:
                if params.get("sleep_classification_method", "kmeans") == "kmeans":
                    x_scaled = StandardScaler().fit_transform(dt_valid.reshape(-1, 1))
                    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(x_scaled)
                    dt_means = [np.mean(dt_valid[labels == i]) for i in (0, 1)]
                    nrem_cluster = np.argmax(dt_means)
                    sleep_states = np.where(labels == nrem_cluster, 0, 1)
                else:
                    threshold = np.median(dt_valid)
                    sleep_states = np.where(dt_valid >= threshold, 0, 1)

                states[sleep_indices_valid] = sleep_states
                print(
                    f"NREM epochs: {np.sum(sleep_states == 0)}, "
                    f"REM epochs: {np.sum(sleep_states == 1)}"
                )
            else:
                print("Not enough valid sleep epochs for NREM/REM classification")
        else:
            print("Not enough sleep epochs to classify NREM/REM")

        # REM fallback
        if np.mean(states == 1) < 0.02 and np.any(sleep_mask):
            theta = features["theta_power"]
            delta = features["delta_power"]
            rem_mask = (
                sleep_mask
                & (theta > np.percentile(theta[sleep_mask], params.get("rem_percentile", 70)))
                & (delta < np.percentile(delta[sleep_mask], 30))
            )
            states[rem_mask] = 1
            print(f"REM fallback applied: {np.sum(rem_mask)} epochs set to REM")

        return states

    def _map_clusters_to_states(self, features, cluster_labels):
        """Map cluster labels to physiological states."""
        n_clusters = len(np.unique(cluster_labels))
        cluster_means = {
            c: {
                "delta": np.mean(features["delta_power"][cluster_labels == c]),
                "theta": np.mean(features["theta_power"][cluster_labels == c]),
                "emg": np.mean(features["emg_power"][cluster_labels == c])
                if features.get("emg_power") is not None
                else 0.0,
                "dt_ratio": np.mean(features["delta_theta_ratio"][cluster_labels == c]),
            }
            for c in range(n_clusters)
        }

        nrem_cluster = np.argmax(
            [cluster_means[c]["dt_ratio"] - cluster_means[c]["emg"] for c in range(n_clusters)]
        )
        state_mapping = {nrem_cluster: 0}

        remaining = [c for c in range(n_clusters) if c != nrem_cluster]
        if len(remaining) > 1:
            wake_cluster = remaining[np.argmax([cluster_means[c]["emg"] for c in remaining])]
            state_mapping[wake_cluster] = 2
            rem_cluster = [c for c in remaining if c != wake_cluster][0]
            state_mapping[rem_cluster] = 1
        else:
            r = remaining[0]
            state_mapping[r] = 2 if cluster_means[r]["emg"] > cluster_means[nrem_cluster]["emg"] else 1

        return np.array([state_mapping.get(label, 0) for label in cluster_labels])

    # ==================== Post-processing ====================

    def _apply_constraints(self, states, params):
        """Apply physiological constraints iteratively."""
        constrained = states.copy()
        iteration = 0
        changes_made = True

        while changes_made and iteration < params["constraint_max_iterations"]:
            changes_made = False
            iteration += 1
            prev_states = np.concatenate([[constrained[0]], constrained[:-1]])

            if params["rem_cannot_follow_wake"]:
                invalid_rem = (constrained == 1) & (prev_states == 2)
                if np.any(invalid_rem):
                    constrained[invalid_rem] = 2
                    changes_made = True

        return constrained

    def _smooth_states(self, states, min_duration, window_size):
        """Remove short bouts and smooth single-point transitions."""
        smoothed = states.copy()
        min_windows = int(min_duration / window_size)

        # Remove single-point transitions
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] != smoothed[i - 1] and smoothed[i] != smoothed[i + 1]:
                smoothed[i] = smoothed[i - 1]

        # Enforce minimum bout duration
        i = 0
        while i < len(smoothed):
            current_state = smoothed[i]
            j = i
            while j < len(smoothed) and smoothed[j] == current_state:
                j += 1
            if (j - i) < min_windows and i > 0:
                smoothed[i:j] = smoothed[i - 1]
            i = j

        return smoothed

    def _states_to_intervals(self, states, timestamps, state):
        """Convert state label array to [[start, stop], ...] intervals."""
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

    # ==================== Fetch / Interval helpers ====================

    def _fetch_interval(self, interval_list_name_field):
        """Shared helper: fetch valid_times for a named interval list."""
        interval_name = self.fetch1(interval_list_name_field)
        key = {
            "nwb_file_name": self.fetch1("nwb_file_name"),
            "interval_list_name": interval_name,
        }
        return (IntervalList & key).fetch_interval()

    def fetch_nrem_times(self):
        """Fetch NREM intervals."""
        return self._fetch_interval("nrem_interval_list_name")

    def fetch_rem_times(self):
        """Fetch REM intervals."""
        return self._fetch_interval("rem_interval_list_name")

    def fetch_wake_times(self):
        """Fetch WAKE intervals."""
        return self._fetch_interval("wake_interval_list_name")

    # ==================== Visualisation ====================

    def plot_hypnogram(self, figsize=(15, 8)):
        """Plot sleep state hypnogram with summary statistics."""
        states = self.fetch1("state_labels")
        timestamps = self.fetch1("timestamps")
        time_hours = (timestamps - timestamps[0]) / 3600

        state_colors = {0: "blue", 1: "red", 2: "green"}
        colors = [state_colors[s] for s in states]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(time_hours, states, c=colors, s=1, alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Sleep State")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["NREM", "REM", "WAKE"])
        ax.set_title("Sleep State Hypnogram")
        ax.grid(True, alpha=0.3)

        nrem_pct, rem_pct, wake_pct = self.fetch1(
            "nrem_percentage", "rem_percentage", "wake_percentage"
        )
        stats_text = (
            f"NREM: {nrem_pct:.1f}%\nREM: {rem_pct:.1f}%\nWAKE: {wake_pct:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        plt.show()
        return fig
