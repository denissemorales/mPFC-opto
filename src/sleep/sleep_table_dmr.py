"""
Spyglass Sleep Scoring Pipeline

These Spyglass tables are for automated sleep state classification (NREM, REM, WAKE).
It uses unsupervised ML methods (GMM, K-means) for state classification.

Author: DMR+Claude
Date: Nov 2025
"""

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
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

schema = dj.schema("denissemorales_sleepscoring")


@schema
class SleepScoringParams(SpyglassMixin, dj.Manual):
    definition = """
    sleep_scoring_params_name: varchar(64)
    ---
    # Classification parameters
    method: enum('gmm', 'kmeans', 'hierarchical')  # Classification method
    use_hierarchical: bool  # Use two-stage classification
    use_pss: bool  # Include power spectrum slope if available

    # Smoothing parameters
    power_smoothing: float  # Gaussian smoothing sigma for power (seconds)
    speed_smoothing: float  # Gaussian smoothing sigma for speed (seconds)

    # Constraint parameters
    apply_constraints: bool  # Apply physiological constraints
    rem_cannot_follow_wake: bool  # REM cannot directly follow WAKE
    constraint_max_iterations: int  # Max iterations for constraint enforcement

    # State duration parameters
    min_duration: float  # Minimum state bout duration (seconds)

    # Wake detection parameters
    speed_threshold: float  # Speed threshold for wake detection (cm/s)
    use_speed_for_wake: bool  # Use speed instead of EMG for wake detection
    """

    @classmethod
    def insert_default(cls):
        """Insert default parameter sets"""
        default_params = [
            {
                "sleep_scoring_params_name": "hierarchical",
                "method": "hierarchical",
                "use_hierarchical": True,
                "use_pss": False,
                "power_smoothing": 0.5,
                "speed_smoothing": 0.5,
                "apply_constraints": True,
                "rem_cannot_follow_wake": True,
                "constraint_max_iterations": 15,
                "min_duration": 5.0,
                "speed_threshold": 3.0,
                "use_speed_for_wake": True,
            }
        ]

        cls.insert(default_params, skip_duplicates=True)


@schema
class SleepScoringSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SleepScoringParams
    target_interval_list_name: varchar(64)
    nwb_file_name: varchar(64)
    theta_filter_name='': varchar(64)
    delta_filter_name='': varchar(64)
    lfp_merge_id: uuid                # Base LFP merge (optional, if needed)
    ---

    emg_merge_id=null: uuid          # Optional EMG band power merge ID
    pss_merge_id=null: uuid           # Optional PSS merge ID

    # Position merge
    pos_merge_id: uuid                # PositionOutput merge ID

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
    state_labels: longblob  # Array of state labels (0=NREM, 1=REM, 2=WAKE)
    timestamps: longblob  # Timestamps for state labels
    nrem_duration: float  # Total NREM duration (seconds)
    rem_duration: float  # Total REM duration (seconds)
    wake_duration: float  # Total WAKE duration (seconds)
    nrem_percentage: float  # Percentage of time in NREM
    rem_percentage: float  # Percentage of time in REM
    wake_percentage: float  # Percentage of time in WAKE
        -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make(self, key):
        """Compute sleep states using ML classification (robust, EMG optional)"""

        # ----------------------------
        # 1. Load parameters & data
        # ----------------------------
        params = (SleepScoringParams & key).fetch1()
        selection_row = (SleepScoringSelection & key).fetch1()
        nwb_file_name = selection_row["nwb_file_name"]
        interval_list_name = selection_row["target_interval_list_name"]

        # --- Theta power ---
        lfp_merge_id = selection_row["lfp_merge_id"]
        theta_power_df = (
            lfp_band.LFPBandV1
            & {
                "lfp_merge_id": lfp_merge_id,
                "filter_name": selection_row["theta_filter_name"],
            }
        ).fetch1_dataframe()
        theta_power = theta_power_df.mean(axis=1).values
        theta_timestamps = theta_power_df.index.values

        # --- Delta power ---
        delta_power_df = (
            lfp_band.LFPBandV1
            & {
                "lfp_merge_id": lfp_merge_id,
                "filter_name": selection_row["delta_filter_name"],
            }
        ).fetch1_dataframe()
        delta_power = delta_power_df.mean(axis=1).values

        # --- Head speed / position ---
        head_speed = None
        if selection_row.get("pos_merge_id"):
            pos_df = (
                PositionOutput & {"merge_id": selection_row["pos_merge_id"]}
            ).fetch1_dataframe()
            head_speed = np.interp(
                theta_timestamps, pos_df.index.values, pos_df["speed"].values
            )

        # --- EMG ---
        emg_power = None
        if selection_row.get("emg_filter_name"):
            emg_df = (
                lfp_band.LFPBandV1
                & {
                    "lfp_merge_id": selection_row["emg_merge_id"],
                    "filter_name": selection_row["emg_filter_name"],
                }
            ).fetch1_dataframe()
            emg_power = emg_df.mean(axis=1).values

        # --- PSS (optional) ---
        pss_data = None
        if selection_row.get("pss_filter_name"):
            pss_df = (
                lfp.LFPOutput.LFPV1()
                & {
                    "lfp_merge_id": selection_row["pss_merge_id"],
                    "filter_name": selection_row["pss_filter_name"],
                }
            ).fetch1_dataframe()
            pss_data = pss_df.mean(axis=1).values

        # ----------------------------
        # 2. Prepare features
        # ----------------------------
        features = self._prepare_features(
            theta_power=theta_power,
            delta_power=delta_power,
            emg_data=emg_power,
            headspeed_data=head_speed,
            pss_data=pss_data,
            timestamps=theta_timestamps,
            params=params,
        )

        # ----------------------------
        # 3. Hierarchical classification
        # ----------------------------
        states = self._hierarchical_classification(features, params)

        # ----------------------------
        # 4. Apply physiological constraints
        # ----------------------------
        if params["apply_constraints"]:
            states = self._apply_constraints(states, params)

        # ----------------------------
        # 5. Smooth states (only NREM/WAKE)
        # ----------------------------
        states = self._smooth_states(
            states,
            min_duration=params["min_duration"],
            window_size=np.median(np.diff(theta_timestamps)),
        )

        # ----------------------------
        # 6. REM fallback (after smoothing)
        # ----------------------------
        rem_mask = states == 1
        if np.mean(rem_mask) < 0.02 and np.any(states != 2):
            dt_ratio = features["delta_theta_ratio"]
            theta = features["theta_power"]
            delta = features["delta_power"]
            sleep_mask = states != 2

            rem_mask = (
                sleep_mask
                & (
                    theta
                    > np.percentile(
                        theta[sleep_mask], params.get("rem_percentile", 70)
                    )
                )
                & (delta < np.percentile(delta[sleep_mask], 30))
            )
            states[rem_mask] = 1
            print(f"REM fallback applied: {np.sum(rem_mask)} epochs set to REM")

        # ----------------------------
        # 7. Convert states to intervals
        # ----------------------------
        nrem_intervals = self._states_to_intervals(
            states, theta_timestamps, state=0
        )
        rem_intervals = self._states_to_intervals(
            states, theta_timestamps, state=1
        )
        wake_intervals = self._states_to_intervals(
            states, theta_timestamps, state=2
        )

        # ----------------------------
        # 8. Calculate durations & percentages
        # ----------------------------
        total_time = theta_timestamps[-1] - theta_timestamps[0]
        nrem_duration = np.sum([end - start for start, end in nrem_intervals])
        rem_duration = np.sum([end - start for start, end in rem_intervals])
        wake_duration = np.sum([end - start for start, end in wake_intervals])

        # ----------------------------
        # 9. Store results
        # ----------------------------
        sleep_intervals = {
            "nrem": nrem_intervals,
            "rem": rem_intervals,
            "wake": wake_intervals,
        }

        sleep_results = {
            "intervals": sleep_intervals,
            "state_labels": states,
            "timestamps": theta_timestamps,
            "durations": {
                "nrem": nrem_duration,
                "rem": rem_duration,
                "wake": wake_duration,
            },
            "percentages": {
                "nrem": 100 * nrem_duration / total_time,
                "rem": 100 * rem_duration / total_time,
                "wake": 100 * wake_duration / total_time,
            },
            "params_name": key["sleep_scoring_params_name"] }


        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(sleep_results)
            analysis_file = builder.analysis_file_name

        # ----------------------------
        # 10. Insert results
        # ----------------------------
        self.insert1(
            {
                **key,
                "state_labels": states,
                "timestamps": theta_timestamps,
                "nrem_duration": nrem_duration,
                "rem_duration": rem_duration,
                "wake_duration": wake_duration,
                "nrem_percentage": 100 * nrem_duration / total_time,
                "rem_percentage": 100 * rem_duration / total_time,
                "wake_percentage": 100 * wake_duration / total_time,
                "analysis_file_name": analysis_file,
                "trial_object_id": obj_id
            }
        )

    # ==================== Feature Preparation ====================
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
        """
        Prepare and smooth features for sleep scoring.

        Ensures:
        - delta/theta ratio
        - EMG (optional)
        - head speed and speed_wake
        - PSS (optional)
        """
        n_samples = len(theta_power)

        # --- Sampling rate ---
        fs = 1.0 / np.median(np.diff(timestamps))

        # --- Smooth power signals ---
        if params.get("power_smoothing", 0) > 0:
            sigma_samples = params["power_smoothing"] * fs
            delta_power = gaussian_filter1d(delta_power, sigma=sigma_samples)
            theta_power = gaussian_filter1d(theta_power, sigma=sigma_samples)
            if emg_data is not None:
                emg_data = gaussian_filter1d(emg_data, sigma=sigma_samples)
            if pss_data is not None:
                pss_data = gaussian_filter1d(pss_data, sigma=sigma_samples)

        # --- Smooth head speed ---
        if headspeed_data is not None and params.get("speed_smoothing", 0) > 0:
            sigma_samples = params["speed_smoothing"] * fs
            headspeed_data = gaussian_filter1d(
                headspeed_data, sigma=sigma_samples
            )

        # --- Compute wake from speed ---
        if headspeed_data is not None:
            speed_threshold = params.get(
                "speed_threshold", 5.0
            )  # example threshold
            speed_wake = (headspeed_data > speed_threshold).astype(int)
        else:
            # No head speed → default all zeros
            headspeed_data = np.zeros(n_samples)
            speed_wake = np.zeros(n_samples, dtype=int)

        # --- Build feature dictionary ---
        features = {
            "time": timestamps,
            "delta_power": delta_power,
            "theta_power": theta_power,
            "delta_theta_ratio": delta_power / (theta_power + 1e-10),
            "emg_power": emg_data,  # can be None
            "headspeed": headspeed_data,
            "speed_wake": speed_wake,  # always available
            "pss": pss_data if pss_data is not None else np.zeros(n_samples),
        }

        return features

    # ==================== Classification Methods ====================

    def _score_states(self, features, params):
        """Single-stage classification using GMM or K-means"""

        # Prepare feature matrix (log-transformed power)
        feature_list = [
            np.log(features["delta_power"] + 1e-10),
            np.log(features["theta_power"] + 1e-10),
            np.log(features["emg_power"] + 1e-10),
        ]

        # Add PSS if available and requested
        has_pss = np.any(features["pss"] != 0)
        if params["use_pss"] and has_pss:
            feature_list.append(features["pss"])

        feature_matrix = np.column_stack(feature_list)

        # Standardize
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Cluster
        if params["method"] == "gmm":
            model = GaussianMixture(
                n_components=3, random_state=42, covariance_type="full"
            )
            cluster_labels = model.fit_predict(feature_matrix_scaled)
        else:  # kmeans
            model = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(feature_matrix_scaled)

        # Map clusters to states
        states = self._map_clusters_to_states(features, cluster_labels)

        return states

    def _hierarchical_classification(self, features, params):
        """
        Robust two-stage sleep scoring.

        Stage 1: Wake vs Sleep
            - Optionally uses EMG or head speed.
            - WAKE = 2, SLEEP = 0/1

        Stage 2: NREM vs REM (within sleep)
            - Uses delta/theta ratio and KMeans.
            - NREM = 0, REM = 1
        """
        n = len(features["time"])
        states = np.full(n, 2)  # default all WAKE

        # --- Stage 1: Wake vs Sleep ---
        use_emg = params.get("use_emg_for_wake", False)
        use_speed = params.get("use_speed_for_wake", True)

        # --- Check EMG availability ---
        emg_available = (
            features.get("emg_power") is not None
            and not np.allclose(features["emg_power"], 0)
            and use_emg
        )

        if emg_available:
            emg_data = np.log(features["emg_power"] + 1e-10)
            emg_z = (emg_data - np.mean(emg_data)) / np.std(emg_data)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            emg_labels = kmeans.fit_predict(emg_z.reshape(-1, 1))

            # Ensure 0 = sleep, 1 = wake
            if np.mean(emg_z[emg_labels == 0]) > np.mean(
                emg_z[emg_labels == 1]
            ):
                emg_labels = 1 - emg_labels

            wake_mask = emg_labels.astype(bool)
            print(f"Wake detection using EMG: {np.sum(wake_mask)} epochs WAKE")

        else:
            # --- Check head speed ---
            if (
                "speed_wake" in features
                and features["speed_wake"] is not None
                and use_speed
            ):
                wake_mask = features["speed_wake"].astype(bool)
                print(
                    f"Wake detection using head speed: {np.sum(wake_mask)} epochs WAKE"
                )
            else:
                # --- No EMG, no speed → assume all sleep ---
                wake_mask = np.zeros(n, dtype=bool)
                print("No EMG or head speed: assuming all sleep")

        sleep_mask = ~wake_mask
        states[wake_mask] = 2  # WAKE

        # --- Stage 2: NREM vs REM (within sleep) ---
        sleep_indices = np.where(sleep_mask)[0]
        min_sleep_epochs = params.get("min_sleep_epochs", 20)

        if len(sleep_indices) >= min_sleep_epochs:
            dt_ratio = features["delta_theta_ratio"][sleep_mask]
            valid = np.isfinite(dt_ratio)

            sleep_indices_valid = sleep_indices[valid]
            dt_valid = dt_ratio[valid]

            if len(dt_valid) >= min_sleep_epochs:
                method = params.get("sleep_classification_method", "kmeans")

                if method == "kmeans":
                    # KMeans on delta/theta ratio
                    X = dt_valid.reshape(-1, 1)
                    X_scaled = StandardScaler().fit_transform(X)

                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)

                    # Identify NREM as cluster with higher delta/theta ratio
                    dt_means = [np.mean(dt_valid[labels == i]) for i in (0, 1)]
                    nrem_cluster = np.argmax(dt_means)
                    sleep_states = np.where(labels == nrem_cluster, 0, 1)

                    states[sleep_indices_valid] = sleep_states

                    print(
                        f"NREM epochs: {np.sum(sleep_states==0)}, REM epochs: {np.sum(sleep_states==1)}"
                    )

                else:
                    # Simple threshold method
                    threshold = np.median(dt_valid)
                    sleep_states = np.where(dt_valid >= threshold, 0, 1)
                    states[sleep_indices_valid] = sleep_states
                    print(
                        f"Threshold-based NREM/REM: NREM={np.sum(sleep_states==0)}, REM={np.sum(sleep_states==1)}"
                    )
            else:
                print(
                    "Not enough valid sleep epochs for NREM/REM classification"
                )
        else:
            print("Not enough sleep epochs to classify NREM/REM")

        # --- REM fallback ---
        rem_fraction = np.mean(states == 1)
        if rem_fraction < 0.02 and np.any(sleep_mask):
            theta = features["theta_power"]
            delta = features["delta_power"]

            rem_mask = (
                sleep_mask
                & (
                    theta
                    > np.percentile(
                        theta[sleep_mask], params.get("rem_percentile", 70)
                    )
                )
                & (delta < np.percentile(delta[sleep_mask], 30))
            )
            states[rem_mask] = 1
            print(f"REM fallback applied: {np.sum(rem_mask)} epochs set to REM")

        return states

    def _map_clusters_to_states(self, features, cluster_labels):
        """Map cluster labels to physiological states"""

        n_clusters = len(np.unique(cluster_labels))
        cluster_means = {}

        for cluster in range(n_clusters):
            mask = cluster_labels == cluster
            cluster_means[cluster] = {
                "delta": np.mean(features["delta_power"][mask]),
                "theta": np.mean(features["theta_power"][mask]),
                "emg": np.mean(features["emg_power"][mask]),
                "dt_ratio": np.mean(features["delta_theta_ratio"][mask]),
            }

        state_mapping = {}

        # Identify NREM: highest delta/theta ratio, low EMG
        nrem_scores = [
            (cluster_means[c]["dt_ratio"] - cluster_means[c]["emg"])
            for c in range(n_clusters)
        ]
        nrem_cluster = np.argmax(nrem_scores)
        state_mapping[nrem_cluster] = 0

        # Identify WAKE and REM from remaining
        remaining = [c for c in range(n_clusters) if c != nrem_cluster]

        if len(remaining) > 1:
            wake_scores = [cluster_means[c]["emg"] for c in remaining]
            wake_idx = np.argmax(wake_scores)
            wake_cluster = remaining[wake_idx]
            state_mapping[wake_cluster] = 2

            rem_cluster = [c for c in remaining if c != wake_cluster][0]
            state_mapping[rem_cluster] = 1
        else:
            remaining_cluster = remaining[0]
            if (
                cluster_means[remaining_cluster]["emg"]
                > cluster_means[nrem_cluster]["emg"]
            ):
                state_mapping[remaining_cluster] = 2
            else:
                state_mapping[remaining_cluster] = 1

        states = np.array(
            [state_mapping.get(label, 0) for label in cluster_labels]
        )
        return states

    # ==================== Post-processing ====================

    def _apply_constraints(self, states, params):
        """Apply physiological constraints"""

        constrained = states.copy()
        changes_made = True
        iteration = 0

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
        """Smooth state transitions"""

        smoothed = states.copy()
        min_windows = int(min_duration / window_size)

        # Remove single-point transitions
        for i in range(1, len(smoothed) - 1):
            if (
                smoothed[i] != smoothed[i - 1]
                and smoothed[i] != smoothed[i + 1]
            ):
                smoothed[i] = smoothed[i - 1]

        # Enforce minimum duration
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
        """Convert state labels to time intervals"""

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

        # Handle if still in state at end
        if in_state:
            intervals.append([start_time, timestamps[-1]])

        return np.array(intervals) if intervals else np.array([]).reshape(0, 2)

    # # ==================== Fetch Methods ====================

    def fetch_nrem_times(self):
        """Fetch NREM intervals"""
        interval_name = self.fetch1("nrem_interval_list_name")
        key = {
            "nwb_file_name": self.fetch1("nwb_file_name"),
            "interval_list_name": interval_name,
        }
        return (IntervalList & key).fetch1("valid_times")

    def fetch_rem_times(self):
        """Fetch REM intervals"""
        interval_name = self.fetch1("rem_interval_list_name")
        key = {
            "nwb_file_name": self.fetch1("nwb_file_name"),
            "interval_list_name": interval_name,
        }
        return (IntervalList & key).fetch1("valid_times")

    def fetch_wake_times(self):
        """Fetch WAKE intervals"""
        interval_name = self.fetch1("wake_interval_list_name")
        key = {
            "nwb_file_name": self.fetch1("nwb_file_name"),
            "interval_list_name": interval_name,
        }
        return (IntervalList & key).fetch1("valid_times")

    def plot_hypnogram(self, figsize=(15, 8)):
        """Plot sleep state hypnogram"""

        states = self.fetch1("state_labels")
        timestamps = self.fetch1("timestamps")

        # Convert to hours
        time_hours = (timestamps - timestamps[0]) / 3600

        # State colors
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

        # Add statistics as text
        nrem_pct = self.fetch1("nrem_percentage")
        rem_pct = self.fetch1("rem_percentage")
        wake_pct = self.fetch1("wake_percentage")

        stats_text = (
            f"NREM: {nrem_pct:.1f}%\nREM: {rem_pct:.1f}%\nWAKE: {wake_pct:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

        return fig
