"""
Spyglass W-track Validation + Behavioral Result Pipeline

This module extracts, processes, and validates DIO (digital input/output)
signals from NWB files using the Spyglass + DataJoint pipeline framework
for W-track behavioral analysis. It supports validation against position
data and external log files.

Author: DMR + ChatGPT
Date: Dec 2025
"""

from collections import defaultdict

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb

from spyglass.common import IntervalList, Nwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.position.position_merge import PositionOutput
from spyglass.utils import SpyglassMixin

schema = dj.schema("denissemorales_wtrack")


class WTrackValidator:
    """
    Validator for comparing processed WTrack events against ground truth log files.
    """

    def compare_events(self, extracted_df, wtrack_dict, tolerance=0.001):
        """
        Compare processed events against ground truth from log parser.

        Parameters
        ----------
        extracted_df : pd.DataFrame
            DataFrame with columns ['wtrack_name', 'wtrack_event_times']
        wtrack_dict : dict
            Dictionary from WTrackLogParser.create_wtrack_dict()
            Format: {channel: {'name': str, 'times': np.array, 'values': np.array, ...}}
        tolerance : float
            Time tolerance in seconds for matching events

        Returns
        -------
        dict
            Validation results for each wtrack/channel
        """
        results = {}

        # Build a name-to-ground-truth mapping, filtering for UP events only
        name_to_gt_times = {}
        for ch, data in wtrack_dict.items():
            well_name = data["name"]
            up_mask = data["values"] == 1
            gt_up_times = data["times"][up_mask]
            name_to_gt_times[well_name] = np.sort(gt_up_times)

            print(f"Ground truth for {well_name}: {len(gt_up_times)} UP events")
            if len(gt_up_times) > 0:
                print(
                    f"  Time range: {gt_up_times.min():.2f} to {gt_up_times.max():.2f}"
                )

        # Compare each well
        for well_name in extracted_df["wtrack_name"].unique():
            rows = extracted_df.loc[
                extracted_df["wtrack_name"] == well_name, "wtrack_event_times"
            ]
            extracted_times = (
                np.sort(np.concatenate(rows.values))
                if len(rows) > 0
                else np.array([])
            )

            print(f"\nProcessed {well_name}: {len(extracted_times)} events")
            if len(extracted_times) > 0:
                print(
                    f"  Time range: {extracted_times.min():.2f} to {extracted_times.max():.2f}"
                )

            gt_times = name_to_gt_times.get(well_name, np.array([]))

            matched, missing, extra = 0, [], []

            # Check each ground truth time
            for gt_time in gt_times:
                if extracted_times.size == 0 or not np.any(
                    np.abs(extracted_times - gt_time) < tolerance
                ):
                    missing.append(
                        {"time": float(gt_time), "reason": "not_found"}
                    )
                else:
                    matched += 1

            # Check each processed time
            for proc_time in extracted_times:
                if gt_times.size == 0 or not np.any(
                    np.abs(gt_times - proc_time) < tolerance
                ):
                    extra.append({"time": float(proc_time)})

            results[well_name] = {
                "status": "COMPARED",
                "ground_truth_count": int(len(gt_times)),
                "processed_count": int(len(extracted_times)),
                "matched": int(matched),
                "missing_in_processed": missing,
                "extra_in_processed": extra,
                "match_rate": (
                    matched / len(gt_times) if len(gt_times) > 0 else 0
                ),
            }

        return results

    def print_validation_report(self, validation_results):
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("W-TRACK EVENT VALIDATION REPORT")
        print("=" * 70)

        for label, result in validation_results.items():
            print(f"\n{label}:")
            if result.get("status") == "NOT_IN_GROUND_TRUTH":
                print(f"  ⚠ {result['message']}")
                continue

            print(f"  Ground truth events: {result['ground_truth_count']}")
            print(f"  Processed events: {result['processed_count']}")
            print(f"  Matched events: {result['matched']}")
            print(f"  Match rate: {result['match_rate']*100:.1f}%")

            if result.get("missing_in_processed"):
                print(
                    f"  ⚠ Missing in processed: {len(result['missing_in_processed'])}"
                )
            if result.get("extra_in_processed"):
                print(
                    f"  ⚠ Extra in processed: {len(result['extra_in_processed'])}"
                )


class WTrackLogParser:
    """Parse DIO event logs and extract events for validation."""

    def __init__(self, timestamp_scale=1000.0):
        """
        Parameters
        ----------
        timestamp_scale : float
            Scale factor to convert timestamps to seconds (default 1000 = milliseconds)
        """
        self.timestamp_scale = timestamp_scale

    def parse_log_file(self, filepath):
        """
        Parse DIO event log file.

        Parameters
        ----------
        filepath : str
            Path to the log file

        Returns
        -------
        dict
            Dictionary mapping DIO channel to event data:
            {
                dio_channel: {
                    'times': np.array,
                    'values': np.array (0 for DOWN, 1 for UP),
                    'raw_values': list of (value1, value2) tuples
                }
            }
        list
            List of reward events (if any)
        """
        dio_events = defaultdict(
            lambda: {"times": [], "values": [], "raw_values": []}
        )
        reward_events = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and separator lines
            if not line or line.startswith("~~~"):
                i += 1
                continue

            # Parse reward/animal state lines
            if "=" in line:
                reward_events.append(line)
                i += 1
                continue

            # Parse DIO events
            parts = line.split()
            if len(parts) >= 3:
                if not parts[0].isdigit():
                    i += 1
                    continue

                try:
                    timestamp = int(parts[0])
                except ValueError:
                    i += 1
                    continue

                time_sec = timestamp / self.timestamp_scale

                if parts[1] in ["UP", "DOWN"]:
                    try:
                        dio_channel = int(parts[2])
                    except ValueError:
                        i += 1
                        continue

                    value = 1 if parts[1] == "UP" else 0

                    dio_events[dio_channel]["times"].append(time_sec)
                    dio_events[dio_channel]["values"].append(value)

                    # Look ahead for raw values
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not next_line.startswith("~~~"):
                            next_parts = next_line.split()
                            if len(next_parts) >= 2 and next_parts[0].isdigit():
                                try:
                                    val1 = int(next_parts[1])
                                    val2 = (
                                        int(next_parts[2])
                                        if len(next_parts) > 2
                                        else None
                                    )
                                    dio_events[dio_channel][
                                        "raw_values"
                                    ].append((val1, val2))
                                except (ValueError, IndexError):
                                    dio_events[dio_channel][
                                        "raw_values"
                                    ].append((None, None))
                            else:
                                dio_events[dio_channel]["raw_values"].append(
                                    (None, None)
                                )
                        else:
                            dio_events[dio_channel]["raw_values"].append(
                                (None, None)
                            )

            i += 1

        parsed_events = {}
        for dio_channel, data in dio_events.items():
            parsed_events[dio_channel] = {
                "times": np.array(data["times"]),
                "values": np.array(data["values"]),
                "raw_values": data["raw_values"],
            }

        return parsed_events, reward_events

    def create_wtrack_dict(self, parsed_events, dio_name_map):
        """
        Convert parsed DIO logs → standardized DIO dictionary.
        """
        dio_dict = {}
        for ch, data in parsed_events.items():
            if ch in dio_name_map:
                dio_dict[ch] = {
                    "name": dio_name_map[ch],
                    "times": data["times"],
                    "values": data["values"],
                    "description": f"DIO channel {ch}",
                }
        return dio_dict


# =====================================================
# PARAMETERS
# =====================================================
@schema
class WTrackParams(SpyglassMixin, dj.Manual):
    definition = """
    wtrack_params_name: varchar(64)
    ---
    event_name_map: longblob
    dio_channel_map: longblob
    distance_threshold: float
    validate_against_position: bool
    validate_against_log: bool
    well_positions: longblob
    """

    @classmethod
    def insert_default(cls):
        default = dict(
            wtrack_params_name="default",
            event_name_map={
                "LeftWell_Poke": "Left_poke",
                "CenterWell_Poke": "Center_poke",
                "RightWell_Poke": "Right_poke",
                "LeftMilk_Pump": "Left_pump",
                "CenterMilk_Pump": "Center_pump",
                "RightMilk_Pump": "Right_pump",
            },
            dio_channel_map={
                6: "Left_poke",
                8: "Center_poke",
                14: "Right_poke",
                7: "Left_pump",
                9: "Center_pump",
                15: "Right_pump",
            },
            distance_threshold=15.0,
            validate_against_position=True,
            validate_against_log=True,
            well_positions={
                "Left_poke": (155, 45),
                "Center_poke": (125, 60),
                "Right_poke": (95, 78),
            },
        )
        cls.insert1(default, skip_duplicates=True)


# =====================================================
# SELECTION TABLE
# =====================================================
@schema
class WTrackSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> WTrackParams
    -> Nwbfile
    -> PositionOutput.proj(pos_merge_id='merge_id')
    ---
    epoch: int
    statescript_path='': varchar(512)
    """


class PositionValidator:
    """
    Validates DIO poke events using animal position data.
    """

    def __init__(
        self, well_positions, distance_threshold=15.0, max_speed=150.0
    ):
        """
        Parameters
        ----------
        well_positions : dict
            Map well names to (x, y) coordinates.
        distance_threshold : float
            Maximum distance from well for valid poke (in position units).
        max_speed : float
            Maximum plausible speed (position units per second).
        """
        self.well_positions = well_positions
        self.distance_threshold = distance_threshold
        self.max_speed = max_speed

    def validate_poke_events(
        self,
        poke_times,
        poke_names,
        poke_values=None,
        position_times=None,
        position_x=None,
        position_y=None,
        plot=True,
    ):
        """
        Validate poke events based on position.

        Parameters
        ----------
        poke_times : np.array
            Timestamps of poke events.
        poke_names : np.array
            Names of poked wells.
        poke_values : np.array, optional
            Poke values (0 or 1). Defaults to 1 (valid) if not provided.
        position_times : np.array
            Position sample times.
        position_x : np.array
            X coordinates.
        position_y : np.array
            Y coordinates.
        plot : bool
            Whether to create validation plot.

        Returns
        -------
        dict
            Dictionary with:
            - 'valid_pokes': DataFrame with valid pokes.
            - 'invalid_pokes': DataFrame with rejected pokes.
            - 'summary': dict with statistics.
        """

        # ----------------------------------
        # EARLY EXIT: no position data
        # ----------------------------------
        if (
            position_times is None
            or position_x is None
            or position_y is None
            or len(position_times) == 0
        ):
            poke_df = pd.DataFrame(
                {
                    "time": poke_times,
                    "well_name": poke_names,
                    "value": (
                        poke_values
                        if poke_values is not None
                        else np.ones(len(poke_times), dtype=int)
                    ),
                }
            )

            summary = {
                "total_pokes": len(poke_df),
                "valid_pokes": len(poke_df),
                "invalid_pokes": 0,
                "percent_valid": 100.0 if len(poke_df) else 0,
                "note": "Position validation skipped (no position data)",
            }

            return {
                "valid_pokes": poke_df.reset_index(drop=True),
                "invalid_pokes": poke_df.iloc[0:0],
                "summary": summary,
            }

        if poke_values is None:
            poke_values = np.ones(len(poke_times), dtype=int)

        animal_positions = interpolate_position(
            position_times, position_x, position_y, poke_times
        )

        # Create DataFrame with poke events and interpolated positions
        poke_df = pd.DataFrame(
            {
                "time": poke_times,
                "well_name": poke_names,
                "value": poke_values,
                "animal_x": animal_positions[:, 0],
                "animal_y": animal_positions[:, 1],
            }
        )

        # Compute distance to wells
        distances, well_x, well_y = [], [], []
        for _, row in poke_df.iterrows():
            if row["well_name"] in self.well_positions:
                wx, wy = self.well_positions[row["well_name"]]
                dist = np.sqrt(
                    (row["animal_x"] - wx) ** 2 + (row["animal_y"] - wy) ** 2
                )
            else:
                wx, wy, dist = np.nan, np.nan, np.inf
            distances.append(dist)
            well_x.append(wx)
            well_y.append(wy)

        # Add distance and well position data to DataFrame
        poke_df["distance_to_well"] = distances
        poke_df["well_x"] = well_x
        poke_df["well_y"] = well_y

        # Validate by distance
        valid_mask = poke_df["distance_to_well"] <= self.distance_threshold
        valid_pokes = poke_df[valid_mask].copy()
        invalid_pokes = poke_df[~valid_mask].copy()

        # Speed check for valid pokes
        if len(valid_pokes) > 1:
            speed_mask = np.ones(len(valid_pokes), dtype=bool)
            for i in range(1, len(valid_pokes)):
                prev = valid_pokes.iloc[i - 1]
                curr = valid_pokes.iloc[i]

                dist = np.sqrt(
                    (curr["animal_x"] - prev["animal_x"]) ** 2
                    + (curr["animal_y"] - prev["animal_y"]) ** 2
                )
                dt = curr["time"] - prev["time"]

                if dt > 0 and dist / dt > self.max_speed:
                    speed_mask[i] = False

            # Apply speed mask
            invalid_speed_pokes = valid_pokes[~speed_mask]
            valid_pokes = valid_pokes[speed_mask]
            invalid_pokes = pd.concat([invalid_pokes, invalid_speed_pokes])

        # Summary statistics
        summary = {
            "total_pokes": len(poke_df),
            "valid_pokes": len(valid_pokes),
            "invalid_pokes": len(invalid_pokes),
            "percent_valid": (
                (100 * len(valid_pokes) / len(poke_df))
                if len(poke_df) > 0
                else 0
            ),
        }

        print(f"\nPosition validation summary:")
        print(f"  Total pokes: {summary['total_pokes']}")
        print(
            f"  Valid pokes: {summary['valid_pokes']} ({summary['percent_valid']:.1f}%)"
        )
        print(f"  Invalid pokes: {summary['invalid_pokes']}")

        if len(invalid_pokes) > 0:
            print(f"\nInvalid poke details:")
            for _, row in invalid_pokes.iterrows():
                print(
                    f"  {row['well_name']} at t={row['time']:.2f}s, distance={row['distance_to_well']:.1f}"
                )

        # Plotting
        if plot:
            self._plot_validation(
                valid_pokes, invalid_pokes, position_x, position_y
            )

        return {
            "valid_pokes": valid_pokes.reset_index(drop=True),
            "invalid_pokes": invalid_pokes.reset_index(drop=True),
            "summary": summary,
        }

    def _plot_validation(
        self, valid_pokes, invalid_pokes, position_x, position_y
    ):
        """Create visualization of position validation."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory
        ax.plot(
            position_x,
            position_y,
            "-",
            alpha=0.2,
            linewidth=0.5,
            label="Trajectory",
        )

        # Plot wells
        for well_name, (wx, wy) in self.well_positions.items():
            ax.plot(wx, wy, "ko", markersize=10)
            ax.text(wx, wy, well_name, ha="center", fontsize=9)
            circle = plt.Circle((wx, wy), self.distance_threshold, alpha=0.2)
            ax.add_patch(circle)

        # Plot valid pokes
        if len(valid_pokes) > 0:
            ax.plot(
                valid_pokes["animal_x"],
                valid_pokes["animal_y"],
                "go",
                markersize=8,
                label="Valid pokes",
                alpha=0.7,
            )

        # Plot invalid pokes
        if len(invalid_pokes) > 0:
            ax.plot(
                invalid_pokes["animal_x"],
                invalid_pokes["animal_y"],
                "rx",
                markersize=10,
                markeredgewidth=2,
                label="Invalid pokes",
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Position-Based DIO Validation")
        ax.legend()
        ax.axis("equal")
        plt.tight_layout()
        plt.show()


# =====================================================
# COMPUTED TABLE
# # =====================================================
@schema
class WTrackEvents(SpyglassMixin, dj.Computed):
    definition = """
    -> WTrackSelection
    ---
    epoch: int
    wtrack_results: longblob
    validation_report: longblob
    n_events: int
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def get_first_pokes_after_well_change(self, poke_df):

        poke_df = (
            poke_df.rename(columns={"wtrack_name": "well_name"})
            .sort_values("time")
            .reset_index(drop=True)
        )
        poke_df["prev_well"] = poke_df["well_name"].shift(1)
        first_pokes = poke_df[
            poke_df["well_name"] != poke_df["prev_well"]
        ].reset_index(drop=True)
        return first_pokes

    def make(self, key):
        """Extract, align, and validate W-track events from NWB."""
        # ----------------------------
        # 1. Fetch inputs
        # ----------------------------
        selection = (WTrackSelection & key).fetch1()
        statescript_path = selection.get("statescript_path", None)
        epoch = selection["epoch"] - 1

        params = (WTrackParams & key).fetch1()
        nwb_file_name = (Nwbfile & key).fetch1("nwb_file_name")
        nwb_path = Nwbfile().get_abs_path(nwb_file_name)

        # ----------------------------
        # 2. Load NWB + extract DIO rising edges
        # ----------------------------
        name_mapping = {
            "LeftWell_Poke": "Left_poke",
            "CenterWell_Poke": "Center_poke",
            "RightWell_Poke": "Right_poke",
            "LeftMilk_Pump": "Left_pump",
            "CenterMilk_Pump": "Center_pump",
            "RightMilk_Pump": "Right_pump",
        }

        rows = []

        with pynwb.NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()

            epoch_row = nwb.intervals["epochs"][epoch].to_numpy()[0]
            epoch_start, epoch_stop, _ = epoch_row

            dios = (
                nwb.processing["behavior"]
                .data_interfaces["behavioral_events"]
                .time_series
            )

            for nwb_name, mapped_name in name_mapping.items():
                if nwb_name not in dios:
                    continue

                ts = np.asarray(dios[nwb_name].timestamps[:])
                data = np.asarray(dios[nwb_name].data[:]).astype(int)

                mask = (ts > epoch_start) & (ts <= epoch_stop)
                ts = ts[mask]
                data = data[mask]

                # 🔑 rising edge detection
                edge_mask = (data[1:] == 1) & (data[:-1] == 0)
                edge_times = ts[1:][edge_mask]

                for t in edge_times:
                    rows.append(dict(time=float(t), wtrack_name=mapped_name))

        wtrack_df = (
            pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        )

        # ----------------------------
        # 3. Separate pokes and pumps
        # ----------------------------
        poke_df = wtrack_df[wtrack_df.wtrack_name.str.contains("poke")].copy()
        pump_df = wtrack_df[wtrack_df.wtrack_name.str.contains("pump")].copy()

        # Save RAW pokes for alignment
        raw_poke_df = poke_df.copy()

        # Filter to first poke after well change
        poke_df = self.get_first_pokes_after_well_change(poke_df)
        # poke_df now has: time | well_name | prev_well

        # ----------------------------
        # 4. Load position data
        # ----------------------------
        position_df = (
            (PositionOutput & {"merge_id": key["pos_merge_id"]})
            .fetch1_dataframe()
            .reset_index()
        )
        position_df = position_df[
            (position_df.position_x > 5) & (position_df.position_y > 5)
        ]

        position_times = position_df.time.to_numpy()
        position_x = position_df.position_x.to_numpy()
        position_y = position_df.position_y.to_numpy()

        # ----------------------------
        # 5. Build final_df (INCLUDING prev_well)
        # ----------------------------
        pump_map = {
            "Left_poke": "Left_pump",
            "Center_poke": "Center_pump",
            "Right_poke": "Right_pump",
        }

        final_rows = []
        last_reward_time = np.nan

        for _, r in poke_df.iterrows():
            t = r.time
            well = r.well_name
            prev_well = r.prev_well
            trial_type = "Inbound" if "Center" in well else "Outbound"
            transition = "" if prev_well is None else f"{prev_well}→{well}"
            prev_well_safe = "" if prev_well is None else prev_well

            pump_triggered = False
            pump_time = np.nan
            pump_delay = np.nan
            time_between_rewards = np.nan

            expected_pump = pump_map.get(well)
            if expected_pump:
                hits = pump_df[
                    (pump_df.wtrack_name == expected_pump)
                    & (pump_df.time >= t)
                    & (pump_df.time <= t + 0.5)
                ]
                if len(hits):
                    pump_triggered = True
                    pump_time = hits.iloc[0].time
                    pump_delay = pump_time - t
                    if pd.notna(last_reward_time):
                        time_between_rewards = pump_time - last_reward_time
                    last_reward_time = pump_time

            final_rows.append(
                dict(
                    time=t,
                    epoch=epoch + 1,
                    well_name=well,
                    prev_well=prev_well,
                    transition=transition,
                    trial_type=trial_type,
                    pump_triggered=pump_triggered,
                    pump_time=pump_time,
                    pump_delay=pump_delay,
                    time_between_rewards=time_between_rewards,
                )
            )

        final_df = pd.DataFrame(final_rows)

        # ----------------------------
        # 6. Position validation
        # ----------------------------
        final_df["position_valid"] = True
        validation_report = {}

        if params["validate_against_position"] and len(final_df):
            validator = PositionValidator(
                params["well_positions"],
                params["distance_threshold"],
            )

            report = validator.validate_poke_events(
                poke_times=final_df.time.to_numpy(),
                poke_names=final_df.well_name.to_numpy(),
                position_times=position_times,
                position_x=position_x,
                position_y=position_y,
                plot=False,
            )

            invalid = set(report["invalid_pokes"].time.values)
            tol = 1e-6
            final_df["position_valid"] = ~final_df.time.apply(
                lambda t: any(abs(t - it) < tol for it in invalid)
            )

            validation_report["position"] = report["summary"]

        # ----------------------------
        # 7. Log validation with ROBUST alignment
        # ----------------------------
        if params["validate_against_log"] and statescript_path:
            parser = WTrackLogParser(timestamp_scale=1000.0)
            parsed, _ = parser.parse_log_file(statescript_path)
            wtrack_dict = parser.create_wtrack_dict(
                parsed, params["dio_channel_map"]
            )

            log_rows = []
            for d in wtrack_dict.values():
                if "poke" not in d["name"].lower():
                    continue
                ups = d["times"][d["values"] == 1]
                for t in ups:
                    log_rows.append(dict(time=float(t), well_name=d["name"]))

            log_df = (
                pd.DataFrame(log_rows)
                .sort_values("time")
                .reset_index(drop=True)
            )

            # 🔑 robust offset from multiple events
            N = min(len(raw_poke_df), len(log_df), 20)
            offset = np.median(
                raw_poke_df.time.to_numpy()[:N] - log_df.time.to_numpy()[:N]
            )
            log_df["time"] += offset

            # Filter AFTER alignment
            log_df = self.get_first_pokes_after_well_change(log_df)

            tolerance = 0.02
            results = {}

            for well, grp in final_df.groupby("well_name"):
                proc = grp.time.to_numpy()
                gt = log_df[log_df.well_name == well].time.to_numpy()

                matched_idx = set()
                matched = 0

                for g in gt:
                    if len(proc):
                        d = np.abs(proc - g)
                        i = np.argmin(d)
                        if d[i] < tolerance and i not in matched_idx:
                            matched_idx.add(i)
                            matched += 1

                results[well] = dict(
                    ground_truth_count=len(gt),
                    processed_count=len(proc),
                    matched=matched,
                    match_rate=matched / len(gt) if len(gt) else 0,
                )

            WTrackValidator().print_validation_report(results)
            validation_report["log"] = results

        # ----------------------------
        # 8. Store results
        # ----------------------------
        final_df["prev_well"] = final_df["prev_well"].astype(str)
        final_df["transition"] = final_df["transition"].astype(str)
        final_rec = final_df.to_records(index=False)

        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(final_df)
            analysis_file = builder.analysis_file_name

        self.insert1(
            dict(
                **key,
                epoch=epoch + 1,
                wtrack_results=final_rec,
                validation_report=validation_report,
                n_events=len(final_df),
                analysis_file_name=analysis_file,
                trial_object_id=obj_id,
            )
        )


def interpolate_position(position_times, position_x, position_y, query_times):

    print(query_times.min(), query_times.max())
    print(position_times[0], position_times[-1])

    position_times = np.asarray(position_times)
    assert np.all(
        np.diff(position_times) > 0
    ), "position_times not strictly increasing"

    interp_x = np.interp(query_times, position_times, position_x)
    interp_y = np.interp(query_times, position_times, position_y)

    return np.column_stack((interp_x, interp_y))
