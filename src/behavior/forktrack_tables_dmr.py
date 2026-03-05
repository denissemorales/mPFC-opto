"""
Spyglass Fork Track Validation + Behavioral Result Pipeline

This module extracts, processes, and validates DIO (digital input/output)
signals from NWB files using the Spyglass + DataJoint pipeline framework
for fork maze behavioral analysis. It supports validation against position
data and external log files.

Author: DMR
Date: March 2026
"""

from collections import defaultdict
import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from spyglass.common import Nwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.position.position_merge import PositionOutput
from spyglass.utils import SpyglassMixin

schema = dj.schema("denissemorales_forktrack")


# =====================================================
# PARAMETERS
# =====================================================
@schema
class ForktTrackParams(SpyglassMixin, dj.Manual):
    definition = """
    forktrack_params_name: varchar(64)
    ---
    event_name_map: blob
    dio_channel_map: blob
    distance_threshold: float
    validate_against_position: bool
    validate_against_log: bool
    well_positions: blob
    """

    @classmethod
    def insert_default(cls):
        default = dict(
            forktrack_params_name="default",
            event_name_map={
                "LeftWell_Poke": "Left_poke",
                "CenterWell_Poke": "Center_poke",
                "HandleWell_Poke": "Handle_poke",
                "RightWell_Poke": "Right_poke",
                "LeftMilk_Pump": "Left_pump",
                "CenterMilk_Pump": "Center_pump",
                "RightMilk_Pump": "Right_pump",
                "HandleMilk_Pump": "Handle_pump",
            },
            dio_channel_map={
                1: "Left_poke",
                13: "Center_poke",
                14: "Right_poke",
                7: "Handle_poke",
                2: "Left_pump",
                12: "Center_pump",
                11: "Right_pump",
                8: "Handle_pump",
            },
            distance_threshold=15.0,
            validate_against_position=True,
            validate_against_log=True,
            well_positions={
                "Left_poke": (155, 45),
                "Center_poke": (125, 60),
                "Right_poke": (95, 78),
                "Handle_poke": (0, 0),
            },
        )
        cls.insert1(default, skip_duplicates=True)


# =====================================================
# SELECTION TABLE
# =====================================================
@schema
class ForktTrackSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ForktTrackParams
    -> Nwbfile
    -> PositionOutput.proj(pos_merge_id='merge_id')
    ---
    epoch: int
    statescript_path='': varchar(512)
    """


# =====================================================
# COMPUTED TABLE
# =====================================================
@schema
class ForktTrackEvents(SpyglassMixin, dj.Computed):
    definition = """
    -> ForktTrackSelection
    ---
    epoch: int
    forktrack_results: blob
    validation_report: blob
    n_events: int
    is_valid: bool                # True if all validation checks passed
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make_fetch(self, key):
        """Fetch all upstream data needed to compute fork track events.

        Returns
        -------
        list
            [nwb_file_name, nwb_path, epoch, params, statescript_path,
             position_times, position_x, position_y]
        """
        selection = (ForktTrackSelection & key).fetch1()
        statescript_path = selection.get("statescript_path", None)
        epoch = selection["epoch"] - 1

        params = (ForktTrackParams & key).fetch1()
        nwb_file_name = (Nwbfile & key).fetch1("nwb_file_name")
        nwb_path = Nwbfile().get_abs_path(nwb_file_name)

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

        return [
            nwb_file_name,
            nwb_path,
            epoch,
            params,
            statescript_path,
            position_times,
            position_x,
            position_y,
        ]

    def make_compute(
        self,
        key,
        nwb_file_name,
        nwb_path,
        epoch,
        params,
        statescript_path,
        position_times,
        position_x,
        position_y,
    ):
        """Extract DIO events, build trial DataFrame, and run validation.

        Parameters
        ----------
        key : dict
            Primary key to ForktTrackSelection.
        nwb_file_name : str
            Name of the NWB file.
        nwb_path : str
            Absolute path to the NWB file.
        epoch : int
            Zero-indexed epoch number.
        params : dict
            Output of ForktTrackParams.fetch1().
        statescript_path : str or None
            Path to the statescript log file, or None.
        position_times : np.array
            Position sample times.
        position_x : np.array
            X coordinates.
        position_y : np.array
            Y coordinates.

        Returns
        -------
        list
            [nwb_file_name, final_df, validation_report, epoch]
        """
        # ----------------------------
        # 1. Load NWB + extract DIO rising edges
        # ----------------------------
        name_mapping = {
            "LeftWell_Poke": "Left_poke",
            "CenterWell_Poke": "Center_poke",
            "HandleWell_Poke": "Handle_poke",
            "RightWell_Poke": "Right_poke",
            "LeftMilk_Pump": "Left_pump",
            "CenterMilk_Pump": "Center_pump",
            "RightMilk_Pump": "Right_pump",
            "HandleMilk_Pump": "Handle_pump",
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

                # rising edge detection
                edge_mask = (data[1:] == 1) & (data[:-1] == 0)
                edge_times = ts[1:][edge_mask]

                for t in edge_times:
                    rows.append(dict(time=float(t), forktrack_name=mapped_name))

        forktrack_df = (
            pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        )

        # ----------------------------
        # 2. Separate pokes and pumps
        # ----------------------------
        poke_df = forktrack_df[forktrack_df.forktrack_name.str.contains("poke")].copy()
        pump_df = forktrack_df[forktrack_df.forktrack_name.str.contains("pump")].copy()

        # Save RAW pokes for alignment
        raw_poke_df = poke_df.copy()

        # Filter to first poke after well change
        poke_df = get_first_pokes_after_well_change(poke_df)

        # ----------------------------
        # 3. Build final_df (INCLUDING prev_well)
        # ----------------------------
        pump_map = {
            "Left_poke": "Left_pump",
            "Center_poke": "Center_pump",
            "Right_poke": "Right_pump",
            "Handle_poke": "Handle_pump",
        }

        final_rows = []
        last_reward_time = np.nan

        for _, r in poke_df.iterrows():
            t = r.time
            well = r.well_name
            prev_well = r.prev_well
            trial_type = "Inbound" if "Center" in well or "Handle" in well else "Outbound"
            transition = "" if prev_well is None else f"{prev_well}→{well}"

            pump_triggered = False
            pump_time = np.nan
            pump_delay = np.nan
            time_between_rewards = np.nan

            expected_pump = pump_map.get(well)
            if expected_pump:
                hits = pump_df[
                    (pump_df.forktrack_name == expected_pump)
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
        # 4. Position validation
        # ----------------------------
        final_df["position_valid"] = True
        validation_report = {}

        if params["validate_against_position"] and len(final_df):
            report = validate_poke_events(
                well_positions=params["well_positions"],
                distance_threshold=params["distance_threshold"],
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
        # 5. Log validation with ROBUST alignment
        # ----------------------------
        if params["validate_against_log"] and statescript_path:
            parsed, _ = parse_log_file(
                statescript_path, timestamp_scale=1000.0
            )
            forktrack_dict = create_forktrack_dict(parsed, params["dio_channel_map"])

            log_rows = []
            for d in forktrack_dict.values():
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

            # robust offset from multiple events
            N = min(len(raw_poke_df), len(log_df), 20)
            offset = np.median(
                raw_poke_df.time.to_numpy()[:N] - log_df.time.to_numpy()[:N]
            )
            log_df["time"] += offset

            # Filter AFTER alignment
            log_df = get_first_pokes_after_well_change(log_df)

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

            print_forktrack_validation_report(results)
            validation_report["log"] = results

        # Compute is_valid: True if all checks passed (no invalid pokes, full log match)
        pos_valid = True
        log_valid = True

        if "position" in validation_report:
            pos_valid = validation_report["position"]["invalid_pokes"] == 0

        if "log" in validation_report:
            log_valid = all(
                r["match_rate"] == 1.0
                for r in validation_report["log"].values()
            )

        is_valid = pos_valid and log_valid

        return [nwb_file_name, final_df, validation_report, epoch, is_valid]

    def make_insert(self, key, nwb_file_name, final_df, validation_report, epoch, is_valid):
        """Write results to NWB and insert into ForktTrackEvents.

        Parameters
        ----------
        key : dict
            Primary key to ForktTrackSelection.
        nwb_file_name : str
            Name of the NWB file.
        final_df : pd.DataFrame
            Trial-level DataFrame produced by make_compute.
        validation_report : dict
            Validation summary produced by make_compute.
        epoch : int
            Zero-indexed epoch number (will be stored as epoch + 1).
        is_valid : bool
            True if all position and log validation checks passed.
        """
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
                forktrack_results=final_rec,
                validation_report=validation_report,
                n_events=len(final_df),
                is_valid=is_valid,
                analysis_file_name=analysis_file,
                trial_object_id=obj_id,
            )
        )


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_first_pokes_after_well_change(poke_df):
    """
    Filter a poke DataFrame to only the first poke each time the animal
    visits a new well.

    Parameters
    ----------
    poke_df : pd.DataFrame
        DataFrame with columns ['forktrack_name' or 'well_name', 'time']

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with added 'prev_well' column
    """
    poke_df = (
        poke_df.rename(columns={"forktrack_name": "well_name"})
        .sort_values("time")
        .reset_index(drop=True)
    )
    poke_df["prev_well"] = poke_df["well_name"].shift(1)
    first_pokes = poke_df[
        poke_df["well_name"] != poke_df["prev_well"]
    ].reset_index(drop=True)
    return first_pokes


def validate_poke_events(
    well_positions,
    poke_times,
    poke_names,
    poke_values=None,
    position_times=None,
    position_x=None,
    position_y=None,
    distance_threshold=15.0,
    max_speed=150.0,
    plot=True,
):
    """
    Validate poke events based on animal position.

    Parameters
    ----------
    well_positions : dict
        Map of well names to (x, y) coordinates.
    poke_times : np.array
        Timestamps of poke events.
    poke_names : np.array
        Names of poked wells.
    poke_values : np.array, optional
        Poke values (0 or 1). Defaults to all 1s if not provided.
    position_times : np.array
        Position sample times.
    position_x : np.array
        X coordinates.
    position_y : np.array
        Y coordinates.
    distance_threshold : float
        Maximum distance from well for a valid poke (in position units).
    max_speed : float
        Maximum plausible speed (position units per second).
    plot : bool
        Whether to create a validation plot.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'valid_pokes': DataFrame of valid pokes
        - 'invalid_pokes': DataFrame of rejected pokes
        - 'summary': dict with statistics
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
        if row["well_name"] in well_positions:
            wx, wy = well_positions[row["well_name"]]
            dist = np.sqrt(
                (row["animal_x"] - wx) ** 2 + (row["animal_y"] - wy) ** 2
            )
        else:
            wx, wy, dist = np.nan, np.nan, np.inf
        distances.append(dist)
        well_x.append(wx)
        well_y.append(wy)

    poke_df["distance_to_well"] = distances
    poke_df["well_x"] = well_x
    poke_df["well_y"] = well_y

    # Validate by distance
    valid_mask = poke_df["distance_to_well"] <= distance_threshold
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

            if dt > 0 and dist / dt > max_speed:
                speed_mask[i] = False

        invalid_speed_pokes = valid_pokes[~speed_mask]
        valid_pokes = valid_pokes[speed_mask]
        invalid_pokes = pd.concat([invalid_pokes, invalid_speed_pokes])

    summary = {
        "total_pokes": len(poke_df),
        "valid_pokes": len(valid_pokes),
        "invalid_pokes": len(invalid_pokes),
        "percent_valid": (
            (100 * len(valid_pokes) / len(poke_df)) if len(poke_df) > 0 else 0
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

    if plot:
        _plot_position_validation(
            valid_pokes, invalid_pokes, position_x, position_y,
            well_positions, distance_threshold,
        )

    return {
        "valid_pokes": valid_pokes.reset_index(drop=True),
        "invalid_pokes": invalid_pokes.reset_index(drop=True),
        "summary": summary,
    }


def _plot_position_validation(
    valid_pokes, invalid_pokes, position_x, position_y,
    well_positions, distance_threshold,
):
    """Create visualization of position-based poke validation."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        position_x,
        position_y,
        "-",
        alpha=0.2,
        linewidth=0.5,
        label="Trajectory",
    )

    for well_name, (wx, wy) in well_positions.items():
        ax.plot(wx, wy, "ko", markersize=10)
        ax.text(wx, wy, well_name, ha="center", fontsize=9)
        circle = plt.Circle((wx, wy), distance_threshold, alpha=0.2)
        ax.add_patch(circle)

    if len(valid_pokes) > 0:
        ax.plot(
            valid_pokes["animal_x"],
            valid_pokes["animal_y"],
            "go",
            markersize=8,
            label="Valid pokes",
            alpha=0.7,
        )

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


def compare_forktrack_events(extracted_df, forktrack_dict, tolerance=0.001):
    """
    Compare processed events against ground truth from log parser.

    Parameters
    ----------
    extracted_df : pd.DataFrame
        DataFrame with columns ['forktrack_name', 'forktrack_event_times']
    forktrack_dict : dict
        Dictionary from create_forktrack_dict()
        Format: {channel: {'name': str, 'times': np.array, 'values': np.array, ...}}
    tolerance : float
        Time tolerance in seconds for matching events

    Returns
    -------
    dict
        Validation results for each forktrack/channel
    """
    results = {}

    # Build a name-to-ground-truth mapping, filtering for UP events only
    name_to_gt_times = {}
    for ch, data in forktrack_dict.items():
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
    for well_name in extracted_df["forktrack_name"].unique():
        rows = extracted_df.loc[
            extracted_df["forktrack_name"] == well_name, "forktrack_event_times"
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

        for gt_time in gt_times:
            if extracted_times.size == 0 or not np.any(
                np.abs(extracted_times - gt_time) < tolerance
            ):
                missing.append({"time": float(gt_time), "reason": "not_found"})
            else:
                matched += 1

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


def print_forktrack_validation_report(validation_results):
    """
    Print formatted validation report.

    Parameters
    ----------
    validation_results : dict
        Output of compare_forktrack_events()
    """
    print("\n" + "=" * 70)
    print("FORK TRACK EVENT VALIDATION REPORT")
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


def _parse_raw_values(lines, i):
    """
    Look ahead one line to extract raw DIO values.

    Parameters
    ----------
    lines : list of str
    i : int
        Current line index.

    Returns
    -------
    tuple
        (val1, val2) parsed from the next line, or (None, None) if unavailable.
    """
    if i + 1 >= len(lines):
        return (None, None)

    next_line = lines[i + 1].strip()

    if not next_line or next_line.startswith("~~~"):
        return (None, None)

    next_parts = next_line.split()

    if len(next_parts) < 2 or not next_parts[0].isdigit():
        return (None, None)

    try:
        val1 = int(next_parts[1])
        val2 = int(next_parts[2]) if len(next_parts) > 2 else None
        return (val1, val2)
    except (ValueError, IndexError):
        return (None, None)


def parse_log_file(filepath, timestamp_scale=1000.0):
    """
    Parse DIO event log file.

    Parameters
    ----------
    filepath : str
        Path to the log file
    timestamp_scale : float
        Scale factor to convert timestamps to seconds (default 1000 = milliseconds)

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

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line or line.startswith("~~~"):
            continue

        if "=" in line:
            reward_events.append(line)
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        if not parts[0].isdigit():
            continue

        try:
            timestamp = int(parts[0])
        except ValueError:
            continue

        if parts[1] not in ["UP", "DOWN"]:
            continue

        try:
            dio_channel = int(parts[2])
        except ValueError:
            continue

        time_sec = timestamp / timestamp_scale
        value = 1 if parts[1] == "UP" else 0

        dio_events[dio_channel]["times"].append(time_sec)
        dio_events[dio_channel]["values"].append(value)
        dio_events[dio_channel]["raw_values"].append(_parse_raw_values(lines, i))

    parsed_events = {
        dio_channel: {
            "times": np.array(data["times"]),
            "values": np.array(data["values"]),
            "raw_values": data["raw_values"],
        }
        for dio_channel, data in dio_events.items()
    }

    return parsed_events, reward_events


def create_forktrack_dict(parsed_events, dio_name_map):
    """
    Convert parsed DIO logs to a standardized DIO dictionary.

    Parameters
    ----------
    parsed_events : dict
        Output of parse_log_file()
    dio_name_map : dict
        Mapping of DIO channel numbers to well names

    Returns
    -------
    dict
        Standardized dictionary keyed by channel with name, times, values, description
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


def interpolate_position(position_times, position_x, position_y, query_times):
    """
    Interpolate x/y position at arbitrary query times.

    Parameters
    ----------
    position_times : np.array
        Strictly increasing array of position sample times.
    position_x : np.array
        X coordinates at each sample time.
    position_y : np.array
        Y coordinates at each sample time.
    query_times : np.array
        Times at which to interpolate position.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with interpolated (x, y) positions.
    """
    print(query_times.min(), query_times.max())
    print(position_times[0], position_times[-1])

    position_times = np.asarray(position_times)
    assert np.all(
        np.diff(position_times) > 0
    ), "position_times not strictly increasing"

    interp_x = np.interp(query_times, position_times, position_x)
    interp_y = np.interp(query_times, position_times, position_y)

    return np.column_stack((interp_x, interp_y))