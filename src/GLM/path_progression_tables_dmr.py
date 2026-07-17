"""
Path Progress Tables

Author: DMR
Date: July 2026
"""

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from spyglass.common import Nwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils import SpyglassMixin
from neurospatial.behavior.navigation import path_progress
from neurospatial import Environment
import spyglass.linearization.v1 as sgpl
from spyglass.linearization.merge import LinearizedPositionOutput
from behavior.forktrack_tables_dmr import ForkTrackEvents

schema = dj.schema("denissemorales_pathprogress")

@schema
class PathProgressSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Nwbfile
    -> LinearizedPositionOutput
    -> ForkTrackEvents
    -> sgpl.TrackGraph
    ---
    epoch: int
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    center_x: float
    center_y: float
    handle_x: float
    handle_y: float
    """

    @classmethod
    def insert_selection(cls, key, left, right, center, handle, skip_duplicates=False):
        """Insert a PathProgressSelection entry with explicit well positions.

        Parameters
        ----------
        key : dict
            Primary key fields for PathProgressSelection (nwb_file_name,
            pos_merge_id, forktrack key fields, track_graph_name) plus
            'epoch'.
        left, right, center, handle : tuple/list/array of (x, y)
            2D coordinates for each well/arm.
        skip_duplicates : bool, optional
            Passed through to `insert1`.
        """
        insert_key = dict(key)
        insert_key.update(
            left_x=left[0], left_y=left[1],
            right_x=right[0], right_y=right[1],
            center_x=center[0], center_y=center[1],
            handle_x=handle[0], handle_y=handle[1],
        )
        cls.insert1(insert_key, skip_duplicates=skip_duplicates)

@schema
class PathProgress(SpyglassMixin, dj.Computed):
    definition = """
    -> PathProgressSelection
    ---
    epoch: int
    pathprogress_results: blob
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make_fetch(self, key):
        """Fetch all upstream data needed to compute path progression.

        Returns
        -------
        list
            [nwb_file_name, nwb_path, epoch, linear_position_df, forktrack_results ]
        """
        selection = (PathProgressSelection & key).fetch1()
        epoch = selection["epoch"]

        nwb_file_name = (Nwbfile & key).fetch1("nwb_file_name")
        nwb_path = Nwbfile().get_abs_path(nwb_file_name)

        linear_position_df = (
            (LinearizedPositionOutput & {"merge_id": key["merge_id"]})
            .fetch1_dataframe()
            .reset_index()
        )

        forktrack_results = pd.DataFrame((ForkTrackEvents() &{'nwb_file_name': nwb_file_name, 'epoch': epoch}).fetch('forktrack_results')[0])

        track_graph_name = selection["track_graph_name"]
        track_graph = (sgpl.TrackGraph & {"track_graph_name": track_graph_name }).get_networkx_track_graph()
        edge_order = (sgpl.TrackGraph & {"track_graph_name": track_graph_name }).fetch1("linear_edge_order")
        edge_spacing = (sgpl.TrackGraph & {"track_graph_name": track_graph_name }).fetch1("linear_edge_spacing")
        edge_order = [(int(a), int(b)) for a, b in edge_order]

        env = Environment.from_graph(
        track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        bin_size=2.5,
        name=track_graph_name ,
        )

        well_positions = {
            "left": (selection["left_x"], selection["left_y"]),
            "right": (selection["right_x"], selection["right_y"]),
            "center": (selection["center_x"], selection["center_y"]),
            "handle": (selection["handle_x"], selection["handle_y"]),
        }

        return [
            nwb_file_name,
            nwb_path,
            epoch,
            linear_position_df,
            forktrack_results,
            env,
            well_positions,
        ]

    def make_compute(
        self,
        key,
        nwb_file_name,
        nwb_path,
        epoch,
        linear_position_df,
        forktrack_results,
        env,
        well_positions,
    ):
        """Compute Path Progress.

        Parameters
        ----------
        key : dict
            Primary key to PathProgressin.
        nwb_file_name : str
            Name of the NWB file.
        epoch : int
            epoch number.
        linear_position:
            linear position
        fork_track_results:
            fork_track_results

        Returns
        -------
        list
            [nwb_file_name, final_df, epoch]
        """
        forktrack_results["trajectory"] = forktrack_results.apply(infer_trajectory, axis=1)

        linear_position_df = pd.merge_asof(
        linear_position_df,
        forktrack_results[["time", "trajectory"]].sort_values("time"),
        on="time",
        direction="backward"
         )

        print(linear_position_df["trajectory"].value_counts(dropna=False))

        position_info = linear_position_df

        goal_positions_2d = {
            "left_arm":  np.array([well_positions["left"]]),
            "right_arm": np.array([well_positions["right"]]),
            "center":    np.array([well_positions["center"]]),
            "handle":    np.array([well_positions["handle"]]),
        }

        # Convert 2D positions to bin indices
        start_bin = env.bin_at(np.array([well_positions["handle"]]))[0]  # home
        left_bin  = env.bin_at(np.array([well_positions["left"]]))[0]
        right_bin = env.bin_at(np.array([well_positions["right"]]))[0]
        center_bin = env.bin_at(np.array([well_positions["center"]]))[0]

        print(start_bin, left_bin, right_bin, center_bin)  # verify these are valid (>= 0)

        _orig_to_scipy_sparse_array = nx.to_scipy_sparse_array

        def _patched_to_scipy_sparse_array(G, weight=None, format="csr"):
            A = _orig_to_scipy_sparse_array(G, weight=weight, format=format)
            A.indices = A.indices.astype(np.int32)
            A.indptr = A.indptr.astype(np.int32)
            return A

        nx.to_scipy_sparse_array = _patched_to_scipy_sparse_array

        trajectory_goals = {
            "handle_to_left":  (left_bin, start_bin),
            "left_to_handle":  (start_bin,  left_bin),
            "handle_to_right": (right_bin, start_bin),
            "right_to_handle": (start_bin, right_bin),
            "left_to_right":   (left_bin,  right_bin),
            "right_to_left":   (right_bin, left_bin),
        }

        xy_cols = ["projected_x_position", "projected_y_position"]
        finite_mask = np.isfinite(position_info[xy_cols].values).all(axis=1)
        print(f"Dropping {(~finite_mask).sum()} of {len(position_info)} rows")

        position_info = position_info.loc[finite_mask].reset_index(drop=True)
        linear_position_df = linear_position_df.loc[finite_mask].reset_index(drop=True)

        position_bins = env.bin_at(position_info[xy_cols].values).astype(int)

        start_bins = np.full(len(position_bins), -1)
        goal_bins  = np.full(len(position_bins), -1)

        for traj, (s, g) in trajectory_goals.items():
            mask = (linear_position_df["trajectory"] == traj).values
            start_bins[mask] = s
            goal_bins[mask]  = g

        progress = path_progress(
            position_bins,
            env,
            start_bins=start_bins,
            goal_bins=goal_bins,
        )
        linear_position_df["trajectory_progress"] = progress

        # NOTE: previously this block appended to an undefined `final_rows`
        # list with a syntax error (`time=` had no value) and only ever
        # produced a single row with no real data. Replaced with the
        # per-timepoint frame the plotting code below actually needs.
        final_df = linear_position_df.copy()
        final_df["epoch"] = epoch + 1

        plt.figure(figsize=(12, 4))
        for traj, g in linear_position_df.dropna(subset=["trajectory_progress"]).groupby("trajectory"):
            plt.scatter(g["time"], g["trajectory_progress"], s=4, label=traj)

        plt.xlabel("Time")
        plt.ylabel("path progress")
        plt.title("path progress")
        plt.ylim(-0.05, 1.05)
        plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        return [nwb_file_name, final_df,epoch,]

    def make_insert(self, key, nwb_file_name, final_df, epoch):
        """Write results to NWB and insert into PathProgress.

        Parameters
        ----------
        key : dict
            Primary key to PathProgressSelection.
        nwb_file_name : str
            Name of the NWB file.
        final_df : pd.DataFrame
            Trial-level DataFrame produced by make_compute.
        epoch : int
            Zero-indexed epoch number (will be stored as epoch + 1).
        """

        with AnalysisNwbfile().build(nwb_file_name) as builder:
            final_df = final_df.copy()
            for col in final_df.columns:
                if pd.api.types.is_numeric_dtype(final_df[col]):
                    final_df[col] = final_df[col].fillna(np.nan)
                else:
                    final_df[col] = final_df[col].fillna("none").astype(str)
            obj_id = builder.add_nwb_object(final_df)
            analysis_file = builder.analysis_file_name

        self.insert1(
            dict(
                **key,
                epoch=epoch,
                pathprogress_results=final_df.to_dict("records"),
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

def normalize_well(name):
    if pd.isna(name):
        return None
    name = str(name).replace("_poke", "").lower()
    if name in {"left", "right", "center", "handle"}:
        return name
    return name

def infer_trajectory(row):
    prev_well = normalize_well(row["prev_well"])
    well = normalize_well(row["well_name"])
    trial_type = str(row["trial_type"]).lower() if not pd.isna(row["trial_type"]) else ""

    # center-out
    if prev_well == "center" and well == "left":
        return "center_to_left"
    if prev_well == "center" and well == "right":
        return "center_to_right"
    if prev_well == "center" and well == "handle":
        return "center_to_handle"

    # side-in
    if prev_well == "left" and well == "center":
        return "left_to_center"
    if prev_well == "right" and well == "center":
        return "right_to_center"
    if prev_well == "left" and well == "handle":
        return "left_to_handle"
    if prev_well == "right" and well == "handle":
        return "right_to_handle"


    # if the row is side-to-side, keep it separate or ignore it
    if prev_well == "left" and well == "right":
        return "left_to_right"
    if prev_well == "right" and well == "left":
        return "right_to_left"

    if prev_well == "handle" and well == "right":
        return "handle_to_right"
    if prev_well == "handle" and well == "left":
        return "handle_to_left"
    if prev_well == "handle" and well == "center":
        return "handle_to_center"


    return np.nan