"""
Spyglass GLM Covariate Pipeline

Bins mPFC spiking, 2D position, path progression, path type, previous arm,
home arm, running speed, upcoming/previous turn, and previous/current
trial reward into 50 ms time bins for GLM analysis.

Author: DMR
Date: July 2026
"""

from collections import deque

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spyglass.common import IntervalList, Nwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.position.position_merge import PositionOutput
from spyglass.linearization.merge import LinearizedPositionOutput
from spyglass.utils import SpyglassMixin
from GLM.path_progression_tables_dmr import PathProgressSelection, PathProgress
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from behavior.forktrack_tables_dmr import ForkTrackEvents
import spyglass.linearization.v1 as sgpl

schema = dj.schema("denissemorales_glm")

BIN_SIZE = 0.05  # 50 ms

WELL_NAME_TO_NODE = {"left": 0, "center": 1, "right": 2, "handle": 6}

@schema
class GLMSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Nwbfile
    -> PathProgress
    -> SortedSpikesGroup
    -> PositionOutput.proj(pos_merge_id='merge_id')
    -> LinearizedPositionOutput.proj(lin_merge_id='merge_id')
    -> ForkTrackEvents
    -> IntervalList
    -> sgpl.TrackGraph
    """

@schema
class GLMStorage(SpyglassMixin, dj.Computed):
    definition = """
    -> GLMSelection
    ---
    epoch: int
    -> AnalysisNwbfile
    trial_object_id: varchar(40)
    """

    def make_fetch(self, key):
        """Fetch all upstream data needed to build the 50 ms covariate table.

        Returns
        -------
        list
            [nwb_file_name, epoch, interval_list_name, forktrack_results,
             spike_counts, path_progress_df, pos, node_positions, edges,
             time_bins]
        """
        selection = (GLMSelection & key).fetch1()
        epoch = selection["epoch"]
        interval_list_name = selection["interval_list_name"]

        nwb_file_name = (Nwbfile & key).fetch1("nwb_file_name")

        pos = (
            PositionOutput & {"merge_id": selection["pos_merge_id"]}
        ).fetch1_dataframe()

        forktrack_results = pd.DataFrame(
            (
                ForkTrackEvents()
                & {"nwb_file_name": nwb_file_name, "epoch": epoch}
            ).fetch("forktrack_results")[0]
        )

        valid_times = (
            IntervalList()
            & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
        ).fetch1("valid_times")
        t_start, t_end = valid_times[0][0], valid_times[0][1]
        time_bins = np.arange(t_start, t_end, BIN_SIZE)

        spike_counts = SortedSpikesGroup.get_spike_indicator(
            {'nwb_file_name': nwb_file_name,
            'unit_filter_params_name': selection['unit_filter_params_name'],
            'sorted_spikes_group_name': selection['sorted_spikes_group_name']},
            time=time_bins,
        )

        path_progress_entry = (PathProgress() & {"epoch": epoch}).fetch_nwb()[0]
        path_progress_df = path_progress_entry["trial"]

        node_positions = (
            sgpl.TrackGraph() & {"track_graph_name": selection["track_graph_name"]}
        ).fetch1("node_positions")
        edges = (
            sgpl.TrackGraph() & {"track_graph_name": selection["track_graph_name"]}
        ).fetch1("edges")

        return [
            nwb_file_name,
            epoch,
            interval_list_name,
            forktrack_results,
            spike_counts,
            path_progress_df,
            pos,
            node_positions,
            edges,
            time_bins,
        ]

    def make_compute(
        self,
        key,
        nwb_file_name,
        epoch,
        interval_list_name,
        forktrack_results,
        spike_counts,
        path_progress_df,
        pos,
        node_positions,
        edges,
        time_bins,
    ):
        """Bin everything into 50 ms bins for GLM analysis.

        Returns
        -------
        list
            [nwb_file_name, final_df, epoch, interval_list_name]
        """
        adjacency, edge_to_segment_id = build_adjacency(edges)
        forktrack_results = forktrack_results.copy()
        forktrack_results["home_arm"] = "handle"
        forktrack_results["previous_arm"] = forktrack_results["prev_well"].shift(1)
        forktrack_results["previous_reward"] = (
            forktrack_results["pump_triggered"].shift(1).fillna(False).astype(bool)
        )
        forktrack_results["current_reward"] = forktrack_results["pump_triggered"].astype(bool)
        forktrack_results["trajectory"] = derive_trajectory_column(forktrack_results)

        unique_trajectories = [
            t
            for t in pd.concat(
                [forktrack_results["trajectory"], path_progress_df["trajectory"]]
            )
            .dropna()
            .unique()
            if t != "none"
        ]
        upcoming_turn_lookup, overall_turn_lookup = build_trajectory_turn_lookups(
            unique_trajectories, node_positions, adjacency, edge_to_segment_id
        )

        forktrack_results["turn"] = forktrack_results["trajectory"].map(overall_turn_lookup)
        forktrack_results["previous_turn"] = forktrack_results["turn"].shift(1)

        forktrack_results = pd.get_dummies(
            forktrack_results, columns=["trajectory"], prefix="path_type"
        )

        path_progress_df = path_progress_df.copy()

        def _upcoming_turn(row):
            traj = row["trajectory"]
            if pd.isna(traj) or traj == "none":
                return None
            return upcoming_turn_lookup.get(traj, {}).get(row["track_segment_id"])

        path_progress_df["upcoming_turn"] = path_progress_df.apply(_upcoming_turn, axis=1)

        path_type_dummies = pd.get_dummies(path_progress_df["trajectory"], prefix="path_type")
        path_progress_df = pd.concat([path_progress_df, path_type_dummies], axis=1)

        # ---- assemble the 50 ms time base ----
        binned = pd.DataFrame({"time": time_bins})

        # spike_counts: (n_time, n_units) from get_spike_indicator
        n_units = spike_counts.shape[1]
        spike_df = pd.DataFrame(spike_counts, columns=[f"unit_{i}" for i in range(n_units)])
        spike_df["time"] = time_bins
        binned = binned.merge(spike_df, on="time", how="left")

        # position: nearest-sample alignment (index is already named "time")
        pos_reset = pos.reset_index()
        binned = pd.merge_asof(
            binned.sort_values("time"),
            pos_reset[["time", "position_x", "position_y", "speed", "orientation"]]
            .sort_values("time"),
            on="time",
            direction="nearest",
        )

        # continuous path progression: nearest-sample alignment
        path_cols = ["time", "linear_position", "track_segment_id", "upcoming_turn"] + list(
            path_type_dummies.columns
        )
        binned = pd.merge_asof(
            binned.sort_values("time"),
            path_progress_df[path_cols].sort_values("time"),
            on="time",
            direction="nearest",
        )

        # trial-level info: carry forward the most recently STARTED trial
        trial_cols = [
            "time",
            "previous_arm",
            "home_arm",
            "previous_reward",
            "current_reward",
            "turn",
            "previous_turn",
        ]
        binned = pd.merge_asof(
            binned.sort_values("time"),
            forktrack_results[trial_cols].sort_values("time"),
            on="time",
            direction="backward",
        )

        binned["nwb_file_name"] = nwb_file_name
        binned["epoch"] = epoch
        nullable_string_columns = [
            "previous_arm",
            "home_arm",
            "turn",
            "previous_turn",
            "upcoming_turn",
        ]
        for col in nullable_string_columns:
            if col in binned.columns:
                binned[col] = binned[col].fillna("none").astype(str)

        return [nwb_file_name, binned, epoch, interval_list_name]

    def make_insert(self, key, nwb_file_name, final_df, epoch, interval_list_name):
        """Write results to NWB and insert into GLMStorage."""
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(final_df)
            analysis_file = builder.analysis_file_name

        self.insert1(
            dict(
                **key,
                epoch=epoch,
                analysis_file_name=analysis_file,
                trial_object_id=obj_id,
            )
        )



def build_adjacency(edges):
    """Build an undirected adjacency list + (node, node) -> segment_id lookup."""
    n_nodes = int(np.max(edges)) + 1
    adjacency = {i: [] for i in range(n_nodes)}
    edge_to_segment_id = {}
    for seg_id, (a, b) in enumerate(edges):
        a, b = int(a), int(b)
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_to_segment_id[(a, b)] = seg_id
        edge_to_segment_id[(b, a)] = seg_id
    return adjacency, edge_to_segment_id


def shortest_path(adjacency, start, end):
    """BFS shortest path between two nodes (unique, since the track is a tree)."""
    visited = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)
    if end not in visited:
        raise ValueError(f"No path found between node {start} and node {end}")
    path = [end]
    while visited[path[-1]] is not None:
        path.append(visited[path[-1]])
    return path[::-1]


def turn_direction(node_positions, prev_node, junction_node, next_node):
    """Sign of the cross product of incoming/outgoing direction vectors.

    Sign convention (positive cross -> 'right') must be validated against
    this track's real coordinate system before trusting downstream output.
    See validation notes in the accompanying message.
    """
    v_in = node_positions[junction_node] - node_positions[prev_node]
    v_out = node_positions[next_node] - node_positions[junction_node]
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    if cross > 0:
        return "right"
    elif cross < 0:
        return "left"
    return "straight"


def get_turns_for_trajectory(trajectory, node_positions, adjacency, edge_to_segment_id):
    """Map segment_id -> upcoming turn direction, for every segment on the path."""
    start_name, end_name = trajectory.split("_to_")
    path = shortest_path(
        adjacency, WELL_NAME_TO_NODE[start_name], WELL_NAME_TO_NODE[end_name]
    )
    turns = {}
    for i in range(1, len(path) - 1):
        prev_node, junction_node, next_node = path[i - 1], path[i], path[i + 1]
        seg_id = edge_to_segment_id[(prev_node, junction_node)]
        turns[seg_id] = turn_direction(node_positions, prev_node, junction_node, next_node)
    return turns


def get_overall_turn_for_trajectory(trajectory, node_positions, adjacency, edge_to_segment_id):
    """The single defining turn direction of a trajectory (used for previous_turn).

    Takes the turn at the LAST junction before reaching the destination well.
    If your task cares about the *first* junction instead, swap path[-3:] for
    path[0:3] below.
    """
    start_name, end_name = trajectory.split("_to_")
    path = shortest_path(
        adjacency, WELL_NAME_TO_NODE[start_name], WELL_NAME_TO_NODE[end_name]
    )
    if len(path) < 3:
        return "straight"
    prev_node, junction_node, next_node = path[-3], path[-2], path[-1]
    return turn_direction(node_positions, prev_node, junction_node, next_node)


def build_trajectory_turn_lookups(trajectories, node_positions, adjacency, edge_to_segment_id):
    """Precompute {trajectory: {segment_id: upcoming_turn}} and {trajectory: overall_turn}."""
    upcoming_by_traj = {}
    overall_by_traj = {}
    for traj in trajectories:
        upcoming_by_traj[traj] = get_turns_for_trajectory(
            traj, node_positions, adjacency, edge_to_segment_id
        )
        overall_by_traj[traj] = get_overall_turn_for_trajectory(
            traj, node_positions, adjacency, edge_to_segment_id
        )
    return upcoming_by_traj, overall_by_traj


def normalize_well_name(name):
    if name is None or (isinstance(name, float) and pd.isna(name)) or name == "":
        return None
    normalized = str(name).replace("_poke", "").replace("_Poke", "").lower()
    if normalized == "none":
        return None
    return normalized


def derive_trajectory_column(df, start_col="prev_well", end_col="well_name"):
    """Build a 'handle_to_left'-style trajectory string from two well-name
    columns, matching the format already used in path_progress_df['trajectory'].

    Returns None for rows where either endpoint is missing (e.g. the very
    first row of a session). Use this instead of forktrack_results'
    'transition' column, which is a different, incompatible string format
    (e.g. "Handle_poke\u2192Left_poke").
    """
    def _row_trajectory(row):
        start = normalize_well_name(row[start_col])
        end = normalize_well_name(row[end_col])
        if start is None or end is None:
            return None
        return f"{start}_to_{end}"

    return df.apply(_row_trajectory, axis=1)


def plot_track_graph_turns(node_positions, edges, trajectory=None, ax=None):
    """Plot the track graph with node/segment labels, optionally overlaid
    with the turn(s) implied by a single trajectory.

    Call this BEFORE trusting turn_direction's sign convention on a new
    track graph -- eyeball whether the arrows/labels match your mental
    model of the maze.

    Parameters
    ----------
    node_positions : array-like, shape (n_nodes, 2)
    edges : array-like, shape (n_edges, 2)
    trajectory : str, optional
        e.g. "left_to_handle". If given, highlights the path taken and
        annotates each junction with its computed turn direction.
    ax : matplotlib Axes, optional
    """
    node_positions = np.asarray(node_positions)
    edges = np.asarray(edges)
    adjacency, edge_to_segment_id = build_adjacency(edges)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # base graph: all nodes + all segments, labeled
    for seg_id, (a, b) in enumerate(edges):
        x1, y1 = node_positions[a]
        x2, y2 = node_positions[b]
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=2, zorder=1)
        ax.annotate(
            f"seg{seg_id}",
            ((x1 + x2) / 2, (y1 + y2) / 2),
            fontsize=9,
            color="gray",
            ha="center",
        )

    for i, (x, y) in enumerate(node_positions):
        ax.scatter(x, y, s=250, color="steelblue", zorder=3)
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(0, 0),
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=4,
        )

    if trajectory is not None:
        start_name, end_name = trajectory.split("_to_")
        path = shortest_path(
            adjacency, WELL_NAME_TO_NODE[start_name], WELL_NAME_TO_NODE[end_name]
        )
        turns = get_turns_for_trajectory(
            trajectory, node_positions, adjacency, edge_to_segment_id
        )

        # highlight the path itself
        path_x = node_positions[path, 0]
        path_y = node_positions[path, 1]
        ax.plot(path_x, path_y, color="crimson", linewidth=3, zorder=2)

        # annotate each junction with its turn
        for i in range(1, len(path) - 1):
            junction_node = path[i]
            seg_id = edge_to_segment_id[(path[i - 1], junction_node)]
            direction = turns[seg_id]
            jx, jy = node_positions[junction_node]
            ax.annotate(
                direction.upper(),
                (jx, jy),
                textcoords="offset points",
                xytext=(15, 15),
                fontsize=12,
                fontweight="bold",
                color="crimson",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="crimson"),
            )

        ax.set_title(f"Trajectory: {trajectory}  |  path nodes: {path}")
    else:
        ax.set_title("Track graph: node indices + segment ids")

    ax.set_aspect("equal")
    return ax
