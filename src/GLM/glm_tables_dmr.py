"""
Spyglass GLM Covariate Pipeline

Author: DMR
Date: July 2026
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
from GLM.path_progression_tables_dmr import PathProgressSelection, PathProgress
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from behavior.forktrack_tables_dmr import ForkTrackEvents
import spyglass.linearization.v1 as sgpl

schema = dj.schema("denissemorales_glm")

@schema
class GLMSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Nwbfile
    -> PathProgress
    -> SortedSpikesGroup
    -> PositionOutput
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
        """Fetch all upstream data

        Returns
        -------
        list
            [nwb_file_name, epoch, sorted_spikes, position, path_progress]
        """
        selection = (GLMSelection & key).fetch1()
        epoch = selection["epoch"]

        nwb_file_name = (Nwbfile & key).fetch1("nwb_file_name")
        nwb_path = Nwbfile().get_abs_path(nwb_file_name)

        path_progress_df = (
            (PathProgress & {"merge_id": key["merge_id"]})
            .fetchnwb()
            .reset_index()
        )

        pos = (PositionOutput & {"merge_id": selection['merge_id']}).fetch1_dataframe()

        forktrack_results = pd.DataFrame((ForkTrackEvents() &{'nwb_file_name': nwb_file_name, 'epoch': selection["epoch"]}).fetch('forktrack_results')[0])

        t_start = (IntervalList() & {'nwb_file_name': nwb_file_name,
                      'interval_list_name': selection['interval_list_name']}).fetch('valid_times')[0][0][0]

        t_end = (IntervalList() & {'nwb_file_name': nwb_file_name,
                      'interval_list_name': selection['interval_list_name']}).fetch('valid_times')[0][0][1]

        bin_size = 0.05  # 50 ms
        time = np.arange(t_start, t_end, bin_size)

        spike_counts = SortedSpikesGroup.get_spike_indicator(
            selection['merge_id'],
            time=time,
        )

        path_progress_df = (PathProgress()& {'epoch': selection['epoch']}).fetch_nwb()
        entry = path_progress_df[0]
        path_progress_df = entry["trial"]

        return [
            nwb_file_name,
            nwb_path,
            epoch,
            forktrack_results,
            spike_counts,
            path_progress_df,
            pos
        ]

    def make_compute(
        nwb_file_name,
        nwb_path,
        epoch,
        forktrack_results,
        spike_counts,
        path_progress_df,
        pos
    ):
        """Bin everything into 50ms bins for GLM analysis

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

        forktrack_results['home_arm'] = 'handle'

        # One-hot encode 'trajectory'
        forktrack_results = pd.get_dummies(forktrack_results, columns=['trajectory'], prefix='trajectory')

        # Previous row's prev_well (stays categorical, not one-hot)
        forktrack_results['2_prev_wells'] = forktrack_results['prev_well'].shift(1).astype('category')

        # Previous row's pump_triggered -> previous_reward
        forktrack_results['previous_reward'] = forktrack_results['pump_triggered'].shift(1).fillna(False).astype(bool)

        # Current row's pump_triggered -> current_reward
        forktrack_results['current_reward'] = forktrack_results['pump_triggered'].astype(bool)

        rel_position = pos[['position_x', 'position_y', 'speed', 'orientation']]




        return [nwb_file_name, final_df,epoch,interval_list_name]

    def make_insert(self, key, nwb_file_name, final_df):
        """Write results to NWB and insert into PathProgress.

        Parameters
        ----------
        key : dict
            Primary key to PathProgressSelection.
        nwb_file_name : str
            Name of the NWB file.
        final_df : pd.DataFrame
            Trial-level DataFrame produced by make_compute.
        """

        with AnalysisNwbfile().build(nwb_file_name) as builder:
            obj_id = builder.add_nwb_object(final_df)
            analysis_file = builder.analysis_file_name

        self.insert1(
            dict(
                **key,
                analysis_file_name=analysis_file,
                trial_object_id=obj_id,
            )
        )


def shortest_path(start, end):
    # simple BFS (tree, so path is unique anyway)
    from collections import deque
    visited = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == end:
            break
        for nbr in adjacency[node]:
            if nbr not in visited:
                visited[nbr] = node
                q.append(nbr)
    path = [end]
    while visited[path[-1]] is not None:
        path.append(visited[path[-1]])
    return path[::-1]

def turn_direction(prev_node, junction_node, next_node):
    v_in = node_positions[junction_node] - node_positions[prev_node]
    v_out = node_positions[next_node] - node_positions[junction_node]
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    if cross > 0:
        return 'right'
    elif cross < 0:
        return 'left'
    return 'straight'

def get_turns_for_trajectory(trajectory):
    """Map segment_id -> upcoming turn direction for a given trajectory string."""
    start_name, end_name = trajectory.split('_to_')
    path = shortest_path(well_name_to_node[start_name], well_name_to_node[end_name])
    turns = {}
    for i in range(1, len(path) - 1):
        prev_node, junction_node, next_node = path[i - 1], path[i], path[i + 1]
        seg_id = edge_to_segment_id[(prev_node, junction_node)]
        turns[seg_id] = turn_direction(prev_node, junction_node, next_node)
    return turns