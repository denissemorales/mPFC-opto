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
from

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

        return [
            nwb_file_name,
            nwb_path,
            epoch,
            forktrack_results,
            spike_counts,
            path_progress_df
        ]

    def make_compute(
        nwb_file_name,
        nwb_path,
        epoch,
        forktrack_results,
        spike_counts,
        path_progress_df
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
