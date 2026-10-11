# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py_scripts//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: spyglass_py310
#     language: python
#     name: python3
# ---

# +
import datajoint as dj

dj.config.load("dj_local_conf.json")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1

from spyglass.position.position_merge import PositionOutput
import spyglass.lfp as lfp
from spyglass.lfp.analysis.v1 import lfp_band

from mpfc_opto.sleep.updown_tables import (
    UpDownStateParams,
    UpDownStateSelection,
    UpDownStates,
)
from mpfc_opto.sleep.sleep_table import (
    SleepScoringParams,
    SleepScoringSelection,
    SleepScoring,
)
# -

# Before we look and UP/DOWN states, we need to verify that we have NREM sleep times and that these times are using cortical data

nwb_file_name = "Charlie20260223_.nwb"  # this is Charlie's first day on the w-track with ontime opto stim
lfp_electrode_group_name = (
    "left and right mPFC"  # since we plan on using this for cortical analyses
)
interval_list_name = (
    "05_s3"  # this is the second sleep session with opto and third overall
)

lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode() & {
    "nwb_file_name": nwb_file_name,
    "lfp_electrode_group_name": lfp_electrode_group_name,
}

# +
electrodes = (
    lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode()
    & {
        "nwb_file_name": nwb_file_name,
        "lfp_electrode_group_name": lfp_electrode_group_name,
    }
).fetch("electrode_id")

electrodes

# these look like they correspond to shanks 6-11 on the animal,
# which are the best mPFC probes

# +
lfp_filter = "LFP 0-400 Hz DMR"
delta_filter = "Delta 0.5-4 Hz DMR"
theta_filter = "Theta 5-11 Hz DMR"

sampling_rate = 1034

pos_merge_id = (
    PositionOutput.DLCPosV1() & {"nwb_file_name": nwb_file_name, "epoch": 5}
).fetch("merge_id")[
    0
]  # epoch has to match interval list name

lfp_s_key = {
    "nwb_file_name": nwb_file_name,
    "lfp_electrode_group_name": lfp_electrode_group_name,
    "target_interval_list_name": interval_list_name,
    "filter_name": lfp_filter,
    "target_sampling_rate": sampling_rate,
}

lfp_merge_id = (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1("merge_id")
# -

key = [
    {
        "sleep_scoring_params_name": "hierarchical",
        "lfp_merge_id": lfp_merge_id,
        "nwb_file_name": nwb_file_name,
        "filter_sampling_rate": sampling_rate,
        "target_interval_list_name": interval_list_name,
        "pos_merge_id": pos_merge_id,
        "theta_filter_name": theta_filter,
        "delta_filter_name": delta_filter,
    }
]

SleepScoring()

SleepScoring() & key

# All the data looks good for this one session and we have a substantial amount of NREM time in this sleep session, so we can move ahead with UP/DOWN state detetction. First, let us look at the parameters.

UpDownStateParams()

# We can add more parameters with different methods, such as the hilbert method.

key = {
    "up_down_params_name": "default_hilbert",
    "method": "hilbert",
    "so_smoothing": 0.1,
    "mua_smoothing": 0.05,
    "lfp_down_percentile": 25.0,
    "mua_down_percentile": 25.0,
    "min_down_duration": 0.1,
    "min_up_duration": 0.1,
    "lfp_weight": 0.5,
    "mua_weight": 0.5,
}

UpDownStateParams().insert1(key, skip_duplicates=True)
UpDownStateParams()

# Like in Sleep Scoring, we select the data we want to process and put it in a Selection table

UpDownStateSelection()

# +
# Let's make sure we have some mPFC spiking data for the MUA portion
import spyglass.spikesorting.v1 as sgs

sgs.SpikeSortingRecordingSelection() << {
    "nwb_file_name": "Charlie20260223_.nwb"
}

# here I am looking at the recording ids to see which shanks I have preprocessed for spike sorting
# in charlie, shanks 4-11 are in the mPFC (4-7 in the left, 8-11 in the right)

# +
list_of_shanks = [4, 5, 6, 7, 8, 9, 10, 11]

group_keys = []

for i in list_of_shanks:
    key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": i,
        "preproc_param_name": "default",
        "interval_list_name": "raw data valid times",
        "team_name": "Denisse Morales-Rodriguez",
    }

    ssr_key = {
        "recording_id": (sgs.SpikeSortingRecordingSelection() & key).fetch1(
            "recording_id"
        ),
    } | key

    group_keys.append(ssr_key)

# +
# now I am only looking at mPFC entries

recording_ids = [str(gk["recording_id"]) for gk in group_keys]

selection = sgs.SpikeSortingSelection & [
    {"recording_id": str(gk["recording_id"])} for gk in group_keys
]

selection

# its kind of mess, data still processing, will get better by next week...

# +
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

spikesorting_merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
    {"nwb_file_name": nwb_file_name, "curation_id": 0},
    sources=["v1"],
)

keys = [{"merge_id": merge_id} for merge_id in spikesorting_merge_ids]
(SpikeSortingOutput.CurationV1 & keys)

# +
# I've made a sortedspikesgroup with just the one mPFC probe that has been fully probessed and
# inserted into the SpikeSortingOutput table

from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

group_key = {
    "nwb_file_name": nwb_file_name,
    "sorted_spikes_group_name": "single_mPFC_probe",
}
SortedSpikesGroup & group_key

# +
# we can now put the sortedspikes group into our selection table with our lfp and sleep scoring info

# lfp is from looking at sleep scoring

key = {
    "up_down_params_name": "default_hilbert",
    "sleep_scoring_params_name": "hierarchical",
    "nwb_file_name": nwb_file_name,
    "target_interval_list_name": interval_list_name,
    "theta_filter_name": theta_filter,
    "delta_filter_name": delta_filter,
    "unit_filter_params_name": "all_units",
    "sorted_spikes_group_name": "single_mPFC_probe",
    "lfp_merge_id": lfp_merge_id,
}
# -

UpDownStateSelection().insert1(key, skip_duplicates=True)

UpDownStateSelection()

UpDownStates.populate(key)

UpDownStates()

# +
# again, we can delete if needed
# (UpDownStates & key).delete()
# -

# Default: plots first 60 seconds
fig = (UpDownStates & key).plot_up_down_states()

# +
import spyglass.common as sgc

times = (
    sgc.IntervalList()
    & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
).fetch("valid_times")[0]
# -

# Or specify a time window
(UpDownStates & key).plot_up_down_states(
    t_start=times[0][0] + 1098, t_stop=times[0][0] + 1099.5
)
