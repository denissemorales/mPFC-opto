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

# Import packages + spyglass + custom tables

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

# SleepScoring has a parameter table. We can look at the existing parameter options, and also add parameters we would like to use for sleep scoring using the ```insert``` function. Importantly, sleep_scoring_params_name must be unique.

SleepScoringParams()

key = {
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

SleepScoringParams().insert1(key, skip_duplicates=True)

# We then have a selection table, where we can select the data we want to sleep score.

SleepScoringSelection()

# First, we get the data we are interested in

lfp.LFPOutput.LFPV1() & {"nwb_file_name": "Charlie20260223_.nwb"}

# +
nwb_file_name = "Charlie20260223_.nwb"  # this is Charlie's first day on the w-track with ontime opto stim
lfp_electrode_group_name = "left and right mPFC"  # since we plan on using this for more cortical analyses
interval_list_name = (
    "05_s3"  # this is the second sleep session with opto and third overall
)

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

# +
# Again, we can insert table entries with the insert function.

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
SleepScoringSelection().insert(key, skip_duplicates=True)
# -

# Now we can look to verify our entry

SleepScoringSelection()

SleepScoringSelection() & key

# +
# if the entry is incorrect, you can uncomment the line below and it will delete the
# selection entry and any SleepScoring entry

# (SleepScoringSelection() & key).delete()
# -

# After verifying our entry, we can now run sleep scoring

SleepScoring().populate(key)

# We can verify our table entry

SleepScoring() & key

# We can restrict to just the session of interest with a few identifiers and then use the table results to analyze data (e.g. visualizing the sleep scoring results, getting the nrem times for downstream analysis, etc.)

(
    SleepScoring()
    & {
        "nwb_file_name": "Charlie20260223_.nwb",
        "target_interval_list_name": "05_s3",
        "lfp_merge_id": lfp_merge_id,
    }
).plot_hypnogram()

(
    SleepScoring()
    & {
        "nwb_file_name": "Charlie20260223_.nwb",
        "target_interval_list_name": "05_s3",
        "lfp_merge_id": lfp_merge_id,
    }
).fetch_nrem_times()
