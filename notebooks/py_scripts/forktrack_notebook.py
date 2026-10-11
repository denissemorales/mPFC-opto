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
#     display_name: spyglass_py310_new
#     language: python
#     name: python3
# ---

# +
import datajoint as dj

dj.config.load("dj_local_conf.json")

import numpy as np
from spyglass.position import PositionOutput
import spyglass.common as sgc
import pandas as pd
from mpfc_opto.behavior.forktrack_tables import (
    ForkTrackParams,
    ForkTrackSelection,
    ForkTrackEvents,
)
# -

ForkTrackParams().insert_default()

nwb_file_name = "Caius20260623_.nwb"

pos_key = {
    "nwb_file_name": nwb_file_name,
    "epoch": 10,
}
merge_id = (PositionOutput.DLCPosV1() & pos_key).fetch1("merge_id")

ForkTrackSelection()

ForkTrackSelection().insert1(
    {
        "forktrack_params_name": "default",
        "nwb_file_name": nwb_file_name,
        "pos_merge_id": merge_id,
        "epoch": 10,
        "statescript_path": "/stelmo/denisse/Caius/20260623/20260623_Caius_10_r5/20260623_Caius_10_r5.stateScriptLog",
    },
    skip_duplicates=True,
)

ForkTrackEvents().populate()

(ForkTrackEvents() & {"epoch": 8}).delete()

x = (ForkTrackEvents() & {"nwb_file_name": nwb_file_name}).fetch(
    "validation_report"
)
x

x = (ForkTrackEvents() & {"epoch": 2}).fetch("forktrack_results")
y = pd.DataFrame(x[0])
y
y

# REVIEW: pasted from the W-track notebook. This one is about the fork
# track and imports ForkTrackEvents, so WTrackEvents is undefined here
# and the cell raises NameError. Likely wants ForkTrackEvents /
# "forktrack_results".
x = (WTrackEvents() & {"epoch": 2}).fetch("wtrack_results")  # noqa: F821
y = pd.DataFrame(x[0])
y
y
