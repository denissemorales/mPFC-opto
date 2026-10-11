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
#     display_name: mPFC-opto
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
from mpfc_opto.behavior.wtrack_tables import (
    WTrackParams,
    WTrackSelection,
    WTrackEvents,
)
# -

WTrackParams().insert_default()

nwb_file_name = "Jacob20250618_.nwb"

pos_key = {
    "nwb_file_name": nwb_file_name,
    "epoch": 2,
    "project_name": "w-track_Jacob",
}
merge_id = (PositionOutput.DLCPosV1() & pos_key).fetch1("merge_id")

WTrackSelection().insert1(
    {
        "wtrack_params_name": "default",
        "nwb_file_name": "Jacob20250618_.nwb",
        "pos_merge_id": merge_id,
        "epoch": 2,
        "statescript_path": "/stelmo/denisse/Jacob/raw/20250618/20250618_Jacob_02_r1/20250618_Jacob_02_r1.stateScriptLog",
    },
    skip_duplicates=True,
)

WTrackSelection()

key = {
    "wtrack_params_name": "default",
    "nwb_file_name": "Jacob20250618_.nwb",
    "pos_merge_id": merge_id,
    "epoch": 2,
    "statescript_path": "/stelmo/denisse/Jacob/raw/20250618/20250618_Jacob_02_r1/20250618_Jacob_02_r1.stateScriptLog",
}

WTrackEvents().populate(key)

WTrackEvents()

x = (WTrackEvents() & {"nwb_file_name": nwb_file_name}).fetch(
    "validation_report"
)
x

x = (WTrackEvents() & {"epoch": 2}).fetch("wtrack_results")
y = pd.DataFrame(x[0])
y
y

x = (WTrackEvents() & {"epoch": 2}).fetch("wtrack_results")
y = pd.DataFrame(x[0])
y
y
