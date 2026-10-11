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
from spyglass.linearization.merge import LinearizedPositionOutput
import spyglass.common as sgc
import pandas as pd

from mpfc_opto.behavior.forktrack_tables import (
    ForkTrackParams,
    ForkTrackSelection,
    ForkTrackEvents,
)
from mpfc_opto.glm.path_progression_tables import (
    PathProgressSelection,
    PathProgress,
)

# +
nwb_file_name = "Caius20260623_.nwb"
epoch = 4
track_graph_name = "dmr fork caius"

pos_key = {
    "nwb_file_name": nwb_file_name,
    "epoch": epoch,
}

pos_merge_id = (PositionOutput.DLCPosV1() & pos_key).fetch1("merge_id")

linear_key = {
    "pos_merge_id": pos_merge_id,
    "linearization_param_name": "default",
}

linear_merge_key = LinearizedPositionOutput.merge_restrict(linear_key).fetch1(
    "KEY"
)

key = {
    "nwb_file_name": nwb_file_name,
    "pos_merge_id": pos_merge_id,
    "merge_id": linear_merge_key["merge_id"],
    "forktrack_params_name": "default",
    "track_graph_name": track_graph_name,
    "epoch": epoch,
}

PathProgressSelection.insert_selection(
    key,
    left=(75, 18),
    right=(15.6, 55),
    center=(49, 34),
    handle=(129.4, 160.5),
)
# -

PathProgressSelection()

PathProgress().populate()

(PathProgress() & {"nwb_file_name": nwb_file_name}).fetch(
    "pathprogress_results"
)
