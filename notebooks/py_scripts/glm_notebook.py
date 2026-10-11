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
#     display_name: nemos (3.10.14.final.0)
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
import spyglass.linearization.v1 as sgpl

from mpfc_opto.behavior.forktrack_tables import (
    ForkTrackParams,
    ForkTrackSelection,
    ForkTrackEvents,
)
from mpfc_opto.glm.path_progression_tables import (
    PathProgressSelection,
    PathProgress,
)
from mpfc_opto.glm.glm_tables import GLMSelection, GLMStorage
# -

from mpfc_opto.glm.basis import GLMBasisParams, GLMBasisSelection, GLMBasis

nwb_file_name = "Caius20260623_.nwb"
epoch = 6

GLM_storage = (GLMStorage() & {"epoch": epoch}).fetch_nwb()[0]
GLM_storage_df = GLM_storage["trial"]

GLM_storage_df

GLMBasisParams()

# +
key = (GLMStorage() & {"nwb_file_name": "Caius20260623_.nwb"}).fetch("KEY")[0]

key.update({"basis_param_id": 1, "basis_param_name": "default_basis"})

key
# -

GLMBasisSelection().insert1(key)

GLMBasisSelection()

GLMBasis()

GLMBasis().populate()
