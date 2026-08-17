"""
Spyglass GLM Basis Pipeline

Uses Nemos, based on Shih-Yi code.

Author: DMR
Date: August 2026

This is a single-file version: the basis-construction functions that used to live
in a separate `build_bases_new_schema.py` (`get_unit_columns`, `build_bases`,
`process_design_matrix`, and their helpers) are now defined directly below, before
the DataJoint table definitions that call them.

Tables (Params -> Selection -> Computed "tripartite" pattern):
  GLMBasisParams     (dj.Lookup)   -- basis-construction hyperparameters
  GLMBasisSelection  (dj.Manual)   -- GLMStorage entry x GLMBasisParams
  GLMBasis           (dj.Computed) -- builds + stores the design matrix via the
                                       tri-part make pattern (make_fetch /
                                       make_compute / make_insert, _parallel_make = True)

NOTE: dropped two imports that weren't referenced anywhere in the previous version
of this file: `from spyglass.common import Session` and `GLMSelection` (from
`GLM.glm_tables_dmr`). Re-add `GLMSelection` if you want `GLMBasisSelection` to also
key on it.
"""

import datajoint as dj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import nemos as nmo
from scipy.stats import zscore
from pynwb.core import ScratchData
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from GLM.glm_tables_dmr import GLMStorage

schema = dj.schema("denissemorales_glmbasis")


# =============================================================================
# Basis-construction functions (formerly build_bases_new_schema.py)
# =============================================================================
#
# Adapted from stseng_glm.py's `build_bases_combined_epochs` / `process_design_matrix`,
# for a per-timebin dataframe with columns:
#
#     time, unit_0 ... unit_N,                       (neural responses -> Y)
#     position_x, position_y, speed, orientation,
#     linear_position, track_segment_id,
#     upcoming_turn, turn, previous_turn,
#     path_type_handle_to_left, path_type_handle_to_right,
#     path_type_left_to_handle, path_type_none, path_type_right_to_handle,
#     previous_arm, home_arm,
#     previous_reward, current_reward,
#     nwb_file_name, epoch
#
# Design:
#   1. Build continuous-variable basis expansions with `nemos.basis`:
#        - linear_position  -> RaisedCosineLinearEval   (like the original "ppt")
#        - position_x/y     -> RaisedCosineLinearEval x RaisedCosineLinearEval (2D place-field basis)
#        - speed            -> BSplineEval
#        - orientation      -> CyclicBSplineEval (circular variable)
#   2. Build categorical interactions by zeroing out the linear_position (and, where
#      relevant, the 2D position) basis outside each category's rows -- the same
#      "mask -> concatenate" pattern used for ppt_reward / ppt_prev_turn / ppt_path /
#      pos_2d_maze in the original code, factored into one reusable helper
#      (`_split_basis_by_category`).
#   3. Assemble everything into a design matrix X with group indices for group-lasso
#      regularization (`process_design_matrix`).
#
# NOTE on things to double check / adapt for your data:
#   - `orientation` is assumed to be in radians on [0, 2*pi); adjust `_wrap_to_2pi`
#     if it's in degrees or signed [-pi, pi].
#   - `path_type_*` is already one-hot (5 mutually exclusive dummy columns), handled
#     directly as boolean masks.
#   - `previous_arm` / `home_arm` / `track_segment_id` / `turn` / `previous_turn` /
#     `upcoming_turn` are treated as arbitrary categorical labels; NaN rows contribute
#     all-zero columns for that interaction.
#   - `nwb_file_name` / `epoch` are NOT part of the design matrix -- they're
#     identifiers, used only for per-epoch nuisance regressors below and (outside
#     this file) for group-aware CV splitting.

UNIT_PREFIX = "unit_"

PATH_TYPE_COLUMNS = [
    "path_type_handle_to_left",
    "path_type_handle_to_right",
    "path_type_left_to_handle",
    "path_type_none",
    "path_type_right_to_handle",
]


def get_unit_columns(df):
    """Return the neuron response columns (Y), in column order."""
    return [c for c in df.columns if c.startswith(UNIT_PREFIX)]


def _wrap_to_2pi(x):
    return np.mod(x, 2 * np.pi)


def _nan_to_zero(feat):
    """nemos basis evaluations return NaN outside `bounds`; original code zeroed these."""
    arr = np.asarray(feat)
    arr = np.nan_to_num(arr, nan=0.0)
    return jnp.array(arr)


def _split_basis_by_category(eval_basis, cat_values, n_basis, name_prefix):
    """
    Zero out `eval_basis` outside each category of `cat_values` and concatenate the
    pieces -- generalizes the ppt_prev_turn / ppt_path / pos_2d_maze pattern from the
    original `build_bases_combined_epochs` into one reusable helper.

    Rows where `cat_values` is NaN contribute all-zero columns for every category
    (i.e. that timepoint doesn't drive any of the interaction terms).

    Returns
    -------
    combined   : jnp.ndarray, shape (n_samples, n_categories * n_basis)
    cat_idx    : ndarray, shape (n_categories * n_basis,) -- which category each column belongs to
    names      : list[str], length n_categories * n_basis
    labels     : sorted list of category labels actually used
    """
    cat_values = np.asarray(cat_values)
    is_valid = ~pd.isna(cat_values)
    labels = sorted(pd.unique(cat_values[is_valid]).tolist())

    pieces, cat_idx, names = [], [], []
    for label in labels:
        mask = np.zeros(eval_basis.shape[0], dtype=bool)
        mask[cat_values == label] = True
        mask2d = np.broadcast_to(mask[:, None], eval_basis.shape)
        piece = eval_basis.at[~mask2d].set(0.0)
        pieces.append(piece)
        cat_idx += [label] * n_basis
        names.extend([f"{name_prefix}_{label}_bump_{i}" for i in range(n_basis)])

    combined = jnp.concatenate(pieces, axis=-1) if pieces else jnp.zeros((eval_basis.shape[0], 0))
    return combined, np.array(cat_idx), names, labels


def build_bases(
    df,
    linear_position_min=None,
    linear_position_max=None,
    n_basis_funcs_linear_position=20,
    n_basis_funcs_x=24,
    n_basis_funcs_y=36,
    n_basis_funcs_speed=5,
    n_basis_funcs_orientation=8,
    pos_x_range=None,
    pos_y_range=None,
    n_linpos_grid=100,
    n_pos_x_grid=60,
    n_pos_y_grid=80,
    n_speed_grid=100,
    n_orientation_grid=100,
):
    """
    Build basis-expanded behavioral features for the combined dataframe.

    Mirrors `build_bases_combined_epochs` in stseng_glm.py, but keyed to the new
    column schema. Returns three dicts, same shape/role as the original:

    feature_dict      : {var_name: jnp.ndarray of shape (n_samples, n_basis_for_var)}
    feature_name_dict  : {var_name: list[str] of length n_basis_for_var}
    kernel_dict        : basis functions evaluated on a fine grid (for tuning-curve plots),
                         plus the category index arrays for each interaction term
    """
    n_samples = df.shape[0]

    # ---------------------------------------------------------------
    # linear_position (analogous to "ppt" in the original -- normalized
    # progression through the track / trial)
    # ---------------------------------------------------------------
    linpos = df["linear_position"].values.astype(float)
    lp_min = linear_position_min if linear_position_min is not None else np.nanmin(linpos)
    lp_max = linear_position_max if linear_position_max is not None else np.nanmax(linpos)

    cos_bases_linpos = nmo.basis.RaisedCosineLinearEval(
        n_basis_funcs=n_basis_funcs_linear_position, width=2, bounds=[lp_min, lp_max]
    )
    eval_linpos = _nan_to_zero(cos_bases_linpos.compute_features(linpos))
    linpos_basis_names = [f"linear_position_bump_{i}" for i in range(n_basis_funcs_linear_position)]
    linpos_grid = np.linspace(lp_min, lp_max, n_linpos_grid)
    kernel_linpos = cos_bases_linpos.compute_features(linpos_grid)

    # interaction: linear_position x current/previous reward (+/-0.5 coded, as in original ppt_reward)
    current_reward = df["current_reward"].values.astype(float) - 0.5
    linpos_reward = eval_linpos * current_reward[:, None]
    linpos_reward_names = [f"linpos_reward_bump_{i}" for i in range(n_basis_funcs_linear_position)]

    previous_reward = df["previous_reward"].values.astype(float) - 0.5
    linpos_prev_reward = eval_linpos * previous_reward[:, None]
    linpos_prev_reward_names = [f"linpos_prev_reward_bump_{i}" for i in range(n_basis_funcs_linear_position)]

    # interaction: linear_position split by turn / previous_turn / upcoming_turn identity
    linpos_turn, linpos_turn_idx, linpos_turn_names, _ = _split_basis_by_category(
        eval_linpos, df["turn"].values, n_basis_funcs_linear_position, "linpos_turn"
    )
    linpos_prev_turn, linpos_prev_turn_idx, linpos_prev_turn_names, _ = _split_basis_by_category(
        eval_linpos, df["previous_turn"].values, n_basis_funcs_linear_position, "linpos_prev_turn"
    )
    linpos_upcoming_turn, linpos_upcoming_turn_idx, linpos_upcoming_turn_names, _ = _split_basis_by_category(
        eval_linpos, df["upcoming_turn"].values, n_basis_funcs_linear_position, "linpos_upcoming_turn"
    )

    # interaction: linear_position split by previous arm / home arm identity
    linpos_prev_arm, linpos_prev_arm_idx, linpos_prev_arm_names, _ = _split_basis_by_category(
        eval_linpos, df["previous_arm"].values, n_basis_funcs_linear_position, "linpos_prev_arm"
    )
    linpos_home_arm, linpos_home_arm_idx, linpos_home_arm_names, _ = _split_basis_by_category(
        eval_linpos, df["home_arm"].values, n_basis_funcs_linear_position, "linpos_home_arm"
    )

    # interaction: linear_position split by track segment (analogous to "ppt_maze" in original)
    linpos_segment, linpos_segment_idx, linpos_segment_names, _ = _split_basis_by_category(
        eval_linpos, df["track_segment_id"].values, n_basis_funcs_linear_position, "linpos_segment"
    )

    # interaction: linear_position split by path type -- path type is already one-hot in the
    # dataframe, so use the dummy columns directly as masks instead of `_split_basis_by_category`
    path_pieces, path_idx, path_names = [], [], []
    for i_path, col in enumerate(PATH_TYPE_COLUMNS):
        mask = df[col].values.astype(bool)
        mask2d = np.broadcast_to(mask[:, None], eval_linpos.shape)
        path_pieces.append(eval_linpos.at[~mask2d].set(0.0))
        path_idx += [i_path] * n_basis_funcs_linear_position
        short_name = col.replace("path_type_", "")
        path_names.extend([f"linpos_path_{short_name}_bump_{i}" for i in range(n_basis_funcs_linear_position)])
    linpos_path_type = jnp.concatenate(path_pieces, axis=-1)
    linpos_path_type_idx = np.array(path_idx)

    # ---------------------------------------------------------------
    # 2D position (position_x, position_y)
    # ---------------------------------------------------------------
    pos_x = df["position_x"].values.astype(float)
    pos_y = df["position_y"].values.astype(float)
    if pos_x_range is None:
        pos_x_range = (float(np.nanmin(pos_x)), float(np.nanmax(pos_x)))
    if pos_y_range is None:
        pos_y_range = (float(np.nanmin(pos_y)), float(np.nanmax(pos_y)))
    pos_x_c = np.clip(pos_x, *pos_x_range)
    pos_y_c = np.clip(pos_y, *pos_y_range)

    cos_bases_pos_x = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis_funcs_x, width=2, bounds=pos_x_range)
    cos_bases_pos_y = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis_funcs_y, width=2, bounds=pos_y_range)
    cos_bases_pos_2d = cos_bases_pos_x * cos_bases_pos_y  # nemos MultiplicativeBasis, like original
    eval_pos_2d = _nan_to_zero(cos_bases_pos_2d.compute_features(pos_x_c, pos_y_c))
    pos_2d_basis_names = [f"pos_2d_bump_{i}" for i in range(n_basis_funcs_x * n_basis_funcs_y)]

    # kernel for visualization, built unbounded (same workaround noted in the original code
    # for a bounds-related bug in evaluate_on_grid)
    cos_bases_pos_x_vis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis_funcs_x, width=2)
    cos_bases_pos_y_vis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis_funcs_y, width=2)
    _, _, kernel_pos = (cos_bases_pos_x_vis * cos_bases_pos_y_vis).evaluate_on_grid(n_pos_x_grid, n_pos_y_grid)

    # interaction: 2D position split by track segment (analogous to "pos_2d_maze" in original)
    pos_2d_segment, pos_2d_segment_idx, pos_2d_segment_names, _ = _split_basis_by_category(
        eval_pos_2d, df["track_segment_id"].values, n_basis_funcs_x * n_basis_funcs_y, "pos_2d_segment"
    )

    # ---------------------------------------------------------------
    # speed (BSpline, same as original)
    # ---------------------------------------------------------------
    speed = df["speed"].values.astype(float)
    speed_min = 0.0
    speed_max = float(np.nanpercentile(speed, 99.0))
    speed_c = np.clip(speed, speed_min, speed_max)
    bspline_basis_speed = nmo.basis.BSplineEval(n_basis_funcs_speed, order=4, bounds=[speed_min, speed_max])
    eval_speed = _nan_to_zero(bspline_basis_speed.compute_features(speed_c))
    speed_basis_names = [f"speed_bump_{i}" for i in range(n_basis_funcs_speed)]
    speed_grid = np.linspace(speed_min, speed_max, n_speed_grid)
    kernel_speed = bspline_basis_speed.compute_features(speed_grid)

    # ---------------------------------------------------------------
    # orientation -- circular variable, use nemos's cyclic basis
    # ---------------------------------------------------------------
    orientation = _wrap_to_2pi(df["orientation"].values.astype(float))
    cyclic_bases_orientation = nmo.basis.CyclicBSplineEval(n_basis_funcs_orientation, order=4, bounds=[0, 2 * np.pi])
    eval_orientation = _nan_to_zero(cyclic_bases_orientation.compute_features(orientation))
    orientation_basis_names = [f"orientation_bump_{i}" for i in range(n_basis_funcs_orientation)]
    orientation_grid = np.linspace(0, 2 * np.pi, n_orientation_grid, endpoint=False)
    kernel_orientation = cyclic_bases_orientation.compute_features(orientation_grid)

    # ---------------------------------------------------------------
    # per-epoch nuisance regressors (analogous to "time_in_epoch" / "epoch_offset")
    # ---------------------------------------------------------------
    epoch = df["epoch"].values
    unique_epoch = pd.unique(epoch)
    time_in_epoch_cols, epoch_offset_cols = [], []
    for this_epoch in unique_epoch:
        this_mask = epoch == this_epoch
        ramp = np.zeros(n_samples, dtype=float)
        ramp[this_mask] = np.linspace(0, 1, int(this_mask.sum()))
        time_in_epoch_cols.append(ramp)
        epoch_offset_cols.append(this_mask.astype(float))
    time_in_epoch = jnp.array(np.stack(time_in_epoch_cols, axis=1))
    epoch_offset = jnp.array(np.stack(epoch_offset_cols, axis=1))
    time_in_epoch_names = [f"time_in_epoch_bump_{i}" for i in range(len(unique_epoch))]
    epoch_offset_names = [f"epoch_offset_bump_{i}" for i in range(len(unique_epoch))]

    feature_dict = {
        "linear_position": eval_linpos,
        "linpos_reward": linpos_reward,
        "linpos_prev_reward": linpos_prev_reward,
        "linpos_turn": linpos_turn,
        "linpos_prev_turn": linpos_prev_turn,
        "linpos_upcoming_turn": linpos_upcoming_turn,
        "linpos_prev_arm": linpos_prev_arm,
        "linpos_home_arm": linpos_home_arm,
        "linpos_segment": linpos_segment,
        "linpos_path_type": linpos_path_type,
        "pos_2d": eval_pos_2d,
        "pos_2d_segment": pos_2d_segment,
        "speed": eval_speed,
        "orientation": eval_orientation,
        "time_in_epoch": time_in_epoch,
        "epoch_offset": epoch_offset,
    }

    feature_name_dict = {
        "linear_position": linpos_basis_names,
        "linpos_reward": linpos_reward_names,
        "linpos_prev_reward": linpos_prev_reward_names,
        "linpos_turn": linpos_turn_names,
        "linpos_prev_turn": linpos_prev_turn_names,
        "linpos_upcoming_turn": linpos_upcoming_turn_names,
        "linpos_prev_arm": linpos_prev_arm_names,
        "linpos_home_arm": linpos_home_arm_names,
        "linpos_segment": linpos_segment_names,
        "linpos_path_type": path_names,
        "pos_2d": pos_2d_basis_names,
        "pos_2d_segment": pos_2d_segment_names,
        "speed": speed_basis_names,
        "orientation": orientation_basis_names,
        "time_in_epoch": time_in_epoch_names,
        "epoch_offset": epoch_offset_names,
    }

    kernel_dict = {
        "kernel_linpos": kernel_linpos,
        "kernel_pos": kernel_pos,
        "kernel_speed": kernel_speed,
        "kernel_orientation": kernel_orientation,
        "linpos_turn_idx": linpos_turn_idx,
        "linpos_prev_turn_idx": linpos_prev_turn_idx,
        "linpos_upcoming_turn_idx": linpos_upcoming_turn_idx,
        "linpos_prev_arm_idx": linpos_prev_arm_idx,
        "linpos_home_arm_idx": linpos_home_arm_idx,
        "linpos_segment_idx": linpos_segment_idx,
        "linpos_path_type_idx": linpos_path_type_idx,
        "pos_2d_segment_idx": pos_2d_segment_idx,
    }

    return feature_dict, feature_name_dict, kernel_dict


# variables that get a single occupancy threshold applied to the whole basis block
_OCCUPANCY_THRESHOLD_VARS = {
    "linear_position", "linpos_reward", "linpos_prev_reward",
    "linpos_turn", "linpos_prev_turn", "linpos_upcoming_turn",
    "linpos_prev_arm", "linpos_home_arm", "linpos_segment", "linpos_path_type",
    "pos_2d", "pos_2d_segment", "orientation",
}
# variables that get z-scored instead (unbounded / already-continuous nuisance regressors)
_ZSCORE_VARS = {"speed", "time_in_epoch"}
# variables kept as-is (binary indicator columns)
_PASSTHROUGH_VARS = {"epoch_offset"}


def process_design_matrix(
    df,
    var_names,
    occupancy_sd_thresh=1e-2,
    **basis_kwargs,
):
    """
    Build the final design matrix X (and grouping info for group-lasso) from `var_names`,
    mirroring `process_design_matrix` in stseng_glm.py.

    Returns
    -------
    X                  : ndarray, shape (n_samples, n_features)
    all_feature_names  : ndarray of str, length n_features
    group_ind          : ndarray of int, length n_features -- which group each column belongs to
    group_name         : list[str], one name per group (for group_lasso / model breakdown)
    feature_group_mask : ndarray of shape (n_groups, n_features), 1/0 group membership (for
                          `PopulationGLM_CV(regularization="group_lasso")`)
    selected_bases_idx : dict[var_name -> bool array] of which basis columns survived
                          occupancy thresholding, for reproducing the same selection at
                          prediction/ablation time
    kernel_dict        : passed through from build_bases, for tuning-curve plotting
    """
    feature_dict, feature_name_dict, kernel_dict = build_bases(df, **basis_kwargs)

    var_val, all_feature_names = [], []
    selected_bases_idx = {}
    group_size, group_name, group_ind = [], [], []
    i_group = 0

    for var in var_names:
        if var not in feature_dict:
            raise ValueError(
                f"Unknown variable '{var}'. Available: {sorted(feature_dict.keys())}"
            )

        if var in _OCCUPANCY_THRESHOLD_VARS:
            selected = np.array(feature_dict[var].std(axis=0)) > occupancy_sd_thresh
            var_val.append(np.array(feature_dict[var])[:, selected])
            selected_bases_idx[var] = selected
            all_feature_names.extend(list(np.array(feature_name_dict[var])[selected]))
            n_sel = int(np.sum(selected))

        elif var in _ZSCORE_VARS:
            vals = zscore(np.nan_to_num(np.array(feature_dict[var])), axis=0)
            var_val.append(vals)
            all_feature_names.extend(feature_name_dict[var])
            n_sel = vals.shape[-1]

        elif var in _PASSTHROUGH_VARS:
            vals = np.array(feature_dict[var])
            var_val.append(vals)
            all_feature_names.extend(feature_name_dict[var])
            n_sel = vals.shape[-1]

        else:
            raise ValueError(f"No handling rule registered for variable '{var}'")

        group_size.append(n_sel)
        group_name.append(var)
        group_ind += [i_group] * n_sel
        i_group += 1

    X = np.concatenate(var_val, axis=-1)
    all_feature_names = np.array(all_feature_names)
    group_ind = np.array(group_ind)

    feature_group_mask = np.zeros((len(group_name), X.shape[-1]), dtype=int)
    start = 0
    for i, sz in enumerate(group_size):
        feature_group_mask[i, start : start + sz] = 1
        start += sz

    return X, all_feature_names, group_ind, group_name, feature_group_mask, selected_bases_idx, kernel_dict


# =============================================================================
# DataJoint tables
# =============================================================================


@schema
class GLMBasisParams(SpyglassMixin, dj.Lookup):
    """Table for defining GLM basis-construction parameters."""

    definition = """
    # GLM basis parameters
    basis_param_id: smallint unsigned  # unique identifier for this basis parameter set
    basis_param_name: varchar(64)      # name of the basis parameter set
    ---
    basis_param_description = NULL : varchar(512)  # description of the basis parameter set
    var_names: blob        # list of variable names (design-matrix groups) to include,
                            # must match keys returned by build_bases() above
    basis_params: blob     # kwargs forwarded to build_bases() (n_basis_funcs_*, ranges, grid sizes, ...)
    occupancy_sd_thresh = 1e-2 : float  # bases with std <= this (no occupancy) are dropped
    """

    def insert_default_params(self):
        """Insert a default basis parameter set covering all variables build_bases() knows about."""
        var_names = [
            "linear_position",
            "linpos_reward",
            "linpos_prev_reward",
            "linpos_turn",
            "linpos_prev_turn",
            "linpos_upcoming_turn",
            "linpos_prev_arm",
            "linpos_home_arm",
            "linpos_segment",
            "linpos_path_type",
            "pos_2d",
            "pos_2d_segment",
            "speed",
            "orientation",
            "time_in_epoch",
            "epoch_offset",
        ]

        basis_params = {
            "n_basis_funcs_linear_position": 20,
            "n_basis_funcs_x": 24,
            "n_basis_funcs_y": 36,
            "n_basis_funcs_speed": 5,
            "n_basis_funcs_orientation": 8,
        }

        new_key = {
            "basis_param_id": 1,
            "basis_param_name": "default_basis",
            "basis_param_description": "Default basis set: linear position, 2D position, "
            "speed, orientation, plus reward/turn/arm/path/segment interactions with linear position.",
            "var_names": var_names,
            "basis_params": basis_params,
            "occupancy_sd_thresh": 1e-2,
        }

        self.insert1(new_key, skip_duplicates=True)
        logger.info(f"Inserted default GLMBasisParams: {new_key}")


@schema
class GLMBasisSelection(SpyglassMixin, dj.Manual):
    """Table for selecting which GLMStorage entry + basis params to build a GLM basis set for."""

    definition = """
    # Selection of combined dataframe (from GLMStorage) + basis parameters for GLM basis construction
    -> GLMStorage
    -> GLMBasisParams
    ---
    """

    def auto_insert_missing_keys(self, restriction_dict=None):
        """Auto-insert missing keys with given restrictions."""
        query = (GLMStorage * GLMBasisParams) - self.proj()
        if restriction_dict is not None:
            query = query & restriction_dict
        keys = query.fetch("KEY")
        self.insert(keys, skip_duplicates=True)


@schema
class GLMBasis(SpyglassMixin, dj.Computed):
    """Table for building and storing GLM basis sets (design matrix + basis kernels)."""

    definition = """
    # GLM basis set (design matrix built from behavioral/task variables)
    -> GLMBasisSelection
    ---
    n_samples: int unsigned                    # number of timebins (rows of X)
    n_features: int unsigned                   # number of design-matrix columns
    n_groups: int unsigned                     # number of variable groups (for group_lasso)
    design_matrix_object_id: varchar(40)       # object id for design matrix X, shape (n_samples, n_features)
    feature_names_object_id: varchar(40)       # object id for per-column feature names
    group_ind_object_id: varchar(40)           # object id for per-column group index
    feature_group_mask_object_id: varchar(40)  # object id for (n_groups x n_features) group membership mask
    unit_names_object_id: varchar(40)          # object id for unit_* column names/order used for Y
    -> AnalysisNwbfile
    """

    class GroupInfo(SpyglassMixinPart, dj.Part):
        """Per-group bookkeeping (name + size), for group_lasso / model-breakdown downstream."""

        definition = """
        -> master
        group_name: varchar(64)
        ---
        group_size: int unsigned
        """

    class Kernel(SpyglassMixinPart, dj.Part):
        """Basis functions evaluated on a fine grid, for tuning-curve plotting."""

        definition = """
        -> master
        ---
        kernel_linpos_object_id: varchar(40)
        kernel_pos_object_id: varchar(40)
        kernel_speed_object_id: varchar(40)
        kernel_orientation_object_id: varchar(40)
        -> AnalysisNwbfile
        """

    # tri-part make: replaces a monolithic make() / `_use_transaction = False`.
    # make_fetch and make_compute run OUTSIDE the DB transaction (so the long-running
    # basis-building + NWB-file writing below doesn't hold a table lock); only
    # make_insert runs inside a transaction, and it does nothing but insert.
    _parallel_make = True

    def make_fetch(self, key):
        """Read inputs. Read-only, deterministic, no DB writes."""
        var_names, basis_params, occupancy_sd_thresh = (
            GLMBasisParams() & key
        ).fetch1("var_names", "basis_params", "occupancy_sd_thresh")

        combined_df = self._fetch_combined_dataframe(key)

        return [var_names, basis_params, occupancy_sd_thresh, combined_df]

    def make_compute(self, key, var_names, basis_params, occupancy_sd_thresh, combined_df):
        """Run the computation and write the (potentially large, slow-to-write)
        analysis NWB files. No DataJoint/DB access here -- `AnalysisNwbfile().create()`
        and `.add_nwb_object()` only write to the NWB file on disk; the DB write that
        registers those files (`AnalysisNwbfile().add(...)`) is deferred to make_insert.
        """
        logger.info(f"Building GLM basis set for {key}...")

        # ------------------------------------------------------------------
        # 1. build basis sets + assemble design matrix (pure computation)
        # ------------------------------------------------------------------
        (
            X,
            feature_names,
            group_ind,
            group_name,
            feature_group_mask,
            selected_bases_idx,
            kernel_dict,
        ) = process_design_matrix(
            combined_df,
            var_names,
            occupancy_sd_thresh=occupancy_sd_thresh,
            **basis_params,
        )
        unit_names = np.array(get_unit_columns(combined_df))

        logger.info(
            f"Design matrix built: {X.shape[0]} samples x {X.shape[1]} features, "
            f"{len(group_name)} groups."
        )

        # ------------------------------------------------------------------
        # 2. write design matrix + metadata into an AnalysisNwbfile (disk I/O only;
        #    the file isn't registered in the DB yet -- that happens in make_insert)
        # ------------------------------------------------------------------
        nwb_file_name = key["nwb_file_name"]
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)

        design_matrix_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="design_matrix",
                data=np.asarray(X),
                description="GLM design matrix X, shape (n_samples, n_features)",
            ),
        )
        feature_names_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="feature_names",
                data=np.asarray(feature_names, dtype="U"),
                description="Per-column feature (coefficient) names for the design matrix",
            ),
        )
        group_ind_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="group_ind",
                data=np.asarray(group_ind),
                description="Per-column group index (which variable group each design-matrix column belongs to)",
            ),
        )
        feature_group_mask_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="feature_group_mask",
                data=np.asarray(feature_group_mask),
                description="Group membership mask (n_groups x n_features), 1/0, for group_lasso regularization",
            ),
        )
        unit_names_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="unit_names",
                data=unit_names.astype("U"),
                description="unit_* column names/order used for the response matrix Y",
            ),
        )

        master_key = {
            **key,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_groups": len(group_name),
            "design_matrix_object_id": design_matrix_object_id,
            "feature_names_object_id": feature_names_object_id,
            "group_ind_object_id": group_ind_object_id,
            "feature_group_mask_object_id": feature_group_mask_object_id,
            "unit_names_object_id": unit_names_object_id,
            "analysis_file_name": analysis_file_name,
        }

        # ------------------------------------------------------------------
        # 3. GroupInfo rows (plain dicts -- not inserted yet)
        # ------------------------------------------------------------------
        group_sizes = np.asarray(feature_group_mask).sum(axis=1)
        group_rows = [
            {**key, "group_name": name, "group_size": int(size)}
            for name, size in zip(group_name, group_sizes)
        ]

        # ------------------------------------------------------------------
        # 4. Kernel part-table row: separate analysis file, for tuning-curve plotting
        #    (again: disk I/O only, not registered in the DB yet)
        # ------------------------------------------------------------------
        kernel_analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        kernel_key = {
            **key,
            "analysis_file_name": kernel_analysis_file_name,
            "kernel_linpos_object_id": AnalysisNwbfile().add_nwb_object(
                kernel_analysis_file_name,
                ScratchData(
                    name="kernel_linpos",
                    data=np.asarray(kernel_dict["kernel_linpos"]),
                    description="Linear-position raised-cosine basis evaluated on a fine grid, for tuning-curve plotting",
                ),
            ),
            "kernel_pos_object_id": AnalysisNwbfile().add_nwb_object(
                kernel_analysis_file_name,
                ScratchData(
                    name="kernel_pos",
                    data=np.asarray(kernel_dict["kernel_pos"]),
                    description="2D position raised-cosine basis evaluated on a fine grid, for tuning-curve plotting",
                ),
            ),
            "kernel_speed_object_id": AnalysisNwbfile().add_nwb_object(
                kernel_analysis_file_name,
                ScratchData(
                    name="kernel_speed",
                    data=np.asarray(kernel_dict["kernel_speed"]),
                    description="Speed B-spline basis evaluated on a fine grid, for tuning-curve plotting",
                ),
            ),
            "kernel_orientation_object_id": AnalysisNwbfile().add_nwb_object(
                kernel_analysis_file_name,
                ScratchData(
                    name="kernel_orientation",
                    data=np.asarray(kernel_dict["kernel_orientation"]),
                    description="Orientation cyclic B-spline basis evaluated on a fine grid, for tuning-curve plotting",
                ),
            ),
        }

        logger.info(f"Finished building GLM basis set for {key}; ready to insert.")

        return [master_key, group_rows, kernel_key]

    def make_insert(self, key, master_key, group_rows, kernel_key):
        """Write results. The only method allowed to touch the database; runs inside
        a transaction, so keep this fast -- just registering already-written NWB
        files and inserting rows.
        """
        AnalysisNwbfile().add(key["nwb_file_name"], master_key["analysis_file_name"])
        self.insert1(master_key)

        self.GroupInfo.insert(group_rows)

        AnalysisNwbfile().add(key["nwb_file_name"], kernel_key["analysis_file_name"])
        self.Kernel.insert1(kernel_key)

    def _fetch_combined_dataframe(self, key):
        """
        Fetch the combined per-timebin dataframe (time, unit_0..unit_N, position_x,
        position_y, speed, orientation, linear_position, track_segment_id,
        upcoming_turn/turn/previous_turn, path_type_* (one-hot), previous_arm,
        home_arm, previous_reward, current_reward, nwb_file_name, epoch) from
        GLMStorage, restricted to this key.
        """
        glm_storage_nwb = (GLMStorage() & key).fetch_nwb()[0]
        return glm_storage_nwb["trial"]

    def fetch_design_matrix(self, key=None):
        """Convenience getter: returns (X, feature_names, group_ind, group_name, feature_group_mask) for one entry."""
        key = self.fetch1("KEY") if key is None else key
        nwbf = (self & key).fetch_nwb()[0]

        X = np.asarray(nwbf["design_matrix"].data)
        feature_names = np.asarray(nwbf["feature_names"].data).astype(str)
        group_ind = np.asarray(nwbf["group_ind"].data)
        feature_group_mask = np.asarray(nwbf["feature_group_mask"].data)
        group_name = list((self.GroupInfo & key).fetch("group_name"))

        return X, feature_names, group_ind, group_name, feature_group_mask

    def fetch_kernels(self, key=None):
        """Convenience getter: returns the kernel_dict pieces needed for tuning-curve plotting."""
        key = self.fetch1("KEY") if key is None else key
        nwbf = (self.Kernel & key).fetch_nwb()[0]
        return {
            "kernel_linpos": np.asarray(nwbf["kernel_linpos"].data),
            "kernel_pos": np.asarray(nwbf["kernel_pos"].data),
            "kernel_speed": np.asarray(nwbf["kernel_speed"].data),
            "kernel_orientation": np.asarray(nwbf["kernel_orientation"].data),
        }