"""
Spyglass GLM Basis Pipeline

Uses Nemos, based on Shih-Yi code.

Author: DMR
Date: August 2026

The basis-construction functions this calls (`get_unit_columns`, `build_bases`,
`process_design_matrix`, and their helpers) live in `basis_utils`, which has no
DataJoint dependency.

Tables (Params -> Selection -> Computed "tripartite" pattern):
  GLMBasisParams     (dj.Lookup)   -- basis-construction hyperparameters
  GLMBasisSelection  (dj.Manual)   -- GLMStorage entry x GLMBasisParams
  GLMBasis           (dj.Computed) -- builds + stores the design matrix via the
                                       tri-part make pattern (make_fetch /
                                       make_compute / make_insert)

`GLMBasisSelection` keys on `GLMStorage` and `GLMBasisParams` only. Add
`GLMSelection` to its definition if the basis should also key on the covariate
selection.
"""

import datajoint as dj
import numpy as np
from pynwb.core import ScratchData
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from mpfc_opto.glm.glm_tables import GLMStorage
from mpfc_opto.glm.basis_utils import (
    get_unit_columns,
    process_design_matrix,
)

schema = dj.schema("denissemorales_glmbasis")


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

    # Tri-part make: make_fetch and make_compute run outside the transaction,
    # so basis building and the NWB write do not hold a table lock. Only
    # make_insert runs inside one, and it does nothing but insert.
    #
    # No `_parallel_make`: in spyglass that flag means the make function itself
    # spawns a process pool, which this one does not. Setting it only routes
    # `populate(processes=N)` through NonDaemonPool for no reason.

    def make_fetch(self, key):
        """Read inputs. Read-only, deterministic, no DB writes."""
        var_names, basis_params, occupancy_sd_thresh = (
            GLMBasisParams() & key
        ).fetch1("var_names", "basis_params", "occupancy_sd_thresh")

        combined_df = self._fetch_combined_dataframe(key)

        return [var_names, basis_params, occupancy_sd_thresh, combined_df]

    def make_compute(
        self, key, var_names, basis_params, occupancy_sd_thresh, combined_df
    ):
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

        logger.info(
            f"Finished building GLM basis set for {key}; ready to insert."
        )

        return [master_key, group_rows, kernel_key]

    def make_insert(self, key, master_key, group_rows, kernel_key):
        """Write results. The only method allowed to touch the database; runs inside
        a transaction, so keep this fast -- just registering already-written NWB
        files and inserting rows.
        """
        AnalysisNwbfile().add(
            key["nwb_file_name"], master_key["analysis_file_name"]
        )
        self.insert1(master_key)

        self.GroupInfo.insert(group_rows)

        AnalysisNwbfile().add(
            key["nwb_file_name"], kernel_key["analysis_file_name"]
        )
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
