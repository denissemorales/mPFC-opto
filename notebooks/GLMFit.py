"""
Spyglass GLM Fit Pipeline

Fits PopulationGLM_CV models (glm_nemos.py) to the design matrices built by
GLMBasis.py

Author: DMR + Claude
Date: August 2026
"""

import datajoint as dj
import numpy as np
import pandas as pd
from pynwb.core import ScratchData
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

from GLMBasis import GLMBasis, get_unit_columns
from GLM_nemos import PopulationGLM_CV

schema = dj.schema("denissemorales_glmfit")


@schema
class GLMFitParams(SpyglassMixin, dj.Lookup):
    """Table for defining GLM fitting parameters (PopulationGLM_CV + model selection)."""

    definition = """
    # GLM fit parameters
    fit_param_id: smallint unsigned   # unique identifier for this fit parameter set
    fit_param_name: varchar(64)       # name of the fit parameter set
    ---
    fit_param_description = NULL : varchar(512)  # description of the fit parameter set
    fit_params: blob             # kwargs forwarded to PopulationGLM_CV(...): activation,
                                  # loss_type, regularization, lambda_series, n_folds,
                                  # auto_split, split_by_group, split_random_state, solver_kwargs
    model_selection_params: blob # kwargs forwarded to model.select_model(...): se_fraction,
                                  # min_lambda, fancy_select
    group_idx_col = NULL : varchar(64)  # optional column of the combined dataframe to group
                                         # CV folds by (used when fit_params["split_by_group"]
                                         # is True); leave NULL for plain shuffled KFold
    """

    def insert_default_params(self):
        """Insert a default ridge-regularized fit parameter set."""
        fit_params = {
            "activation": "exp",
            "loss_type": "poisson",
            "regularization": "ridge",
            "lambda_series": (10.0 ** np.linspace(-1, -8, 30)).tolist(),
            "n_folds": 5,
            "auto_split": True,
            "split_by_group": False,
            "solver_kwargs": {"tol": 1e-3, "maxiter": 1000},
        }
        model_selection_params = {
            "se_fraction": 1.0,
            "min_lambda": 0.0,
            "fancy_select": False,
        }

        new_key = {
            "fit_param_id": 1,
            "fit_param_name": "default_ridge",
            "fit_param_description": "Default ridge-regularized Poisson GLM fit, "
            "5-fold shuffled CV (no grouping), 1SE model selection.",
            "fit_params": fit_params,
            "model_selection_params": model_selection_params,
            "group_idx_col": None,
        }

        self.insert1(new_key, skip_duplicates=True)
        logger.info(f"Inserted default GLMFitParams: {new_key}")

    def insert_default_group_lasso_params(self):
        """Insert a default group-lasso fit parameter set (uses GLMBasis's feature_group_mask)."""
        fit_params = {
            "activation": "exp",
            "loss_type": "poisson",
            "regularization": "group_lasso",
            "lambda_series": (10.0 ** np.linspace(2, -5, 30)).tolist(),
            "n_folds": 5,
            "auto_split": True,
            "split_by_group": False,
            "solver_kwargs": {"tol": 1e-3, "maxiter": 1000},
        }
        model_selection_params = {
            "se_fraction": 1.0,
            "min_lambda": 0.0,
            "fancy_select": False,
        }

        new_key = {
            "fit_param_id": 2,
            "fit_param_name": "default_group_lasso",
            "fit_param_description": "Default group-lasso Poisson GLM fit using the "
            "variable groups from GLMBasis's feature_group_mask, 5-fold shuffled CV.",
            "fit_params": fit_params,
            "model_selection_params": model_selection_params,
            "group_idx_col": None,
        }

        self.insert1(new_key, skip_duplicates=True)
        logger.info(f"Inserted default group-lasso GLMFitParams: {new_key}")


@schema
class GLMFitSelection(SpyglassMixin, dj.Manual):
    """Table for selecting which GLMBasis design matrix + fit params to fit a GLM with."""

    definition = """
    # Selection of design matrix (from GLMBasis) + fit parameters for GLM fitting
    -> GLMBasis
    -> GLMFitParams
    ---
    """

    def auto_insert_missing_keys(self, restriction_dict=None):
        """Auto-insert missing keys with given restrictions."""
        query = (GLMBasis * GLMFitParams) - self.proj()
        if restriction_dict is not None:
            query = query & restriction_dict
        keys = query.fetch("KEY")
        self.insert(keys, skip_duplicates=True)


@schema
class GLMFit(SpyglassMixin, dj.Computed):
    """Table for fitting and storing PopulationGLM_CV results."""

    definition = """
    # GLM fit results (coefficients + CV fit quality) for one design matrix / fit-params combo
    -> GLMFitSelection
    ---
    n_units: int unsigned                    # number of neurons (Y columns) fit
    frac_dev_expl_cv_mean: float             # mean fraction deviance explained on CV held-out data, across units
    frac_dev_expl_cv_sd: float                # SD of fraction deviance explained on CV held-out data, across units
    frac_dev_expl_train_mean: float          # mean fraction deviance explained on training data, across units
    frac_dev_expl_train_sd: float            # SD of fraction deviance explained on training data, across units
    coef_object_id: varchar(40)              # object id for selected weights, shape (n_features, n_units)
    intercept_object_id: varchar(40)         # object id for selected intercepts, shape (n_units,)
    coef_cv_object_id: varchar(40)           # object id for per-fold weights, shape (n_folds, n_features, n_units)
    intercept_cv_object_id: varchar(40)      # object id for per-fold intercepts, shape (n_folds, n_units)
    fit_results_object_id: varchar(40)       # object id for per-unit results dataframe (unit_name,
                                              # frac_dev_expl_train, frac_dev_expl_cv, selected_lambda,
                                              # selected_lambda_ind)
    -> AnalysisNwbfile
    """

    # tri-part make: make_fetch/make_compute run outside the DB transaction (fitting
    # can take a while, especially over a long lambda_series x many folds), so the
    # long-running work doesn't hold a table lock; only make_insert (fast, just
    # registering the analysis file + inserting rows) runs inside a transaction.
    _parallel_make = True

    def make_fetch(self, key):
        """Read inputs. Read-only, deterministic, no DB writes."""
        fit_params, model_selection_params, group_idx_col = (
            GLMFitParams() & key
        ).fetch1("fit_params", "model_selection_params", "group_idx_col")

        X, feature_names, group_ind, group_name, feature_group_mask = (
            GLMBasis() & key
        ).fetch_design_matrix(key)

        # Y isn't stored by GLMBasis (only unit_names/order) -- refetch the combined
        # dataframe the same way GLMBasis did, and pull out the unit_* columns.
        combined_df = GLMBasis()._fetch_combined_dataframe(key)
        unit_names = np.array(get_unit_columns(combined_df))
        Y = combined_df[list(unit_names)].values.astype(float)

        group_idx = None
        if group_idx_col:
            group_idx = combined_df[group_idx_col].astype("category").cat.codes.values

        return [
            fit_params,
            model_selection_params,
            X,
            feature_names,
            group_ind,
            group_name,
            feature_group_mask,
            Y,
            unit_names,
            group_idx,
        ]

    def make_compute(
        self,
        key,
        fit_params,
        model_selection_params,
        X,
        feature_names,
        group_ind,
        group_name,
        feature_group_mask,
        Y,
        unit_names,
        group_idx,
    ):
        """Fit the model and write results into an AnalysisNwbfile (disk I/O only;
        no DataJoint/DB access here -- the file isn't registered in the DB yet, that
        happens in make_insert).
        """
        logger.info(f"Fitting GLM for {key} ({X.shape[0]} samples x {X.shape[1]} features, "
                    f"{Y.shape[1]} units)...")

        fit_params = dict(fit_params)
        model_selection_params = dict(model_selection_params or {})
        regularization = fit_params.get("regularization", "ridge")

        model = PopulationGLM_CV(**fit_params)
        model.fit(
            X,
            Y,
            group_idx=group_idx,
            feature_group_mask=feature_group_mask if regularization == "group_lasso" else None,
            verbose=False,
        )
        model.select_model(make_fig=False, **model_selection_params)

        # per-fold coefficients (for downstream ablation/stability analyses)
        w_cv, w0_cv = model.get_coefficients_cv()

        # training-data fit quality: evaluate the selected (all-data-refit) model on
        # the same X/Y it was fit on -- mirrors the original pipeline's frac_dev_expl_train
        frac_dev_expl_train, _, _, _ = model.evaluate(X, Y, make_fig=False)

        frac_dev_expl_cv_cap = np.clip(model.selected_frac_dev_expl_cv, 0, None)
        frac_dev_expl_train_cap = np.clip(frac_dev_expl_train, 0, None)

        fit_results_df = pd.DataFrame(
            {
                "unit_name": unit_names,
                "frac_dev_expl_train": frac_dev_expl_train,
                "frac_dev_expl_cv": model.selected_frac_dev_expl_cv,
                "selected_lambda": model.selected_lambda,
                "selected_lambda_ind": model.selected_lambda_ind,
            }
        )

        nwb_file_name = key["nwb_file_name"]
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)

        coef_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="coef",
                data=np.asarray(model.selected_w),
                description="Selected model weights, shape (n_features, n_units)",
            ),
        )
        intercept_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="intercept",
                data=np.asarray(model.selected_w0),
                description="Selected model intercepts, shape (n_units,)",
            ),
        )
        coef_cv_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="coef_cv",
                data=np.asarray(w_cv),
                description="Per-fold weights at each unit's selected lambda, shape (n_folds, n_features, n_units)",
            ),
        )
        intercept_cv_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ScratchData(
                name="intercept_cv",
                data=np.asarray(w0_cv),
                description="Per-fold intercepts at each unit's selected lambda, shape (n_folds, n_units)",
            ),
        )
        fit_results_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, fit_results_df, "fit_results"
        )

        master_key = {
            **key,
            "n_units": int(Y.shape[1]),
            "frac_dev_expl_cv_mean": float(np.nanmean(frac_dev_expl_cv_cap)),
            "frac_dev_expl_cv_sd": float(np.nanstd(frac_dev_expl_cv_cap)),
            "frac_dev_expl_train_mean": float(np.nanmean(frac_dev_expl_train_cap)),
            "frac_dev_expl_train_sd": float(np.nanstd(frac_dev_expl_train_cap)),
            "coef_object_id": coef_object_id,
            "intercept_object_id": intercept_object_id,
            "coef_cv_object_id": coef_cv_object_id,
            "intercept_cv_object_id": intercept_cv_object_id,
            "fit_results_object_id": fit_results_object_id,
            "analysis_file_name": analysis_file_name,
        }

        logger.info(
            f"Finished fitting GLM for {key}: mean frac_dev_expl_cv = "
            f"{master_key['frac_dev_expl_cv_mean']:.4f}"
        )

        return [master_key]

    def make_insert(self, key, master_key):
        """Write results. The only method allowed to touch the database; runs inside
        a transaction, so keep this fast.
        """
        AnalysisNwbfile().add(key["nwb_file_name"], master_key["analysis_file_name"])
        self.insert1(master_key)

    def fetch_coefficients(self, key=None):
        """Convenience getter: returns (w, w0, feature_names) for the selected model."""
        key = self.fetch1("KEY") if key is None else key
        nwbf = (self & key).fetch_nwb()[0]
        w = np.asarray(nwbf["coef"].data)
        w0 = np.asarray(nwbf["intercept"].data)
        _, feature_names, _, _, _ = (GLMBasis() & key).fetch_design_matrix(key)
        return w, w0, feature_names

    def fetch_cv_coefficients(self, key=None):
        """Convenience getter: returns (w_cv, w0_cv) -- per-fold weights/intercepts."""
        key = self.fetch1("KEY") if key is None else key
        nwbf = (self & key).fetch_nwb()[0]
        w_cv = np.asarray(nwbf["coef_cv"].data)
        w0_cv = np.asarray(nwbf["intercept_cv"].data)
        return w_cv, w0_cv

    def fetch_fit_results(self, key=None):
        """Convenience getter: returns the per-unit fit-results dataframe."""
        key = self.fetch1("KEY") if key is None else key
        nwbf = (self & key).fetch_nwb()[0]
        return nwbf["fit_results"]