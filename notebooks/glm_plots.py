"""
glm_diagnostic_plots.py

Diagnostic plots for the GLM basis + fit pipeline (GLMBasis.py / GLMFit.py / GLM_nemos.py).

Two families of plots here:

1. Plots built purely from `GLMFit().fetch_fit_results()` -- the per-unit table
   (unit_name, frac_dev_expl_train, frac_dev_expl_cv, selected_lambda,
   selected_lambda_ind). These work right now, with no pipeline changes, since
   that table is already stored for every populated GLMFit entry.

2. The classic "regularization path" plot: fraction deviance explained vs. EVERY
   lambda in lambda_series (not just the one selected), one curve per unit. This
   is what `PopulationGLM_CV._model_selection_plot` draws internally when you call
   `select_model(make_fig=True)` -- but GLMFit currently only stores the ONE
   selected point per unit, not the full per-lambda curve, so `plot_regularization_path`
   below needs a live, already-`.fit()` `PopulationGLM_CV` model object (e.g. fit
   manually in a notebook cell -- see its docstring for a ready-to-paste example).
   If you want the full path plotted directly from stored GLMFit results with no
   refitting needed, GLMFit's make_compute would need to additionally persist
   `all_frac_dev_expl_cv` (shape n_lambdas x n_units) and `lambda_series` -- happy
   to add that if you want it.

3. Tuning-curve reconstruction, combining `GLMBasis().fetch_kernels()` (basis
   functions evaluated on a fine grid) with `GLMFit().fetch_coefficients()`
   (each unit's fitted weights, in the same feature order as the design matrix).
   Multiplying a unit's weights for one variable's basis columns by that
   variable's kernel reconstructs the shape of the tuning function the model
   learned for that variable, without needing to touch the design matrix or
   raw data at all.

   Only base (non-interaction) variables have a stored kernel: linear_position,
   pos_2d, speed, orientation. Interaction terms (linpos_turn, linpos_reward,
   pos_2d_segment, etc.) reuse the SAME kernel as their base variable (e.g.
   linpos_turn's bumps live on the same linear-position grid as linear_position
   itself) -- pass the interaction's own feature-name prefix via `var_name` and
   the base variable's kernel via `kernel`/`kernel_key` to reconstruct those.

   Caveat: linear_position, pos_2d, and orientation are "occupancy thresholded"
   in `process_design_matrix` (bumps with near-zero occupancy get dropped from
   the design matrix), so the number of surviving feature columns for one of
   these variables can be SMALLER than the kernel's basis dimension. When that
   happens there's no way to know which specific bumps survived (that mapping
   isn't currently persisted anywhere), so the reconstruction functions below
   raise a clear error rather than silently mis-aligning weights to bumps. If
   you hit that error and want it to just work, say the word and I'll add
   `selected_bases_idx` to what GLMBasis persists.
"""

import re

import numpy as np
import matplotlib.pyplot as plt


# Which stored kernel a given (base) variable's basis columns were evaluated
# on. Interaction terms built on top of a base variable (e.g. "linpos_turn" on
# top of "linear_position") share that base variable's kernel/grid.
_BASE_KERNEL_KEY = {
    "linear_position": "kernel_linpos",
    "trajectory_progress": "kernel_trajectory_progress",
    "pos_2d": "kernel_pos",
    "speed": "kernel_speed",
    "orientation": "kernel_orientation",
}


def plot_regularization_scatter(fit_results_df, ax=None, log_x=True):
    """
    Scatter of each unit's SELECTED regularization strength (x) vs. its CV
    fraction deviance explained (y). One point per unit -- this is the
    "regularization on x, deviance explained on y" view of what's already stored
    in GLMFit right now.

    NOTE: this is one point per unit (wherever model selection landed), not the
    full regularization path -- see plot_regularization_path for that.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(
        fit_results_df["selected_lambda"],
        fit_results_df["frac_dev_expl_cv"],
        s=25,
        alpha=0.7,
        edgecolor="none",
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Selected regularization strength (lambda)")
    ax.set_ylabel("Fraction deviance explained (CV)")
    ax.set_title("Selected lambda vs. CV fit quality, per unit")
    return ax


def plot_train_vs_cv(fit_results_df, ax=None):
    """
    Scatter of frac_dev_expl_train (x) vs. frac_dev_expl_cv (y) per unit, with a
    y=x reference line. Points noticeably below the line are units whose fit is
    more overfit (train performance not backed up by held-out performance) than
    others.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    train = fit_results_df["frac_dev_expl_train"]
    cv = fit_results_df["frac_dev_expl_cv"]
    lo = min(0, float(train.min()), float(cv.min()))
    hi = max(float(train.max()), float(cv.max())) * 1.05

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=(0.6, 0.6, 0.6), linewidth=1, zorder=0)
    ax.scatter(train, cv, s=25, alpha=0.7, edgecolor="none")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Fraction deviance explained (train)")
    ax.set_ylabel("Fraction deviance explained (CV)")
    ax.set_title("Train vs. CV fit quality, per unit")
    return ax


def plot_frac_dev_expl_hist(fit_results_df, ax=None, bins=None):
    """Histogram of CV fraction deviance explained across units."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if bins is None:
        bins = np.linspace(-0.1, 1.0, 30)

    vals = fit_results_df["frac_dev_expl_cv"]
    ax.hist(vals, bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(
        vals.mean(), color="k", linestyle="--", linewidth=1,
        label=f"mean = {vals.mean():.3f}",
    )
    ax.set_xlabel("Fraction deviance explained (CV)")
    ax.set_ylabel("Number of units")
    ax.set_title("Distribution of CV fit quality across units")
    ax.legend()
    return ax


def summary_panel(fit_results_df, figsize=(13, 4)):
    """Convenience: all three fit_results_df-based plots side by side.

    Usage:
        fit_results_df = (GLMFit() & key).fetch_fit_results(key)
        summary_panel(fit_results_df)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_regularization_scatter(fit_results_df, ax=axes[0])
    plot_train_vs_cv(fit_results_df, ax=axes[1])
    plot_frac_dev_expl_hist(fit_results_df, ax=axes[2])
    plt.tight_layout()
    return fig, axes


def plot_regularization_path(model, unit_idx=None, ax=None, highlight_selected=True):
    """
    The full regularization PATH: fraction deviance explained (CV) vs. every
    lambda in model.lambda_series -- either all units overlaid (thin grey lines,
    with each unit's selected point marked in red) or a single highlighted unit.

    Must be called on a `GLM_nemos.PopulationGLM_CV` model object that has
    already had `.fit()` called (select_model() optional, only needed to mark
    the selected point) -- e.g. fit manually in a notebook cell:

        from mpfc_opto.glm.basis import GLMBasis
        from mpfc_opto.glm.basis_utils import get_unit_columns
        from mpfc_opto.glm.glm_nemos import PopulationGLM_CV
        from mpfc_opto.glm.glm_fit_tables import GLMFitParams

        key = {...}  # a GLMFitSelection key
        fit_params, _, _ = (GLMFitParams() & key).fetch1(
            "fit_params", "model_selection_params", "group_idx_col"
        )
        X, feature_names, group_ind, group_name, feature_group_mask = (
            GLMBasis() & key
        ).fetch_design_matrix(key)
        combined_df = GLMBasis()._fetch_combined_dataframe(key)
        Y = combined_df[get_unit_columns(combined_df)].values.astype(float)

        model = PopulationGLM_CV(**fit_params)
        model.fit(X, Y, verbose=False)
        model.select_model(make_fig=False)

        plot_regularization_path(model)             # all units overlaid
        plot_regularization_path(model, unit_idx=3)  # just unit index 3
    """
    assert model.fitted, "model.fit() must be called before plotting the regularization path"

    all_frac_dev_expl_cv, _, _ = model._calculate_fit_quality_cv()  # shape (n_lambdas, n_units)
    log_lambda = np.log10(model.lambda_series)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if unit_idx is None:
        ax.plot(log_lambda, all_frac_dev_expl_cv, color="k", linewidth=0.5, alpha=0.4)
        if highlight_selected and getattr(model, "selected", False):
            ax.plot(
                np.log10(model.selected_lambda),
                model.selected_frac_dev_expl_cv,
                "o", color="crimson", markersize=3, linestyle="none",
            )
        title = "Regularization path (all units, red = each unit's selected lambda)"
    else:
        ax.plot(log_lambda, all_frac_dev_expl_cv[:, unit_idx], color="steelblue", linewidth=1.5)
        if highlight_selected and getattr(model, "selected", False):
            ax.plot(
                np.log10(model.selected_lambda[unit_idx]),
                model.selected_frac_dev_expl_cv[unit_idx],
                "o", color="crimson", markersize=8, label="selected",
            )
            ax.legend()
        title = f"Regularization path (unit index {unit_idx})"

    ax.set_xlabel("log10(lambda)  [weaker regularization <- -> stronger]")
    ax.set_ylabel("Fraction deviance explained (CV)")
    ax.set_title(title)
    return ax


# =============================================================================
# Tuning-curve reconstruction (GLMBasis.fetch_kernels() + GLMFit.fetch_coefficients())
# =============================================================================

def _feature_indices(feature_names, var_name):
    """Indices of feature_names belonging to `var_name`'s basis (its columns are
    always named f"{var_name}_bump_{i}" by build_bases/process_design_matrix)."""
    prefix = f"{var_name}_bump_"
    return [i for i, name in enumerate(np.asarray(feature_names)) if str(name).startswith(prefix)]


def _check_kernel_matches(kernel, idx, var_name, n_kernel_basis):
    if len(idx) == 0:
        raise ValueError(
            f"No design-matrix columns named '{var_name}_bump_*' were found in "
            f"feature_names -- '{var_name}' wasn't included in this GLMBasis entry's "
            f"var_names, or the name doesn't match what build_bases produces."
        )
    if len(idx) != n_kernel_basis:
        raise ValueError(
            f"'{var_name}' has {len(idx)} surviving design-matrix columns but its "
            f"kernel has {n_kernel_basis} basis functions. This happens when "
            f"occupancy thresholding (see process_design_matrix's "
            f"_OCCUPANCY_THRESHOLD_VARS) dropped some low-occupancy bumps for "
            f"'{var_name}' -- which specific bumps survived isn't currently "
            f"persisted, so weights can't be reliably matched back to grid "
            f"positions. Ask to have `selected_bases_idx` persisted in GLMBasis "
            f"if you need this to work for a thresholded variable."
        )


def reconstruct_tuning_curve(kernel, coef, feature_names, var_name, unit_idx, intercept=None, link="exp"):
    """
    Reconstruct one unit's tuning function for one basis variable by projecting
    its fitted weights back onto the basis evaluated on a fine grid.

    Parameters
    ----------
    kernel : array
        From `GLMBasis().fetch_kernels()[_BASE_KERNEL_KEY[var_name]]` (or the
        appropriate stored kernel for an interaction term's base variable).
        Shape (n_grid, n_basis) for 1D variables (linear_position, speed,
        orientation), or (n_x_grid, n_y_grid, n_basis) for pos_2d.
    coef : array, shape (n_features, n_units)
        From `GLMFit().fetch_coefficients()` (the `w` return value).
    feature_names : array of str
        From `GLMFit().fetch_coefficients()` (or `GLMBasis().fetch_design_matrix()`)
        -- must be in the same order as `coef`'s rows.
    var_name : str
        Which basis this is, e.g. "linear_position", "speed", "orientation",
        "pos_2d", or an interaction term like "linpos_turn_male" whose columns
        follow the same f"{var_name}_bump_{i}" naming.
    unit_idx : int
        Column of `coef` (and `intercept`) to reconstruct.
    intercept : array, shape (n_units,), optional
        From `GLMFit().fetch_coefficients()` (the `w0` return value). If given,
        it's added to the linear predictor before applying `link`.
    link : {"exp", "identity", None}
        "exp" (default) returns the reconstructed firing rate (matches the
        default Poisson GLM activation function); "identity"/None returns the
        raw linear predictor (log firing rate, for Poisson).

    Returns
    -------
    ndarray, shape (n_grid,) for 1D variables or (n_x_grid, n_y_grid) for pos_2d.
    """
    kernel = np.asarray(kernel)
    coef = np.asarray(coef)
    n_kernel_basis = kernel.shape[-1]
    idx = _feature_indices(feature_names, var_name)
    _check_kernel_matches(kernel, idx, var_name, n_kernel_basis)

    w = coef[idx, unit_idx]
    linear_pred = kernel @ w
    if intercept is not None:
        linear_pred = linear_pred + np.asarray(intercept)[unit_idx]

    if link == "exp":
        return np.exp(linear_pred)
    elif link in (None, "identity"):
        return linear_pred
    else:
        raise ValueError(f"Unknown link '{link}'; use 'exp', 'identity', or None.")


def plot_tuning_curve_1d(
    kernels, coef, feature_names, var_name, unit_idx, unit_name=None,
    intercept=None, link="exp", x_grid=None, ax=None,
):
    """
    Plot the reconstructed 1D tuning curve for one unit and one variable
    (linear_position, speed, orientation, or a 1D interaction term).

    Usage
    -----
        kernels = GLMBasis().fetch_kernels(key)
        w, w0, feature_names = GLMFit().fetch_coefficients(key)
        plot_tuning_curve_1d(kernels, w, feature_names, "linear_position",
                              unit_idx=3, unit_name="unit_3", intercept=w0)

    `x_grid`, if given, puts real units on the x-axis (e.g.
    `np.linspace(linpos_min, linpos_max, kernel.shape[0])` using the same
    bounds/grid size passed to `build_bases`/`GLMBasisParams` -- otherwise the
    x-axis is just the grid index).
    """
    kernel_key = _BASE_KERNEL_KEY.get(var_name)
    if kernel_key is None:
        raise ValueError(
            f"No known stored kernel for '{var_name}'. For an interaction term, "
            f"pass the BASE variable's kernel key explicitly by calling "
            f"reconstruct_tuning_curve(kernels['kernel_linpos'], ...) directly."
        )
    kernel = kernels[kernel_key]
    curve = reconstruct_tuning_curve(kernel, coef, feature_names, var_name, unit_idx, intercept, link)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    grid = x_grid if x_grid is not None else np.arange(len(curve))
    ax.plot(grid, curve, color="steelblue", linewidth=1.5)
    ax.set_xlabel(var_name if x_grid is not None else f"{var_name} (grid index)")
    ax.set_ylabel("Firing rate (a.u.)" if link == "exp" else "log firing rate (a.u.)")
    ax.set_title(f"{unit_name or f'unit index {unit_idx}'}: {var_name} tuning")
    return ax


def plot_tuning_curves_grid(
    kernels, coef, feature_names, var_name, unit_names=None, unit_idx=None,
    intercept=None, link="exp", x_grid=None, ncols=5, figsize=None,
):
    """
    Small multiples: reconstructed 1D tuning curve for `var_name`, one panel
    per unit, for a quick population-wide scan of tuning shape/strength.

    unit_idx : sequence of int, optional -- defaults to all units (coef.shape[1]).
    """
    coef = np.asarray(coef)
    n_units = coef.shape[1]
    unit_idx = range(n_units) if unit_idx is None else unit_idx
    n = len(unit_idx)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (2.6 * ncols, 2.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax_flat_idx, u in enumerate(unit_idx):
        ax = axes.flat[ax_flat_idx]
        name = unit_names[u] if unit_names is not None else f"unit {u}"
        plot_tuning_curve_1d(
            kernels, coef, feature_names, var_name, u,
            unit_name=name, intercept=intercept, link=link, x_grid=x_grid, ax=ax,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("")
    for ax_flat_idx in range(n, nrows * ncols):
        axes.flat[ax_flat_idx].axis("off")
    fig.suptitle(f"{var_name} tuning across units")
    plt.tight_layout()
    return fig, axes


def plot_place_field(
    kernels, coef, feature_names, unit_idx, unit_name=None,
    intercept=None, link="exp", extent=None, ax=None, cmap="viridis",
):
    """
    Plot the reconstructed 2D place field (pos_2d tuning) for one unit, as a
    heatmap over (position_x, position_y).

    `extent`, if given, is `[x_min, x_max, y_min, y_max]` in real position
    units (the pos_x_range/pos_y_range used to build the basis) -- otherwise
    axes are just grid index.
    """
    kernel = kernels["kernel_pos"]
    field = reconstruct_tuning_curve(kernel, coef, feature_names, "pos_2d", unit_idx, intercept, link)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(
        field.T, origin="lower", aspect="auto", cmap=cmap,
        extent=extent if extent is not None else None,
    )
    plt.colorbar(im, ax=ax, label="Firing rate (a.u.)" if link == "exp" else "log firing rate (a.u.)")
    ax.set_xlabel("position_x" if extent is not None else "position_x (grid index)")
    ax.set_ylabel("position_y" if extent is not None else "position_y (grid index)")
    ax.set_title(f"{unit_name or f'unit index {unit_idx}'}: place field")
    return ax


def _usable_covariates(kernels, feature_names, vars_1d):
    """Which 1D variables (in vars_1d) and whether pos_2d have both a stored
    kernel AND a matching (non-occupancy-thresholded-away) column count in
    feature_names. Returns (usable_1d, has_place_field, skipped)."""
    usable_1d, skipped = [], []
    for var in vars_1d:
        kernel_key = _BASE_KERNEL_KEY.get(var)
        kernel = kernels.get(kernel_key) if kernel_key else None
        idx = _feature_indices(feature_names, var)
        if kernel is None or len(idx) == 0:
            continue  # this GLMBasis entry simply wasn't built with `var`
        if len(idx) != np.asarray(kernel).shape[-1]:
            skipped.append(var)
            continue
        usable_1d.append(var)

    has_place_field = False
    kernel_pos = kernels.get("kernel_pos")
    idx_pos = _feature_indices(feature_names, "pos_2d")
    if kernel_pos is not None and idx_pos:
        if len(idx_pos) == np.asarray(kernel_pos).shape[-1]:
            has_place_field = True
        else:
            skipped.append("pos_2d")

    return usable_1d, has_place_field, skipped


def plot_unit_tuning_summary(
    kernels, coef, feature_names, unit_idx, unit_name=None, intercept=None,
    link="exp", vars_1d=("linear_position", "trajectory_progress", "speed", "orientation"),
    x_grids=None, figsize=None, verbose=True,
    overlay_categories=(("trajectory_progress", "trajectory_progress_path_type"),),
):
    """
    For ONE neuron, reconstruct and plot its predicted tuning curve for every
    covariate that has a stored kernel, side by side in one figure -- the
    per-neuron complement to `plot_tuning_curves_grid` (which is per-variable,
    across neurons).

    Usage
    -----
        kernels = GLMBasis().fetch_kernels(key)
        w, w0, feature_names = GLMFit().fetch_coefficients(key)
        plot_unit_tuning_summary(kernels, w, feature_names, unit_idx=3,
                                  unit_name="unit_3", intercept=w0)

    Parameters
    ----------
    vars_1d : which 1D covariates to try to include (only ones this GLMBasis
        entry was actually built with, and whose columns weren't partially
        dropped by occupancy thresholding, are plotted -- see `verbose`).
    x_grids : optional dict {var_name: array} to put real units on each 1D
        panel's x-axis (otherwise grid index).
    verbose : if True, prints a one-line note when a covariate is skipped
        (not in this basis, or occupancy-thresholded so it can't be safely
        reconstructed -- see reconstruct_tuning_curve's docstring).
    overlay_categories : pairs (var_name, interaction_prefix). On that
        variable's panel, the per-category curves of the interaction (via
        `plot_categorical_interaction`) are drawn over the main curve --
        by default trajectory_progress split by path type. Pairs whose
        variable or interaction isn't in this fit are ignored; pass () to
        turn the overlay off.

    Returns
    -------
    (fig, axes) -- axes is a flat array in the order: vars_1d (in the order
    given, skipping unusable ones), then the pos_2d place field if available.
    """
    usable_1d, has_place_field, skipped = _usable_covariates(kernels, feature_names, vars_1d)
    if verbose and skipped:
        print(f"plot_unit_tuning_summary: skipping {skipped} (not in this basis, or "
              f"occupancy-thresholded -- see reconstruct_tuning_curve docstring)")

    n_panels = len(usable_1d) + (1 if has_place_field else 0)
    if n_panels == 0:
        raise ValueError(
            "No usable covariates found -- check that `kernels` came from "
            "GLMBasis().fetch_kernels() for the same entry `coef`/feature_names "
            "came from GLMFit().fetch_coefficients()."
        )

    name = unit_name or f"unit index {unit_idx}"
    fig, axes = plt.subplots(1, n_panels, figsize=figsize or (4.2 * n_panels, 3.6))
    axes = np.atleast_1d(axes)

    for ax, var in zip(axes, usable_1d):
        plot_tuning_curve_1d(
            kernels, coef, feature_names, var, unit_idx,
            unit_name=name, intercept=intercept, link=link,
            x_grid=(x_grids or {}).get(var), ax=ax,
        )
        ax.set_title(var)
        for base_var, cat_prefix in overlay_categories:
            if var != base_var or not _category_labels_for_interaction(feature_names, cat_prefix):
                continue
            ax.lines[-1].set_label(f"{var} (all)")
            ax.lines[-1].set_color("0.4")
            ax.lines[-1].set_linestyle("--")
            plot_categorical_interaction(
                kernels, coef, feature_names, var, cat_prefix, unit_idx,
                unit_name=name, intercept=intercept, link=link,
                x_grid=(x_grids or {}).get(var), verbose=False, ax=ax,
            )
            ax.set_title(f"{var} x {cat_prefix.replace(var + '_', '')}")

    if has_place_field:
        plot_place_field(
            kernels, coef, feature_names, unit_idx, unit_name=name,
            intercept=intercept, link=link, ax=axes[-1],
        )
        axes[-1].set_title("pos_2d (place field)")

    fig.suptitle(f"{name}: predicted tuning to each covariate")
    plt.tight_layout()
    return fig, axes


def save_tuning_summaries_pdf(
    kernels, coef, feature_names, unit_names=None, intercept=None, link="exp",
    vars_1d=("linear_position", "trajectory_progress", "speed", "orientation"), x_grids=None,
    path="tuning_summaries.pdf", unit_idx=None, verbose=True,
    overlay_categories=(("trajectory_progress", "trajectory_progress_path_type"),),
):
    """
    Same as `plot_unit_tuning_summary`, but loops over every unit (or a subset
    via `unit_idx`) and saves one page per neuron into a single multi-page PDF
    -- the practical way to page through predicted tuning for a whole
    population (e.g. all 63 units) instead of calling the plot function once
    per unit in a notebook.

    Returns the output `path`.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    coef = np.asarray(coef)
    n_units = coef.shape[1]
    unit_idx = range(n_units) if unit_idx is None else unit_idx

    with PdfPages(path) as pdf:
        for u in unit_idx:
            name = unit_names[u] if unit_names is not None else f"unit index {u}"
            fig, _ = plot_unit_tuning_summary(
                kernels, coef, feature_names, u, unit_name=name,
                intercept=intercept, link=link, vars_1d=vars_1d,
                x_grids=x_grids, verbose=verbose,
                overlay_categories=overlay_categories,
            )
            pdf.savefig(fig)
            plt.close(fig)

    if verbose:
        print(f"Saved tuning-curve summaries for {len(list(unit_idx))} units to {path}")
    return path


# =============================================================================
# Categorical interactions (e.g. trajectory_progress x path_type, linpos x turn)
# =============================================================================

def _category_labels_for_interaction(feature_names, cat_prefix):
    """
    Find every category label present for a categorical interaction, given its
    feature-name prefix (e.g. "trajectory_progress_path_type", "linpos_turn").
    build_bases names these columns f"{cat_prefix}_{label}_bump_{i}" (both the
    `_split_basis_by_category` interactions -- turn, prev_turn, upcoming_turn,
    prev_arm, home_arm, segment -- and the path_type interaction use this exact
    convention), so this works for any of them.

    Returns
    -------
    dict {label: [column indices in feature_names]}, in the order labels were
    first encountered.
    """
    prefix = f"{cat_prefix}_"
    pattern = re.compile(rf"^{re.escape(prefix)}(.+)_bump_\d+$")
    labels_to_idx = {}
    for i, name in enumerate(np.asarray(feature_names)):
        m = pattern.match(str(name))
        if m:
            labels_to_idx.setdefault(m.group(1), []).append(i)
    return labels_to_idx


def plot_categorical_interaction(
    kernels, coef, feature_names, base_var, cat_prefix, unit_idx, unit_name=None,
    intercept=None, link="exp", x_grid=None, labels=None, colors=None,
    verbose=True, ax=None, include_base=True,
):
    """
    Overlay one unit's reconstructed tuning curve for `base_var`, split out
    separately for each category of a categorical interaction built on top of
    it -- e.g. "does this unit's trajectory-progress tuning look different
    depending on which path type this trip is?"

    Usage
    -----
        kernels = GLMBasis().fetch_kernels(key)
        w, w0, feature_names = GLMFit().fetch_coefficients(key)
        plot_categorical_interaction(
            kernels, w, feature_names, "trajectory_progress",
            "trajectory_progress_path_type", unit_idx=3, unit_name="unit_3", intercept=w0,
        )

    This is generic, not path_type-specific -- it works for any categorical
    interaction build_bases produces, e.g.:
        plot_categorical_interaction(kernels, w, feature_names, "trajectory_progress",
                                      "trajectory_progress_turn", unit_idx=3, intercept=w0)
        plot_categorical_interaction(kernels, w, feature_names, "linear_position",
                                      "linpos_segment", unit_idx=3, intercept=w0)

    Parameters
    ----------
    base_var : which base variable's kernel/grid to reconstruct onto -- must be
        a key in `_BASE_KERNEL_KEY` (e.g. "trajectory_progress", "linear_position").
        Interactions live on the same grid as their base variable.
    cat_prefix : the interaction's feature-name prefix, e.g.
        "trajectory_progress_path_type", "trajectory_progress_turn", "linpos_segment".
    labels : optional explicit list/order of category labels to plot (default:
        every label found in feature_names for this interaction, sorted).
    colors : optional list (matched to `labels`'/sorted order) or dict {label: color}.
    verbose : if True, prints a one-line note for any category skipped because
        occupancy thresholding left it with fewer surviving columns than the
        kernel's basis dimension (same caveat as reconstruct_tuning_curve).
    include_base : if True (default), each category's curve is the base
        variable's own weights PLUS that category's interaction weights, i.e.
        the full tuning the model predicts for that category. If False, only
        the interaction weights are used (the category's deviation from the
        base curve, which is not itself a firing rate).

    Returns
    -------
    ax
    """
    kernel_key = _BASE_KERNEL_KEY.get(base_var)
    if kernel_key is None:
        raise ValueError(
            f"No known stored kernel for base_var='{base_var}'. Must be one of "
            f"{sorted(_BASE_KERNEL_KEY)}."
        )
    kernel = np.asarray(kernels[kernel_key])
    coef = np.asarray(coef)

    labels_to_idx = _category_labels_for_interaction(feature_names, cat_prefix)
    if not labels_to_idx:
        raise ValueError(
            f"No design-matrix columns matching '{cat_prefix}_<label>_bump_*' were "
            f"found in feature_names -- '{cat_prefix}' wasn't included in this "
            f"GLMBasis entry's var_names, or the prefix doesn't match what "
            f"build_bases produces."
        )

    label_order = list(labels) if labels is not None else sorted(labels_to_idx.keys())

    w_base = 0.0
    if include_base:
        base_idx = _feature_indices(feature_names, base_var)
        _check_kernel_matches(kernel, base_idx, base_var, kernel.shape[-1])
        w_base = coef[base_idx, unit_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    grid = x_grid if x_grid is not None else np.arange(kernel.shape[0])
    skipped = []
    for i_label, label in enumerate(label_order):
        idx = labels_to_idx.get(label)
        if idx is None:
            skipped.append(label)  # this label just doesn't exist in feature_names at all
            continue
        if len(idx) != kernel.shape[-1]:
            skipped.append(label)  # occupancy thresholding dropped some bumps for this category
            continue

        w = coef[idx, unit_idx] + w_base
        linear_pred = kernel @ w
        if intercept is not None:
            linear_pred = linear_pred + np.asarray(intercept)[unit_idx]
        curve = np.exp(linear_pred) if link == "exp" else linear_pred

        color = colors.get(label) if isinstance(colors, dict) else (
            colors[i_label] if colors is not None else None
        )
        ax.plot(grid, curve, label=str(label), color=color, linewidth=1.5)

    if verbose and skipped:
        print(f"plot_categorical_interaction: skipping categories {skipped} for "
              f"'{cat_prefix}' (missing from this basis, or occupancy-thresholded "
              f"-- see reconstruct_tuning_curve docstring)")

    ax.set_xlabel(base_var if x_grid is not None else f"{base_var} (grid index)")
    ax.set_ylabel("Firing rate (a.u.)" if link == "exp" else "log firing rate (a.u.)")
    ax.set_title(f"{unit_name or f'unit index {unit_idx}'}: {base_var} x {cat_prefix}")
    ax.legend(fontsize=8)
    return ax


# =============================================================================
# Raw exp(beta) coefficients ("incidence rate ratio"-style view)
# =============================================================================
#
# This Poisson GLM uses a log link (activation="exp" by default in
# PopulationGLM_CV), so every raw fitted weight `beta` behaves the way a "beta
# coefficient" does in any log-linear/Poisson model: exp(beta) is a multiplicative
# effect on firing rate -- exp(beta) > 1 means that basis function pushes the
# rate up, < 1 means it pushes the rate down, and exp(beta) == 1 (beta == 0)
# means no effect. This is sometimes called an "incidence rate ratio" (IRR) in
# count-regression terminology.
#
# The wrinkle: trajectory_progress isn't a single raw predictor, it's basis-
# expanded into several bumps, each with its own beta -- there's no single
# "the beta for trajectory_progress." The smooth tuning-curve plots above
# (plot_tuning_curve_1d, plot_categorical_interaction) already show the COMBINED
# effect of all those bumps together, evaluated on a fine grid (exp(kernel @ w)).
# The two functions below instead show each bump's own RAW exp(beta) directly,
# without going through the kernel/grid at all -- useful for asking "which
# specific basis function(s)/categories are actually driving the effect,"
# independent of occupancy thresholding (works even if the kernel and
# surviving-column counts don't match, unlike the reconstruction functions above).

def plot_exp_beta_bar(coef, feature_names, var_name, unit_idx, unit_name=None, ax=None):
    """
    Bar plot of exp(beta) for one unit's `var_name` coefficients (e.g.
    "trajectory_progress") -- one bar per basis function ("bump"), with a
    dashed reference line at exp(beta)=1 (no effect).

    Usage
    -----
        w, w0, feature_names = GLMFit().fetch_coefficients(key)
        plot_exp_beta_bar(w, feature_names, "trajectory_progress", unit_idx=3, unit_name="unit_3")
    """
    idx = _feature_indices(feature_names, var_name)
    if not idx:
        raise ValueError(
            f"No design-matrix columns named '{var_name}_bump_*' were found in "
            f"feature_names -- '{var_name}' wasn't included in this GLMFit entry."
        )
    coef = np.asarray(coef)
    exp_betas = np.exp(coef[idx, unit_idx])

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 0.45 * len(idx)), 4))

    bump_labels = [f"bump {i}" for i in range(len(idx))]
    colors = ["crimson" if b < 1 else "steelblue" for b in exp_betas]
    ax.bar(bump_labels, exp_betas, color=colors)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("exp(beta)  (multiplicative firing-rate gain)")
    ax.set_title(f"{unit_name or f'unit index {unit_idx}'}: {var_name} -- raw exp(beta) per bump")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_exp_beta_heatmap(
    coef, feature_names, cat_prefix, unit_idx, unit_name=None, labels=None,
    ax=None, cmap="RdBu_r", annotate=True,
):
    """
    Heatmap of exp(beta) across bump index (rows) x category (columns), for a
    categorical interaction like "trajectory_progress_path_type" -- the raw-
    coefficient complement to `plot_categorical_interaction` (which shows the
    smooth combined tuning curve per category instead).

    Color is centered on beta=0 (i.e. exp(beta)=1, white) with a diverging
    colormap, so increases (exp(beta)>1, blue by default) and decreases
    (exp(beta)<1, red by default) read symmetrically; each cell is also
    annotated with its exact exp(beta) value when `annotate=True`.

    Usage
    -----
        w, w0, feature_names = GLMFit().fetch_coefficients(key)
        plot_exp_beta_heatmap(w, feature_names, "trajectory_progress_path_type",
                               unit_idx=3, unit_name="unit_3")
    """
    labels_to_idx = _category_labels_for_interaction(feature_names, cat_prefix)
    if not labels_to_idx:
        raise ValueError(
            f"No design-matrix columns matching '{cat_prefix}_<label>_bump_*' were "
            f"found in feature_names -- '{cat_prefix}' wasn't included in this "
            f"GLMFit entry's design matrix, or the prefix doesn't match what "
            f"build_bases produces."
        )
    label_order = list(labels) if labels is not None else sorted(labels_to_idx.keys())
    coef = np.asarray(coef)

    n_bumps = max(len(labels_to_idx[label]) for label in label_order if label in labels_to_idx)
    beta_mat = np.full((n_bumps, len(label_order)), np.nan)
    for j, label in enumerate(label_order):
        idx = labels_to_idx.get(label)
        if idx is None:
            continue
        beta_mat[: len(idx), j] = coef[idx, unit_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * len(label_order) + 2, 0.4 * n_bumps + 2))

    vmax = np.nanmax(np.abs(beta_mat)) if np.isfinite(beta_mat).any() else 1.0
    im = ax.imshow(beta_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("beta (log scale) -- exp(beta)=1 at 0 (white)")

    if annotate:
        for i in range(n_bumps):
            for j in range(len(label_order)):
                if np.isfinite(beta_mat[i, j]):
                    ax.text(j, i, f"{np.exp(beta_mat[i, j]):.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_yticks(range(n_bumps))
    ax.set_yticklabels([f"bump {i}" for i in range(n_bumps)])
    ax.set_title(f"{unit_name or f'unit index {unit_idx}'}: {cat_prefix} -- exp(beta) per bump x category")
    return ax

# =============================================================================
# 2D position: occupancy and predicted firing-rate maps
# =============================================================================
#
# `plot_place_field` above only works when every pos_2d bump survived
# occupancy thresholding. The functions below don't need the kernel: they
# predict each time bin's firing rate from the full design matrix
# (exp(X @ w + w0), every covariate included) and average it over position, so
# they work for any fit. Rows of X, the dataframe and the spike columns are all
# the same 50 ms bins, in the same order.

def plot_position_2d(df, color_by="time", ax=None, s=2, cmap="viridis"):
    """Scatter of the animal's 2D position (position_x, position_y), one point
    per time bin, colored by `color_by` (a column of `df`, default "time")."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    c = df[color_by].values
    if color_by == "time":
        c = c - np.nanmin(c)  # seconds since the epoch started
    sc = ax.scatter(df["position_x"], df["position_y"], c=c, s=s, cmap=cmap)
    plt.colorbar(sc, ax=ax, label=f"{color_by} (s from start)" if color_by == "time" else color_by)
    ax.set_xlabel("position_x")
    ax.set_ylabel("position_y")
    ax.set_aspect("equal")
    ax.set_title("2D position")
    return ax


def predict_firing_rate(X, coef, intercept, bin_size=0.05, link="exp"):
    """Predicted firing rate in Hz, shape (n_samples, n_units), from the fitted
    weights: exp(X @ coef + intercept) / bin_size. `bin_size` is GLMStorage's
    BIN_SIZE (50 ms)."""
    linear_pred = np.asarray(X) @ np.asarray(coef) + np.asarray(intercept)[None, :]
    rate = np.exp(linear_pred) if link == "exp" else linear_pred
    return rate / bin_size


def _binned_mean_map(values, x, y, x_edges, y_edges, min_occupancy):
    """Mean of `values` per (x, y) bin; bins with fewer than `min_occupancy`
    samples are NaN. Returns (map, occupancy), both shaped (n_y, n_x)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    occ, _, _ = np.histogram2d(y[ok], x[ok], bins=[y_edges, x_edges])
    total, _, _ = np.histogram2d(y[ok], x[ok], bins=[y_edges, x_edges], weights=values[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_map = total / occ
    mean_map[occ < min_occupancy] = np.nan
    return mean_map, occ


def plot_predicted_rate_map(
    df, X, coef, intercept, unit_idx, unit_name=None, bin_size=0.05, n_bins=(40, 60),
    min_occupancy=10, cmap="magma", show_observed=True, axes=None,
):
    """
    Heatmap of one unit's PREDICTED firing rate over 2D position (mean of the
    model's predicted rate in each spatial bin), next to the OBSERVED rate
    (spikes / bin_size) in the same bins when `show_observed=True`, so the
    model can be checked against the data directly.

    df : the GLMStorage dataframe for the same epoch (position_x/y, unit_* columns);
        must have the same rows, in the same order, as `X`.
    X, coef, intercept : GLMBasis().fetch_design_matrix()[0] and
        GLMFit().fetch_coefficients() (w, w0).
    n_bins : (n_x, n_y) spatial bins. min_occupancy : bins with fewer samples
    than this are left blank.
    """
    unit_cols = [c for c in df.columns if c.startswith("unit_")]
    x = df["position_x"].values.astype(float)
    y = df["position_y"].values.astype(float)
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins[0] + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), n_bins[1] + 1)

    predicted = predict_firing_rate(X, coef[:, [unit_idx]], np.asarray(intercept)[[unit_idx]], bin_size)[:, 0]
    pred_map, _ = _binned_mean_map(predicted, x, y, x_edges, y_edges, min_occupancy)
    maps = [("predicted", pred_map)]
    if show_observed:
        observed = df[unit_cols[unit_idx]].values.astype(float) / bin_size
        obs_map, _ = _binned_mean_map(observed, x, y, x_edges, y_edges, min_occupancy)
        maps.insert(0, ("observed", obs_map))

    if axes is None:
        fig, axes = plt.subplots(1, len(maps), figsize=(4.8 * len(maps), 5), squeeze=False)
        axes = axes[0]
    vmax = np.nanmax(np.concatenate([m[~np.isnan(m)] for _, m in maps]))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    name = unit_name or unit_cols[unit_idx]
    for ax, (label, m) in zip(axes, maps):
        im = ax.imshow(m, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
        ax.set_xlabel("position_x")
        ax.set_ylabel("position_y")
        ax.set_title(f"{name}: {label} rate")
    axes[0].figure.tight_layout()
    return axes


def plot_predicted_rate_maps_grid(
    df, X, coef, intercept, unit_names=None, unit_idx=None, bin_size=0.05,
    n_bins=(40, 60), min_occupancy=10, ncols=6, cmap="magma",
):
    """Small multiples of the predicted-rate heatmap, one panel per unit
    (default all), each with its own color scale. Returns (fig, axes)."""
    coef = np.asarray(coef)
    unit_idx = list(range(coef.shape[1])) if unit_idx is None else list(unit_idx)
    x = df["position_x"].values.astype(float)
    y = df["position_y"].values.astype(float)
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins[0] + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), n_bins[1] + 1)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    predicted = predict_firing_rate(X, coef[:, unit_idx], np.asarray(intercept)[unit_idx], bin_size)
    nrows = int(np.ceil(len(unit_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k, u in zip(axes.ravel(), range(len(unit_idx)), unit_idx):
        m, _ = _binned_mean_map(predicted[:, k], x, y, x_edges, y_edges, min_occupancy)
        ax.axis("on")
        im = ax.imshow(m, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(unit_names[u] if unit_names is not None else f"unit {u}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Predicted firing rate (Hz) over 2D position")
    plt.tight_layout()
    return fig, axes
