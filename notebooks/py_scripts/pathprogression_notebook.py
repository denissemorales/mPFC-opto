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
from spyglass.position import PositionOutput
import spyglass.common as sgc
import pandas as pd

from mpfc_opto.behavior.wtrack_tables import (
    WTrackParams,
    WTrackSelection,
    WTrackEvents,
)
# -

nwb_file_name = "Seth20251210_.nwb"

WTrackEvents() & {"nwb_file_name": nwb_file_name}

wtrack_results = pd.DataFrame(
    (WTrackEvents() & {"nwb_file_name": nwb_file_name}).fetch("wtrack_results")[
        0
    ]
)
wtrack_results

# +
import pandas as pd


def normalize_well(name):
    if pd.isna(name):
        return None
    name = str(name).replace("_poke", "").lower()
    if name in {"left", "right", "center"}:
        return name
    return name


# REVIEW: this infers a trajectory from prev_well and well only. It also
# read row['trial_type'] (Inbound/Outbound) and ignored it -- that dead
# assignment is gone, but if trial type should affect the result, the
# logic is missing. Mirrored in glm/path_progression_tables.py.
def infer_trajectory(row):
    prev_well = normalize_well(row["prev_well"])
    well = normalize_well(row["well_name"])
    # center-out
    if prev_well == "center" and well == "left":
        return "center_to_left"
    if prev_well == "center" and well == "right":
        return "center_to_right"

    # side-in
    if prev_well == "left" and well == "center":
        return "left_to_center"
    if prev_well == "right" and well == "center":
        return "right_to_center"

    # if the row is side-to-side, keep it separate or ignore it
    if prev_well == "left" and well == "right":
        return "left_to_right"
    if prev_well == "right" and well == "left":
        return "right_to_left"

    return pd.NA


# -

wtrack_results["trajectory"] = wtrack_results.apply(infer_trajectory, axis=1)

wtrack_results

# +
nwb_file_name = "Seth20251210_.nwb"
pos_key = {
    "nwb_file_name": nwb_file_name,
    "epoch": 2,
    "project_name": "w-track_Seth",
}

# Note: You'll have to change the part table to the one where your data came from
merge_id = (PositionOutput.DLCPosV1() & pos_key).fetch1("merge_id")
position_info = (PositionOutput & {"merge_id": merge_id}).fetch1_dataframe()
position_info

# +
linear_key = {
    "pos_merge_id": merge_id,
    "track_graph_name": "w-track_Seth",
    "linearization_param_name": "default",
}

from spyglass.linearization.merge import LinearizedPositionOutput

linear_merge_key = LinearizedPositionOutput.merge_restrict(linear_key).fetch1(
    "KEY"
)
linear_position_df = (
    LinearizedPositionOutput & linear_merge_key
).fetch1_dataframe()
linear_position_df
# -

linear_position_df = pd.merge_asof(
    linear_position_df,
    wtrack_results[["time", "trajectory"]].sort_values("time"),
    on="time",
    direction="backward",
)

linear_position_df["trajectory"].value_counts(dropna=False)

# +
import matplotlib.pyplot as plt
from neurospatial.behavior.navigation import path_progress
import numpy as np
import spyglass.linearization.v1 as sgpl
from neurospatial import Environment

track_graph = (
    sgpl.TrackGraph & {"track_graph_name": "w-track_Seth"}
).get_networkx_track_graph()
edge_order = (sgpl.TrackGraph & {"track_graph_name": "w-track_Seth"}).fetch1(
    "linear_edge_order"
)
edge_spacing = (sgpl.TrackGraph & {"track_graph_name": "w-track_Seth"}).fetch1(
    "linear_edge_spacing"
)
edge_order = [(int(a), int(b)) for a, b in edge_order]

env = Environment.from_graph(
    track_graph,
    edge_order=edge_order,
    edge_spacing=edge_spacing,
    bin_size=2.5,
    name="w-track_Seth",
)

# Define goal positions in 2D (x, y) for each arm end
goal_positions_2d = {
    "left_arm": np.array([[196.9, 111.8]]),
    "right_arm": np.array([[139.3, 144.1]]),
    "center": np.array([[168.1, 128.0]]),
}

# Convert 2D positions to bin indices
start_bin = env.bin_at(np.array([[168.1, 128.0]]))[0]  # center/home
left_bin = env.bin_at(np.array([[196.9, 111.8]]))[0]
right_bin = env.bin_at(np.array([[139.3, 144.1]]))[0]

print(start_bin, left_bin, right_bin)  # verify these are valid (>= 0)

# +
position_bins = env.bin_at(
    position_info[["position_x", "position_y"]].values  # shape (n, 2)
).astype(int)

trajectory_goals = {
    "center_to_left": (right_bin, start_bin),
    "left_to_center": (start_bin, left_bin),
    "center_to_right": (left_bin, start_bin),
    "right_to_center": (start_bin, right_bin),
    "left_to_right": (left_bin, right_bin),
    "right_to_left": (right_bin, left_bin),
}

node_lin = env.to_linear(
    np.array([[168.1, 128], [196.9, 111.8], [139.3, 144.1]])
)

start_bins = np.full(len(position_bins), -1)
goal_bins = np.full(len(position_bins), -1)

for traj, (s, g) in trajectory_goals.items():
    mask = (linear_position_df["trajectory"] == traj).values
    start_bins[mask] = s
    goal_bins[mask] = g

progress = path_progress(env, position_bins, start_bins, goal_bins)
linear_position_df["trajectory_progress"] = progress

# +
plt.figure(figsize=(12, 4))
for traj, g in linear_position_df.dropna(
    subset=["trajectory_progress"]
).groupby("trajectory"):
    plt.scatter(g["time"], g["trajectory_progress"], s=4, label=traj)

plt.xlabel("Time")
plt.ylabel("path progress")
plt.title("path progress")
plt.ylim(-0.05, 1.05)
plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# +
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

group_key = {
    "nwb_file_name": "Seth20251210_.nwb",
    "sorted_spikes_group_name": "mPFC",
}
SortedSpikesGroup & group_key
# -

group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
SortedSpikesGroup().fetch_spike_data(group_key)

spike_times, unit_ids = SortedSpikesGroup().fetch_spike_data(
    group_key, return_unit_ids=True
)
print(unit_ids[0])
print(spike_times[0])

len(spike_times)

# +
# from pathprogressiontuning import compute_tuning_curve

path_times_arr = linear_position_df["trajectory_progress"].index.to_numpy(
    dtype=float
)
path_values_arr = linear_position_df["trajectory_progress"].to_numpy(
    dtype=float
)

# +
# Cell 2
bins = np.linspace(0, 1, 20 + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_neurons = len(spike_times)  # ← your variable name

all_rates = np.full((n_neurons, 20), np.nan)

# for i, spk in enumerate(spike_times):  # ← your variable name
#     rate, _ = compute_tuning_curve(spk, path_times_arr, path_values_arr, bins)
#     all_rates[i] = rate
# -

plt.plot(path_values_arr)

path_df = linear_position_df
path_df["path_progression"] = linear_position_df["trajectory_progress"]

# +
session_start = path_df["time"].iloc[0]

spike_times_relative = [spk - session_start for spk in spike_times]

# verify
path_end = path_df.index[-1]
for i in range(5):
    spk = spike_times_relative[i]
    n_in = np.sum((spk >= 0) & (spk <= path_end))
    print(f"neuron {i}: {n_in}/{len(spk)} spikes inside maze window")

# +
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

N_BINS = 20
SMOOTH_SIGMA = 1.0
MIN_OCCUPANCY_S = 0.05
N_SHUFFLES = 500
ALPHA = 0.05
SPEED_THRESHOLD = None
OUTPUT_DIR = "."

# EDIT: If your path_df has discontinuous path traversals, this prevents interpolation
# across large behavioral gaps. If None, it is inferred as 5 * median dt.
MAX_TIME_GAP_S = None


def get_contiguous_segments(
    times: np.ndarray,
    max_time_gap_s: float | None = None,
) -> list[slice]:
    """Find contiguous behavioral segments.

    Parameters
    ----------
    times : np.ndarray
        Time values with shape ``(n_time,)``.
    max_time_gap_s : float | None, default=None
        Maximum allowed gap between adjacent time samples. If None, use
        ``5 * median(diff(times))``.

    Returns
    -------
    segments : list[slice]
        List of slices defining contiguous time segments.
    """
    times = np.asarray(times, dtype=float)

    if times.size < 2:
        return [slice(0, times.size)]

    dt = np.diff(times)
    if max_time_gap_s is None:
        max_time_gap_s = 5.0 * float(np.nanmedian(dt))

    break_points = np.flatnonzero(dt > max_time_gap_s) + 1
    segment_starts = np.r_[0, break_points]
    segment_stops = np.r_[break_points, times.size]

    return [
        slice(start, stop)
        for start, stop in zip(segment_starts, segment_stops)
        if stop - start >= 2
    ]


def compute_tuning_curve(
    spike_times: np.ndarray,
    path_times: np.ndarray,
    path_values: np.ndarray,
    bins: np.ndarray,
    min_occupancy_s: float = MIN_OCCUPANCY_S,
    smooth_sigma: float = SMOOTH_SIGMA,
    max_time_gap_s: float | None = MAX_TIME_GAP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute path-progression tuning curve for one neuron.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times for one neuron, with shape ``(n_spikes,)``.
    path_times : np.ndarray
        Behavioral timestamps, with shape ``(n_time,)``.
    path_values : np.ndarray
        Path-progression values aligned to ``path_times``, with shape
        ``(n_time,)``.
    bins : np.ndarray
        Path-progression bin edges, with shape ``(n_bins + 1,)``.
    min_occupancy_s : float, default=MIN_OCCUPANCY_S
        Minimum occupancy in seconds required for a bin to be valid.
    smooth_sigma : float, default=SMOOTH_SIGMA
        Gaussian smoothing sigma in bin units.
    max_time_gap_s : float | None, default=MAX_TIME_GAP_S
        Maximum allowed gap between adjacent behavioral time samples.
        This prevents interpolation across discontinuous path traversals.

    Returns
    -------
    rate : np.ndarray
        Firing rate in Hz, with shape ``(n_bins,)``.
    occupancy : np.ndarray
        Occupancy in seconds, with shape ``(n_bins,)``.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    path_times = np.asarray(path_times, dtype=float)
    path_values = np.asarray(path_values, dtype=float)

    n_bins = len(bins) - 1

    finite_path_mask = np.isfinite(path_times) & np.isfinite(path_values)
    path_times = path_times[finite_path_mask]
    path_values = path_values[finite_path_mask]

    sort_order = np.argsort(path_times)
    path_times = path_times[sort_order]
    path_values = path_values[sort_order]

    occupancy = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    segments = get_contiguous_segments(
        times=path_times,
        max_time_gap_s=max_time_gap_s,
    )

    for segment in segments:
        segment_times = path_times[segment]
        segment_values = path_values[segment]

        if segment_times.size < 2:
            continue

        dt = float(np.nanmedian(np.diff(segment_times)))

        bin_idx = np.clip(
            np.digitize(segment_values, bins) - 1,
            0,
            n_bins - 1,
        )
        occupancy += np.bincount(bin_idx, minlength=n_bins) * dt

        # EDIT: Only use spikes inside this actual behavioral segment.
        in_segment = (spike_times >= segment_times[0]) & (
            spike_times <= segment_times[-1]
        )
        segment_spike_times = spike_times[in_segment]

        if segment_spike_times.size == 0:
            continue

        # EDIT: left/right NaN prevents endpoint clamping by np.interp.
        spike_pos = np.interp(
            segment_spike_times,
            segment_times,
            segment_values,
            left=np.nan,
            right=np.nan,
        )

        valid_spike_pos = (
            np.isfinite(spike_pos)
            & (spike_pos >= bins[0])
            & (spike_pos <= bins[-1])
        )
        spike_pos = spike_pos[valid_spike_pos]

        segment_counts, _ = np.histogram(spike_pos, bins=bins)
        counts += segment_counts

    rate = np.full(n_bins, np.nan)
    valid = occupancy > min_occupancy_s
    rate[valid] = counts[valid] / occupancy[valid]

    if smooth_sigma > 0:
        filled = np.where(valid, rate, 0.0)
        smoothed = gaussian_filter1d(filled, sigma=smooth_sigma)

        valid_smoothed = gaussian_filter1d(
            valid.astype(float), sigma=smooth_sigma
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(
                valid_smoothed > 0.01,
                smoothed / valid_smoothed,
                np.nan,
            )

        rate = np.where(valid, smoothed, np.nan)

    return rate, occupancy


def run_shuffle_test(
    spike_times: np.ndarray,
    path_times: np.ndarray,
    path_values: np.ndarray,
    bins: np.ndarray,
    n_shuffles: int = N_SHUFFLES,
    max_time_gap_s: float | None = MAX_TIME_GAP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute circular-shift shuffle null distribution.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times for one neuron, with shape ``(n_spikes,)``.
    path_times : np.ndarray
        Behavioral timestamps, with shape ``(n_time,)``.
    path_values : np.ndarray
        Path-progression values aligned to ``path_times``, with shape
        ``(n_time,)``.
    bins : np.ndarray
        Path-progression bin edges, with shape ``(n_bins + 1,)``.
    n_shuffles : int, default=N_SHUFFLES
        Number of circular shuffles.
    max_time_gap_s : float | None, default=MAX_TIME_GAP_S
        Maximum allowed gap between adjacent behavioral samples.

    Returns
    -------
    p95_per_bin : np.ndarray
        Per-bin shuffle threshold, with shape ``(n_bins,)``.
    mean_per_bin : np.ndarray
        Per-bin shuffle mean, with shape ``(n_bins,)``.
    """
    path_times = np.asarray(path_times, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float)

    t_start = float(path_times[0])
    t_stop = float(path_times[-1])
    total_duration = t_stop - t_start

    shuffle_rates = np.full((n_shuffles, len(bins) - 1), np.nan)

    for i_shuffle in range(n_shuffles):
        # EDIT: Keep the original absolute time origin.
        shift = np.random.uniform(20.0, total_duration - 20.0)
        shuffled = ((spike_times - t_start + shift) % total_duration) + t_start

        rate, _ = compute_tuning_curve(
            spike_times=shuffled,
            path_times=path_times,
            path_values=path_values,
            bins=bins,
            max_time_gap_s=max_time_gap_s,
        )
        shuffle_rates[i_shuffle] = rate

    return (
        np.nanpercentile(shuffle_rates, 100 * (1 - ALPHA), axis=0),
        np.nanmean(shuffle_rates, axis=0),
    )


def safe_row_normalize(rates: np.ndarray) -> np.ndarray:
    """Normalize each row to its maximum finite value.

    Parameters
    ----------
    rates : np.ndarray
        Tuning curves with shape ``(n_neurons, n_bins)``.

    Returns
    -------
    norm_rates : np.ndarray
        Row-normalized tuning curves with shape ``(n_neurons, n_bins)``.
    """
    rates = np.asarray(rates, dtype=float)

    row_max = np.nanmax(rates, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)

    return rates / row_max


# +
session_start = path_df["time"].iloc[0]
spike_times_relative = [spk - session_start for spk in spike_times]

N_BINS = 20

path_times_arr = path_df.index.to_numpy(dtype=float)
path_values_arr = path_df["path_progression"].to_numpy(dtype=float)

n_neurons = len(spike_times_relative)
all_rates = np.full((n_neurons, N_BINS), np.nan)

print(f"Computing tuning curves for {n_neurons} neurons ...")
for i, spk in enumerate(spike_times_relative):
    all_rates[i], _ = compute_tuning_curve(
        spk, path_times_arr, path_values_arr, bins
    )

for i, spk in enumerate(spike_times_relative):
    if spk.size < 10:
        continue
    p95, _ = run_shuffle_test(spk, path_times_arr, path_values_arr, bins)
is_significant = np.zeros(n_neurons, dtype=bool)
shuffle_p95_all = np.full((n_neurons, N_BINS), np.nan)

for i, spk in enumerate(spike_times_relative):
    if spk.size < 10:
        continue
    p95, _ = run_shuffle_test(spk, path_times_arr, path_values_arr, bins)
    shuffle_p95_all[i] = p95
    peak = int(np.nanargmax(all_rates[i]))
    if not np.isnan(all_rates[i, peak]) and not np.isnan(p95[peak]):
        is_significant[i] = all_rates[i, peak] > p95[peak]

n_sig = int(is_significant.sum())
print(f"{n_sig}/{n_neurons} neurons significantly tuned (α={ALPHA})")

row_max = np.where(
    np.nanmax(all_rates, axis=1, keepdims=True) == 0,
    1,
    np.nanmax(all_rates, axis=1, keepdims=True),
)
norm_rates = all_rates / row_max
peak_bin = np.nanargmax(norm_rates, axis=1)
sort_order = np.argsort(peak_bin)
bin_centers = (bins[:-1] + bins[1:]) / 2

# +
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SIG_COLOR = "#58a6ff"
INSIG_COLOR = "#484f58"
SHUF_COLOR = "#f0883e"

# ── Plot: per-neuron tuning curve grid ───────────────────────────────────────

n_cols = 8
n_rows = int(np.ceil(n_neurons / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 2.0, n_rows * 1.6),
    facecolor="#0e1117",
    squeeze=False,
)
fig.suptitle(
    "Path-Progression Tuning Curves", fontsize=14, color="#e6edf3", y=1.01
)

for idx in range(n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    ax = axes[row, col]
    if idx >= n_neurons:
        ax.set_visible(False)
        continue
    rate = all_rates[idx]
    p95 = shuffle_p95_all[idx]
    sig = is_significant[idx]
    color = SIG_COLOR if sig else INSIG_COLOR
    ax.fill_between(bin_centers, np.nan_to_num(rate), alpha=0.25, color=color)
    ax.plot(bin_centers, rate, color=color, lw=1.2)
    if not np.all(np.isnan(p95)):
        ax.plot(
            bin_centers,
            p95,
            color=SHUF_COLOR,
            lw=0.8,
            linestyle="--",
            alpha=0.7,
        )
    peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
    ax.set_ylim(0, max(peak * 1.2, 0.1))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_title(
        f"n{idx}", fontsize=6, pad=2, color=SIG_COLOR if sig else "#8b949e"
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.grid(True, axis="y", linewidth=0.4)

fig.text(
    0.5, -0.01, "Path progression", ha="center", fontsize=9, color="#8b949e"
)
fig.text(
    -0.01,
    0.5,
    "Firing rate (Hz)",
    va="center",
    rotation="vertical",
    fontsize=9,
    color="#8b949e",
)
fig.legend(
    handles=[
        Line2D(
            [0], [0], color=SIG_COLOR, lw=1.5, label=f"Significant (n={n_sig})"
        ),
        Line2D([0], [0], color=INSIG_COLOR, lw=1.5, label="Not significant"),
        Line2D(
            [0],
            [0],
            color=SHUF_COLOR,
            lw=1.0,
            linestyle="--",
            label=f"Shuffle 95th pct (α={ALPHA})",
        ),
    ],
    loc="lower center",
    ncol=3,
    fontsize=7,
    framealpha=0.3,
    bbox_to_anchor=(0.5, -0.04),
)
plt.tight_layout()
plt.show()

# ── Plot: population heatmap ─────────────────────────────────────────────────

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 7),
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    facecolor="#0e1117",
)
sorted_norm = norm_rates[sort_order]
im = axes[0].imshow(
    sorted_norm,
    aspect="auto",
    origin="lower",
    cmap="magma",
    norm=Normalize(vmin=0, vmax=1),
    extent=[0, 1, 0, n_neurons],
    interpolation="nearest",
)
axes[0].set_xlabel("Path progression", fontsize=10)
axes[0].set_ylabel("Neuron (sorted by peak)", fontsize=10)
axes[0].set_title(
    "Population tuning — sorted by preferred position", fontsize=11
)
sig_sorted = is_significant[sort_order]
for y_idx, sig in enumerate(sig_sorted):
    if sig:
        axes[0].axhline(
            y_idx + 0.5, color=SIG_COLOR, lw=0.3, alpha=0.4, xmin=0, xmax=0.02
        )
cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
cbar.set_label("Normalised rate", fontsize=8)
cbar.ax.tick_params(labelsize=7)
axes[1].imshow(
    sig_sorted.astype(float).reshape(-1, 1),
    aspect="auto",
    origin="lower",
    cmap="Blues",
    vmin=0,
    vmax=1,
    extent=[0, 1, 0, n_neurons],
    interpolation="nearest",
)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title(f"Sig.\n(α={ALPHA})", fontsize=8)
axes[1].set_xlabel(f"{n_sig}/{n_neurons}", fontsize=8)
fig.suptitle(
    "Path-Progression Population Heatmap", fontsize=13, color="#e6edf3", y=1.01
)
plt.tight_layout()
plt.show()

# ── Plot: shuffle significance examples ──────────────────────────────────────

sig_indices = np.where(is_significant)[0]
# insig_indices = np.where(~is_significant & has_valid_rate)[0]

if sig_indices.size > 0:
    margins = np.array(
        [
            (
                np.nanmax(all_rates[i]) - np.nanmax(shuffle_p95_all[i])
                if not np.all(np.isnan(shuffle_p95_all[i]))
                else -np.inf
            )
            for i in sig_indices
        ]
    )
    top_sig = sig_indices[np.argsort(-margins)[:6]]
else:
    top_sig = np.array([], dtype=int)

# REVIEW: insig_indices is assigned on a commented-out line above, so
# this whole branch raises NameError. Re-enable that line or drop it.
if insig_indices.size > 0:  # noqa: F821
    insig_peak_rates = np.array(
        [
            (
                np.nanmax(all_rates[i])
                if not np.all(np.isnan(all_rates[i]))
                else -np.inf
            )
            for i in insig_indices  # noqa: F821
        ]
    )
    top_insig = insig_indices[np.argsort(-insig_peak_rates)[:6]]  # noqa: F821
else:
    top_insig = np.array([], dtype=int)

example_neurons = np.concatenate([top_sig, top_insig]).astype(int)
n_examples = len(example_neurons)

if n_examples > 0:
    n_cols_ex = max(int(np.ceil(n_examples / 2)), 1)
    fig, axes = plt.subplots(
        2,
        n_cols_ex,
        figsize=(n_cols_ex * 2.5, 5.5),
        facecolor="#0e1117",
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"Shuffle Significance  ·  {n_sig}/{n_neurons} tuned  (α={ALPHA}, {N_SHUFFLES} shuffles)",
        fontsize=11,
        color="#e6edf3",
        y=1.02,
    )
    for plot_idx, neuron_idx in enumerate(example_neurons):
        row = plot_idx // n_cols_ex
        col = plot_idx % n_cols_ex
        ax = axes[row, col]
        rate = all_rates[neuron_idx]
        p95 = shuffle_p95_all[neuron_idx]
        sig = is_significant[neuron_idx]
        color = SIG_COLOR if sig else INSIG_COLOR
        ax.fill_between(
            bin_centers, np.nan_to_num(rate), alpha=0.2, color=color
        )
        ax.plot(bin_centers, rate, color=color, lw=1.5)
        if not np.all(np.isnan(p95)):
            ax.fill_between(
                bin_centers, np.nan_to_num(p95), alpha=0.12, color=SHUF_COLOR
            )
            ax.plot(
                bin_centers,
                p95,
                color=SHUF_COLOR,
                lw=1.0,
                linestyle="--",
                alpha=0.9,
            )
        peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
        ax.set_ylim(0, max(peak * 1.25, 0.1))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Path progression", fontsize=7)
        ax.set_ylabel("Hz", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"{unit_ids[idx]}",
            fontsize=6,
            pad=2,
            color=SIG_COLOR if sig else "#8b949e",
        )
        ax.grid(True, axis="y", linewidth=0.4)

    for plot_idx in range(n_examples, axes.size):
        axes[plot_idx // n_cols_ex, plot_idx % n_cols_ex].set_visible(False)

    fig.legend(
        handles=[
            Line2D([0], [0], color=SIG_COLOR, lw=1.5, label="Observed rate"),
            Line2D(
                [0],
                [0],
                color=SHUF_COLOR,
                lw=1.0,
                linestyle="--",
                label="Shuffle 95th pct",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.3,
        bbox_to_anchor=(0.5, -0.03),
    )
    plt.tight_layout()
    plt.show()
else:
    print("No valid example neurons to plot.")

# +
sig_indices = np.where(is_significant)[0]
n_sig_neurons = len(sig_indices)

n_cols = 8
n_rows = int(np.ceil(n_sig_neurons / n_cols))
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 1.6), facecolor="#0e1117"
)
axes = np.atleast_2d(axes)  # safe if only 1 row
fig.suptitle(
    f"Significant Tuning Curves — {traj} (n={n_sig_neurons})",
    fontsize=14,
    color="#e6edf3",
    y=1.01,
)

for plot_idx in range(n_rows * n_cols):
    row, col = divmod(plot_idx, n_cols)
    ax = axes[row, col]
    if plot_idx >= n_sig_neurons:
        ax.set_visible(False)
        continue
    neuron_idx = sig_indices[plot_idx]  # ← actual neuron index
    rate = all_rates[neuron_idx]
    p95 = shuffle_p95_all[neuron_idx]

    ax.fill_between(
        bin_centers, np.nan_to_num(rate), alpha=0.25, color=SIG_COLOR
    )
    ax.plot(bin_centers, rate, color=SIG_COLOR, lw=1.2)
    if not np.all(np.isnan(p95)):
        ax.plot(
            bin_centers,
            p95,
            color=SHUF_COLOR,
            lw=0.8,
            linestyle="--",
            alpha=0.7,
        )
    peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
    ax.set_ylim(0, max(peak * 1.2, 0.1))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_title(f"n{neuron_idx}", fontsize=6, pad=2, color=SIG_COLOR)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.grid(True, axis="y", linewidth=0.4)

# +
# Cell 3 — trajectory-separated tuning curves
session_start = path_df["time"].iloc[0]
spike_times_relative = [spk - session_start for spk in spike_times]

bins = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_neurons = len(spike_times_relative)
trajectories = [t for t in path_df["trajectory"].unique() if pd.notna(t)]
print(f"Found trajectories: {trajectories}")

# results are stored per trajectory
all_rates_by_traj = {}
shuffle_p95_by_traj = {}
is_significant_by_traj = {}
norm_rates_by_traj = {}
sort_order_by_traj = {}

for traj in trajectories:
    print(f"\nTrajectory: {traj}")
    mask = path_df["trajectory"] == traj
    traj_times = path_df.index[mask].to_numpy(dtype=float)
    traj_values = path_df.loc[mask, "path_progression"].to_numpy(dtype=float)

    all_rates = np.full((n_neurons, N_BINS), np.nan)
    print(f"  Computing tuning curves ({traj_times.size} samples) ...")
    for i, spk in enumerate(spike_times_relative):
        all_rates[i], _ = compute_tuning_curve(
            spk, traj_times, traj_values, bins
        )

    print(f"  Running {N_SHUFFLES} shuffles ...")
    shuffle_p95_all = np.full((n_neurons, N_BINS), np.nan)
    is_significant = np.zeros(n_neurons, dtype=bool)
    for i, spk in enumerate(spike_times_relative):
        if spk.size < 10:
            continue
        p95, _ = run_shuffle_test(spk, traj_times, traj_values, bins)
        shuffle_p95_all[i] = p95
        peak = int(np.nanargmax(all_rates[i]))
        if not np.isnan(all_rates[i, peak]) and not np.isnan(p95[peak]):
            is_significant[i] = all_rates[i, peak] > p95[peak]

    n_sig = int(is_significant.sum())
    print(f"  {n_sig}/{n_neurons} significant (α={ALPHA})")

    row_max = np.where(
        np.nanmax(all_rates, axis=1, keepdims=True) == 0,
        1,
        np.nanmax(all_rates, axis=1, keepdims=True),
    )
    norm_rates = all_rates / row_max
    peak_bin = np.nanargmax(norm_rates, axis=1)

    all_rates_by_traj[traj] = all_rates
    shuffle_p95_by_traj[traj] = shuffle_p95_all
    is_significant_by_traj[traj] = is_significant
    norm_rates_by_traj[traj] = norm_rates
    sort_order_by_traj[traj] = np.argsort(peak_bin)
# -

# Cell 5 — per-neuron grid, one figure per trajectory
for traj in trajectories:
    all_rates = all_rates_by_traj[traj]
    shuffle_p95_all = shuffle_p95_by_traj[traj]
    is_significant = is_significant_by_traj[traj]
    n_sig = int(is_significant.sum())

    n_cols = 8
    n_rows = int(np.ceil(n_neurons / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.0, n_rows * 1.6),
        facecolor="#0e1117",
    )
    fig.suptitle(
        f"Tuning Curves — {traj}", fontsize=14, color="#e6edf3", y=1.01
    )

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        if idx >= n_neurons:
            ax.set_visible(False)
            continue
        rate = all_rates[idx]
        p95 = shuffle_p95_all[idx]
        sig = is_significant[idx]
        color = SIG_COLOR if sig else INSIG_COLOR

        ax.fill_between(
            bin_centers, np.nan_to_num(rate), alpha=0.25, color=color
        )
        ax.plot(bin_centers, rate, color=color, lw=1.2)
        if not np.all(np.isnan(p95)):
            ax.plot(
                bin_centers,
                p95,
                color=SHUF_COLOR,
                lw=0.8,
                linestyle="--",
                alpha=0.7,
            )
        peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
        ax.set_ylim(0, max(peak * 1.2, 0.1))
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", ".5", "1"], fontsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.set_title(
            f"n{idx}", fontsize=6, pad=2, color=SIG_COLOR if sig else "#8b949e"
        )
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.grid(True, axis="y", linewidth=0.4)

    fig.text(
        0.5, -0.01, "Path progression", ha="center", fontsize=9, color="#8b949e"
    )
    fig.text(
        -0.01,
        0.5,
        "Firing rate (Hz)",
        va="center",
        rotation="vertical",
        fontsize=9,
        color="#8b949e",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=SIG_COLOR,
                lw=1.5,
                label=f"Significant (n={n_sig})",
            ),
            Line2D(
                [0], [0], color=INSIG_COLOR, lw=1.5, label="Not significant"
            ),
            Line2D(
                [0],
                [0],
                color=SHUF_COLOR,
                lw=1.0,
                linestyle="--",
                label=f"Shuffle 95th pct (α={ALPHA})",
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.3,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()
    fig.savefig(
        f"{OUTPUT_DIR}/tuning_curves_grid_{traj}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0e1117",
    )
    plt.show()

# Cell 6 — population heatmap, one figure per trajectory
for traj in trajectories:
    norm_rates = norm_rates_by_traj[traj]
    sort_order = sort_order_by_traj[traj]
    is_significant = is_significant_by_traj[traj]
    n_sig = int(is_significant.sum())
    sorted_norm = norm_rates[sort_order]
    sig_sorted = is_significant[sort_order]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 7),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        facecolor="#0e1117",
    )
    im = axes[0].imshow(
        sorted_norm,
        aspect="auto",
        origin="lower",
        cmap="magma",
        norm=Normalize(vmin=0, vmax=1),
        extent=[0, 1, 0, n_neurons],
        interpolation="nearest",
    )
    axes[0].set_xlabel("Path progression", fontsize=10)
    axes[0].set_ylabel("Neuron (sorted by peak)", fontsize=10)
    axes[0].set_title(f"Population tuning — {traj}", fontsize=11)
    for y_idx, sig in enumerate(sig_sorted):
        if sig:
            axes[0].axhline(
                y_idx + 0.5,
                color=SIG_COLOR,
                lw=0.3,
                alpha=0.4,
                xmin=0,
                xmax=0.02,
            )
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
    cbar.set_label("Normalised rate", fontsize=8)
    axes[1].imshow(
        sig_sorted.astype(float).reshape(-1, 1),
        aspect="auto",
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=[0, 1, 0, n_neurons],
        interpolation="nearest",
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(f"Sig.\n(α={ALPHA})", fontsize=8)
    axes[1].set_xlabel(f"{n_sig}/{n_neurons}", fontsize=8)
    fig.suptitle(
        f"Population Heatmap — {traj}", fontsize=13, color="#e6edf3", y=1.01
    )
    plt.tight_layout()
    fig.savefig(
        f"{OUTPUT_DIR}/population_heatmap_{traj}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0e1117",
    )
    plt.show()
plt.show()

# +
# get_contiguous_segments, compute_tuning_curve, run_shuffle_test and
# safe_row_normalize come from the cell above; run it first.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

N_BINS = 20
SMOOTH_SIGMA = 1.0
MIN_OCCUPANCY_S = 0.05
N_SHUFFLES = 500
ALPHA = 0.05
SPEED_THRESHOLD = None
OUTPUT_DIR = "."

# EDIT: If your path_df has discontinuous path traversals, this prevents interpolation
# across large behavioral gaps. If None, it is inferred as 5 * median dt.
MAX_TIME_GAP_S = None


# ── Compute tuning curves ────────────────────────────────────────────────────

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

path_times_arr = path_df.index.to_numpy(dtype=float)
path_values_arr = path_df["path_progression"].to_numpy(dtype=float)

finite_path_mask = np.isfinite(path_times_arr) & np.isfinite(path_values_arr)
path_times_arr = path_times_arr[finite_path_mask]
path_values_arr = path_values_arr[finite_path_mask]

sort_order_path = np.argsort(path_times_arr)
path_times_arr = path_times_arr[sort_order_path]
path_values_arr = path_values_arr[sort_order_path]

bins = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_neurons = len(spike_times)

all_rates = np.full((n_neurons, N_BINS), np.nan)
all_occupancy = np.full((n_neurons, N_BINS), np.nan)

print(f"Computing tuning curves for {n_neurons} neurons ...")

for i_neuron, spk in enumerate(spike_times):
    all_rates[i_neuron], all_occupancy[i_neuron] = compute_tuning_curve(
        spike_times=np.asarray(spk, dtype=float),
        path_times=path_times_arr,
        path_values=path_values_arr,
        bins=bins,
    )

# Normalize rows to [0, 1] for heatmap; find peak bin per neuron.
norm_rates = safe_row_normalize(all_rates)

# EDIT: Avoid nanargmax crashing or behaving unexpectedly for all-NaN rows.
has_valid_rate = np.any(np.isfinite(norm_rates), axis=1)
peak_bin = np.zeros(n_neurons, dtype=int)
peak_bin[has_valid_rate] = np.nanargmax(norm_rates[has_valid_rate], axis=1)

sort_order = np.lexsort((np.arange(n_neurons), peak_bin))


# ── Shuffle significance ─────────────────────────────────────────────────────

print(f"Running {N_SHUFFLES} shuffles per neuron ...")

is_significant = np.zeros(n_neurons, dtype=bool)
shuffle_p95_all = np.full((n_neurons, N_BINS), np.nan)

for i_neuron, spk in enumerate(spike_times):
    spk = np.asarray(spk, dtype=float)

    if spk.size < 10 or not has_valid_rate[i_neuron]:
        continue

    p95, _ = run_shuffle_test(
        spike_times=spk,
        path_times=path_times_arr,
        path_values=path_values_arr,
        bins=bins,
    )
    shuffle_p95_all[i_neuron] = p95

    peak = int(peak_bin[i_neuron])
    if np.isfinite(all_rates[i_neuron, peak]) and np.isfinite(p95[peak]):
        is_significant[i_neuron] = all_rates[i_neuron, peak] > p95[peak]

n_sig = int(is_significant.sum())
print(f"{n_sig}/{n_neurons} neurons significantly tuned (α={ALPHA})")


# ── Sanity checks ─────────────────────────────────────────────────────────────

print("\nSanity check: first few spike arrays")
for i_neuron in range(min(5, n_neurons)):
    spk = np.asarray(spike_times[i_neuron], dtype=float)
    print(f"neuron {i_neuron}: shape={spk.shape}, first_spikes={spk[:5]}")

print("\nSanity check: identical spike arrays")
for i_neuron in range(1, min(5, n_neurons)):
    print(
        f"cell 0 identical to cell {i_neuron}: "
        f"{np.array_equal(np.asarray(spike_times[0]), np.asarray(spike_times[i_neuron]))}"
    )


# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SIG_COLOR = "#58a6ff"
INSIG_COLOR = "#484f58"
SHUF_COLOR = "#f0883e"


# ── Plot: per-neuron tuning curve grid ───────────────────────────────────────

n_cols = 8
n_rows = int(np.ceil(n_neurons / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 2.0, n_rows * 1.6),
    facecolor="#0e1117",
    squeeze=False,
)

fig.suptitle(
    "Path-Progression Tuning Curves",
    fontsize=14,
    color="#e6edf3",
    y=1.01,
)

for idx in range(n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    ax = axes[row, col]

    if idx >= n_neurons:
        ax.set_visible(False)
        continue

    rate = all_rates[idx]
    p95 = shuffle_p95_all[idx]
    sig = is_significant[idx]
    color = SIG_COLOR if sig else INSIG_COLOR

    ax.fill_between(bin_centers, np.nan_to_num(rate), alpha=0.25, color=color)
    ax.plot(bin_centers, rate, color=color, lw=1.2)

    if not np.all(np.isnan(p95)):
        ax.plot(
            bin_centers,
            p95,
            color=SHUF_COLOR,
            lw=0.8,
            linestyle="--",
            alpha=0.7,
        )

    peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
    ax.set_ylim(0, max(peak * 1.2, 0.1))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_title(
        f"n{idx}",
        fontsize=6,
        pad=2,
        color=SIG_COLOR if sig else "#8b949e",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.grid(True, axis="y", linewidth=0.4)

fig.text(
    0.5,
    -0.01,
    "Path progression",
    ha="center",
    fontsize=9,
    color="#8b949e",
)
fig.text(
    -0.01,
    0.5,
    "Firing rate (Hz)",
    va="center",
    rotation="vertical",
    fontsize=9,
    color="#8b949e",
)

fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=SIG_COLOR,
            lw=1.5,
            label=f"Significant (n={n_sig})",
        ),
        Line2D(
            [0],
            [0],
            color=INSIG_COLOR,
            lw=1.5,
            label="Not significant",
        ),
        Line2D(
            [0],
            [0],
            color=SHUF_COLOR,
            lw=1.0,
            linestyle="--",
            label=f"Shuffle 95th pct (α={ALPHA})",
        ),
    ],
    loc="lower center",
    ncol=3,
    fontsize=7,
    framealpha=0.3,
    bbox_to_anchor=(0.5, -0.04),
)

plt.tight_layout()
fig.savefig(
    output_dir / "tuning_curves_grid.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0e1117",
)
plt.show()


# ── Plot: population heatmap ─────────────────────────────────────────────────

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 7),
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    facecolor="#0e1117",
)

sorted_norm = norm_rates[sort_order]

im = axes[0].imshow(
    sorted_norm,
    aspect="auto",
    origin="lower",
    cmap="magma",
    norm=Normalize(vmin=0, vmax=1),
    extent=[0, 1, 0, n_neurons],
    interpolation="nearest",
)

axes[0].set_xlabel("Path progression", fontsize=10)
axes[0].set_ylabel("Neuron (sorted by peak)", fontsize=10)
axes[0].set_title(
    "Population tuning — sorted by preferred position", fontsize=11
)

sig_sorted = is_significant[sort_order]
for y_idx, sig in enumerate(sig_sorted):
    if sig:
        axes[0].axhline(
            y_idx + 0.5,
            color=SIG_COLOR,
            lw=0.3,
            alpha=0.4,
            xmin=0,
            xmax=0.02,
        )

cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
cbar.set_label("Normalised rate", fontsize=8)
cbar.ax.tick_params(labelsize=7)

axes[1].imshow(
    sig_sorted.astype(float).reshape(-1, 1),
    aspect="auto",
    origin="lower",
    cmap="Blues",
    vmin=0,
    vmax=1,
    extent=[0, 1, 0, n_neurons],
    interpolation="nearest",
)

axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title(f"Sig.\n(α={ALPHA})", fontsize=8)
axes[1].set_xlabel(f"{n_sig}/{n_neurons}", fontsize=8)

fig.suptitle(
    "Path-Progression Population Heatmap",
    fontsize=13,
    color="#e6edf3",
    y=1.01,
)

plt.tight_layout()
fig.savefig(
    output_dir / "population_heatmap.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0e1117",
)
plt.show()


# ── Plot: shuffle significance examples ──────────────────────────────────────

sig_indices = np.where(is_significant)[0]
insig_indices = np.where(~is_significant & has_valid_rate)[0]

if sig_indices.size > 0:
    margins = np.array(
        [
            (
                np.nanmax(all_rates[i]) - np.nanmax(shuffle_p95_all[i])
                if not np.all(np.isnan(shuffle_p95_all[i]))
                else -np.inf
            )
            for i in sig_indices
        ]
    )
    top_sig = sig_indices[np.argsort(-margins)[:6]]
else:
    top_sig = np.array([], dtype=int)

if insig_indices.size > 0:
    insig_peak_rates = np.array(
        [
            (
                np.nanmax(all_rates[i])
                if not np.all(np.isnan(all_rates[i]))
                else -np.inf
            )
            for i in insig_indices
        ]
    )
    top_insig = insig_indices[np.argsort(-insig_peak_rates)[:6]]
else:
    top_insig = np.array([], dtype=int)

example_neurons = np.concatenate([top_sig, top_insig]).astype(int)
n_examples = len(example_neurons)

if n_examples > 0:
    n_cols_ex = max(int(np.ceil(n_examples / 2)), 1)

    fig, axes = plt.subplots(
        2,
        n_cols_ex,
        figsize=(n_cols_ex * 2.5, 5.5),
        facecolor="#0e1117",
        sharey=False,
        squeeze=False,
    )

    fig.suptitle(
        (
            f"Shuffle Significance  ·  {n_sig}/{n_neurons} tuned  "
            f"(α={ALPHA}, {N_SHUFFLES} shuffles)"
        ),
        fontsize=11,
        color="#e6edf3",
        y=1.02,
    )

    for plot_idx, neuron_idx in enumerate(example_neurons):
        row = plot_idx // n_cols_ex
        col = plot_idx % n_cols_ex
        ax = axes[row, col]

        rate = all_rates[neuron_idx]
        p95 = shuffle_p95_all[neuron_idx]
        sig = is_significant[neuron_idx]
        color = SIG_COLOR if sig else INSIG_COLOR

        ax.fill_between(
            bin_centers, np.nan_to_num(rate), alpha=0.2, color=color
        )
        ax.plot(bin_centers, rate, color=color, lw=1.5)

        if not np.all(np.isnan(p95)):
            ax.fill_between(
                bin_centers,
                np.nan_to_num(p95),
                alpha=0.12,
                color=SHUF_COLOR,
            )
            ax.plot(
                bin_centers,
                p95,
                color=SHUF_COLOR,
                lw=1.0,
                linestyle="--",
                alpha=0.9,
            )

        peak = np.nanmax(rate) if not np.all(np.isnan(rate)) else 0
        ax.set_ylim(0, max(peak * 1.25, 0.1))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Path progression", fontsize=7)
        ax.set_ylabel("Hz", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"{unit_ids[idx]}",
            fontsize=6,
            pad=2,
            color=SIG_COLOR if sig else "#8b949e",
        )
        ax.grid(True, axis="y", linewidth=0.4)

    for plot_idx in range(n_examples, axes.size):
        axes[plot_idx // n_cols_ex, plot_idx % n_cols_ex].set_visible(False)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=SIG_COLOR,
                lw=1.5,
                label="Observed rate",
            ),
            Line2D(
                [0],
                [0],
                color=SHUF_COLOR,
                lw=1.0,
                linestyle="--",
                label="Shuffle 95th pct",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.3,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    fig.savefig(
        output_dir / "shuffle_significance.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0e1117",
    )
    plt.show()
else:
    print("No valid example neurons to plot.")
# -

len(spike_times)
