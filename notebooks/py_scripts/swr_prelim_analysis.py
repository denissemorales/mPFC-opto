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
import os
import datajoint as dj
import numpy as np
import pandas as pd

dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.position.v1 as sgp
import spyglass.lfp.analysis.v1 as lfp_analysis
from spyglass.lfp import LFPOutput
import spyglass.lfp as sglfp
from spyglass.position import PositionOutput
import spyglass.ripple.v1 as sgrip
import spyglass.ripple.v1 as sgr

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

sglfp.v1.LFPV1() & {"nwb_file_name": "Charlie20260223_.nwb"}

nwb_file_name = "Charlie20260224_.nwb"

lfp_electrode_group_name = "Left and Right CA1 Hippocampus"
lfp_eg_key = {
    "nwb_file_name": nwb_file_name,
    "lfp_electrode_group_name": lfp_electrode_group_name,
}

LFPOutput.LFPV1() << {"nwb_file_name": nwb_file_name}

sgrip.RippleTimesV1() & {"nwb_file_name": nwb_file_name}

ripple_times = (
    sgrip.RippleTimesV1()
    & {
        "lfp_merge_id": "39059d0b-24c0-1062-502c-0675ca0686c0",
        "ripple_param_name": "default_trodes",
    }
).fetch1_dataframe()
ripple_times

np.mean(ripple_times["duration"])

np.median(ripple_times["duration"])

np.mean(ripple_times["mean_zscore"])

np.median(ripple_times["mean_zscore"])

import pynwb
import spyglass.common as sgc

io = pynwb.NWBHDF5IO("/stelmo/nwb/raw/Charlie20260224.nwb", mode="r")
nwbf = io.read()

# +
epoch_start_time, epoch_stop_time, epoch_name = nwbf.intervals["epochs"][
    4
].to_numpy()[0]

laser_data = (
    nwbf.processing["behavior"]
    .data_interfaces["behavioral_events"]
    .time_series["Laser"]
    .data[:]
)
laser_timestamps = (
    nwbf.processing["behavior"]
    .data_interfaces["behavioral_events"]
    .time_series["Laser"]
    .timestamps[:]
)

# Restrict to epoch
mask = (laser_timestamps > epoch_start_time) & (
    laser_timestamps <= epoch_stop_time
)

laser_data_epoch = laser_data[mask]
laser_timestamps_epoch = laser_timestamps[mask]

# Ensure clean start
laser_data_epoch[0] = 0

# --- KEY STEP: only look at ON samples ---
laser_on_times_epoch = laser_timestamps_epoch[laser_data_epoch == 1]

# --- Detect first pulse of each train ---
first_laser_on_times = laser_on_times_epoch[
    np.insert(np.diff(laser_on_times_epoch) > 0.01, 0, True)
]
# -

df = pd.DataFrame({"timestamp": first_laser_on_times})
df

from spyglass.lfp.analysis.v1 import lfp_band

lfp_band.LFPBandV1() & {"nwb_file_name": "Charlie20260224_.nwb"}

# +
import matplotlib.pyplot as plt

lfp_band_key = {
    "nwb_file_name": "Charlie20260224_.nwb",
    "target_interval_list_name": "05_s3",
    "filter_name": "Ripple 150-250 Hz DMR",
}

ripple_band_df = (
    lfp_band.LFPBandV1()
    & {
        "nwb_file_name": "Charlie20260224_.nwb",
        "filter_name": "Ripple 150-250 Hz DMR",
        "target_interval_list_name": "05_s3",
    }
).fetch1_dataframe()

window = 0.1
i = 10  # [-10, 505, 601, 610, 200]
ripple_start = ripple_times.iloc[i].start_time
ripple_end = ripple_times.iloc[i].end_time
plt.plot(
    ripple_band_df.loc[ripple_start - window : ripple_end + window].index,
    ripple_band_df.loc[ripple_start - window : ripple_end + window].iloc[
        :, ::2
    ],
)
plt.axvline(ripple_start, color="r")
plt.axvline(ripple_end, color="r")

plt.xlabel("Time (s)")
plt.ylabel("Voltage (uV)")
# -

subset = ripple_band_df.loc[ripple_start - window : ripple_end + window]
print(subset.shape)

# +
import pandas as pd

# Assume df['timestamps'] holds your pulse times (sorted, in seconds)
timestamps = df["timestamp"].sort_values().reset_index(drop=True)

# Compute differences between consecutive pulses
dt = timestamps.diff()

# A "new burst" starts when the gap is bigger than ~3 ms
gap_threshold = 0.01  # seconds (3.5 ms to be safe)
burst_id = (dt > gap_threshold).cumsum()

# Group by burst and count how many pulses in each
burst_counts = timestamps.groupby(burst_id).count()

print(burst_counts)

# +
import matplotlib.pyplot as plt
import numpy as np

window = 0.100  # 100 ms
burst_size = 40
delays = []

for start in ripple_times["start_time"]:
    timestamps_in_window = df[
        (df["timestamp"] >= start) & (df["timestamp"] <= start + window)
    ]["timestamp"]
    if len(timestamps_in_window) > 0:
        laser_start = timestamps_in_window.values[0]
        delay = laser_start - start
        delays.append(delay)
delays = np.array(delays)

# +
import matplotlib.pyplot as plt
import numpy as np

window = 0.1
delays = []
for start in ripple_times["start_time"]:
    time_window = ripple_band_df.loc[start : start + window]

    timestamps_in_window = df[
        (df["timestamp"] >= start) & (df["timestamp"] <= start + window)
    ]["timestamp"]

    # Only keep every other timestamp
    timestamps_in_window = timestamps_in_window.iloc[::40]
    delays.append(timestamps_in_window - start)
# -

timestamps_in_window

# +
# --- Binary 0/1 trace ---
binary_trace = np.zeros(len(time_window.index))
for ts in timestamps_in_window:
    idx = time_window.index.get_indexer([ts], method="nearest")[0]
    if idx != -1:
        binary_trace[idx] = 1

# --- Create subplots ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
)

# Top subplot
ax1.step(time_window.index, binary_trace, where="post", color="blue")
ax1.set_ylabel("Laser\nOn/Off")
ax1.set_ylim(-0.1, 1.1)

# Bottom subplot
ax2.plot(time_window.index, time_window.iloc[:, 0], alpha=0.7)
ax2.axvline(ripple_start, color="r", linestyle="--", label="Ripple Start/End")
ax2.axvline(ripple_end, color="r", linestyle="--")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (µV)")
ax2.set_title("Example Ripple - Jacob 6/18 03_s2")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np

ripple_band_df = (lfp_band.LFPBandV1() & lfp_band_key).fetch1_dataframe()

window = 3
i = 150
ripple_start = ripple_times.iloc[i].start_time
ripple_end = ripple_times.iloc[i].end_time

# --- Extract ripple band window ---
time_window = ripple_band_df.loc[ripple_start - window : ripple_end + window]

# Find all ripples within the time window
window_start = ripple_start - window
window_end = ripple_end + window
ripples_in_window = ripple_times[
    (ripple_times["start_time"] >= window_start)
    & (ripple_times["end_time"] <= window_end)
]

# --- Extract timestamps in window ---
timestamps_in_window = df[
    (df["timestamp"] >= ripple_start - window)
    & (df["timestamp"] <= ripple_end + window)
]["timestamp"]

# Only keep every other timestamp
timestamps_in_window = timestamps_in_window.iloc[::2]


# --- Make binary 0/1 trace ---
binary_trace = np.zeros(len(time_window.index))
# For each stim timestamp, set nearest index in time_window to 1
for ts in timestamps_in_window:
    idx = time_window.index.get_indexer([ts], method="nearest")[0]
    if idx != -1:  # Only update if a valid index was found
        binary_trace[idx] = 1

# --- Create subplots ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
)

# Top subplot: binary opto stim trace
ax1.step(time_window.index, binary_trace, where="post", color="blue")
ax1.set_ylabel("Laser\nOn/Off")
ax1.set_ylim(-0.1, 1.1)

# Add all ripple markers to top subplot as well
for idx, ripple in ripples_in_window.iterrows():
    ax1.axvline(ripple["start_time"], color="r", linestyle="--", alpha=0.8)
    ax1.axvline(ripple["end_time"], color="g", linestyle="--", alpha=0.8)

# Bottom subplot: ripple band with markers
ax2.plot(time_window.index, time_window.iloc[:, ::15], alpha=0.7)

# Mark all ripple start and end times
for idx, ripple in ripples_in_window.iterrows():
    if idx == 0:  # Add label only to first ripple for legend
        ax2.axvline(
            ripple["start_time"],
            color="r",
            linestyle="--",
            alpha=0.8,
            label="Ripple Start",
        )
        ax2.axvline(
            ripple["end_time"],
            color="g",
            linestyle="--",
            alpha=0.8,
            label="Ripple End",
        )
    else:
        ax2.axvline(ripple["start_time"], color="r", linestyle="--", alpha=0.8)
        ax2.axvline(ripple["end_time"], color="g", linestyle="--", alpha=0.8)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (µV)")
ax2.set_title("Example Ripples Jacob 6/18 03_s2")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np

ripple_band_df = (lfp_band.LFPBandV1() & lfp_band_key).fetch1_dataframe()

window = 0.7
i = 2048
ripple_start = ripple_times.iloc[i].start_time
ripple_end = ripple_times.iloc[i].end_time

# --- Extract ripple band window ---
time_window = ripple_band_df.loc[ripple_start - window : ripple_end + window]

# --- Extract timestamps in window ---
timestamps_in_window = df[
    (df["timestamp"] >= ripple_start - window)
    & (df["timestamp"] <= ripple_end + window)
]["timestamp"]

# Only keep every other timestamp
timestamps_in_window = timestamps_in_window.iloc[::2]


# --- Make binary 0/1 trace ---
binary_trace = np.zeros(len(time_window.index))
# For each stim timestamp, set nearest index in time_window to 1
for ts in timestamps_in_window:
    idx = time_window.index.get_indexer([ts], method="nearest")[0]
    if idx != -1:  # Only update if a valid index was found
        binary_trace[idx] = 1

# --- Create subplots ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
)

# Top subplot: binary opto stim trace
ax1.step(time_window.index, binary_trace, where="post", color="blue")
ax1.set_ylabel("Laser\nOn/Off")
ax1.set_ylim(-0.1, 1.1)

# Bottom subplot: ripple band with markers
ax2.plot(time_window.index, time_window.iloc[:, ::15], alpha=0.7)

# # Mark ripple start and end
ax2.axvline(ripple_start, color="r", linestyle="--", label="Ripple Start/End")
ax2.axvline(ripple_end, color="r", linestyle="--")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (µV)")
ax2.set_title("Example True Neg")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np

# --- Load ripple band ---
ripple_band_df = (lfp_band.LFPBandV1() & lfp_band_key).fetch1_dataframe()

# --- Parameters ---
window = 0.7
i = 200

ripple_start = ripple_times.iloc[i].start_time
ripple_end = ripple_times.iloc[i].end_time

# --- Extract ripple window ---
time_window = ripple_band_df.loc[ripple_start - window : ripple_end + window]

# --- Get stim timestamps in same window ---
timestamps_in_window = df[
    (df["timestamp"] >= ripple_start - window)
    & (df["timestamp"] <= ripple_end + window)
]["timestamp"].values

# --- Build binary stim trace ---
binary_trace = np.zeros(len(time_window.index))

for ts in timestamps_in_window:
    # find closest index in LFP time axis
    idx = np.argmin(np.abs(time_window.index.values - ts))
    binary_trace[idx] = 1

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
)

# --- Top: laser trace ---
ax1.step(time_window.index, binary_trace, where="post", color="blue")
ax1.set_ylabel("Laser\nOn/Off")
ax1.set_ylim(-0.1, 1.1)

# --- Bottom: ripple band ---
ax2.plot(time_window.index, time_window.iloc[:, ::15], alpha=0.7)

# --- Ripple boundaries ---
ax2.axvline(ripple_start, color="r", linestyle="--", label="Ripple Start")
ax2.axvline(ripple_end, color="r", linestyle="--", label="Ripple End")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (µV)")
ax2.set_title("True Positive")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# +
import pandas as pd

# Assume your timestamps are in df['timestamps'] (in seconds)
timestamps = df["timestamp"].sort_values().reset_index(drop=True)

# Compute the difference between consecutive timestamps
dt = timestamps.diff()

# Consider a gap bigger than, e.g., 0.0025 s (2.5 ms) as a separation between bursts
gap_threshold = 0.004

# First timestamp of each burst: either the first one, or after a gap
first_of_bursts = timestamps[(dt.isna()) | (dt > gap_threshold)]

print(first_of_bursts)
# -

ripple_times["mean_zscore"]

# +
ripple_times["mean_zscore"]

# Compute the mean duration
mean_amplitude = ripple_times["mean_zscore"].mean()
mean_amplitude
# -

median = np.median(ripple_times["mean_zscore"])
# Plot histogram
plt.figure(figsize=(8, 4))
plt.hist(ripple_times["mean_zscore"], bins=40, edgecolor="black", density=True)
plt.xlabel("Ripple Mean Z-Score (s)")
plt.axvline(median, color="red", linestyle="dashed", linewidth=1)
plt.ylabel("Density (%)")
plt.title("Histogram of Ripple Amplitude - Charlie 2/23 03_s2")
plt.show()

ripple_times["duration"] = ripple_times["end_time"] - ripple_times["start_time"]
median = np.median(ripple_times["duration"])
# Plot histogram
plt.figure(figsize=(8, 4))
plt.hist(ripple_times["duration"], bins=40, edgecolor="black", density=True)
plt.xlabel("Ripple Duration (s)")
plt.axvline(median, color="red", linestyle="dashed", linewidth=1)
plt.ylabel("Density (%)")
plt.title("Histogram of Ripple Durations -  Charlie 2/23 03_s2")
plt.show()

# +
window = 0.25  # 250 ms before ripple

# --- Convert timestamps to numpy arrays ---
pulse_times = df["timestamp"].values
ripple_starts = ripple_times["start_time"].values
ripple_ends = ripple_times["end_time"].values

# --- Compute ripple duration ---
ripple_times["duration"] = ripple_times["end_time"] - ripple_times["start_time"]

# --- Mark ripples with pulse BEFORE ---
pulse_before = (
    (pulse_times[:, None] >= (ripple_starts - window))
    & (pulse_times[:, None] < ripple_starts)
).any(axis=0)

# --- Add to dataframe ---
ripple_times["pulse_before"] = pulse_before

# --- Split durations ---
amp_with_pulse_before = ripple_times.loc[
    ripple_times["pulse_before"], "mean_zscore"
]
amp_without_pulse_before = ripple_times.loc[
    ~ripple_times["pulse_before"], "mean_zscore"
]

# --- Print statistics ---
print(f"N ripples with pulse before: {len(amp_with_pulse_before)}")
print(f"N ripples without pulse before: {len(amp_without_pulse_before)}")
print(f"Mean duration with pulse before: {amp_with_pulse_before.mean():.3f} s")
print(
    f"Mean duration without pulse before: {amp_without_pulse_before.mean():.3f} s"
)

# --- Statistical test ---
# stat, p_value = mannwhitneyu(amp_with_pulse_before, amp_without_pulse_before, alternative='two-sided')
# REVIEW: the mannwhitneyu call that produces stat and p_value is
# commented out directly above, so this print raises NameError.
print(f"Mann-Whitney U test: stat={stat}, p={p_value:.4f}")  # noqa: F821

# --- Plot distributions ---
plt.hist(
    amp_with_pulse_before,
    bins=30,
    alpha=0.5,
    label="Stim at least 250ms before",
    density=True,
)
plt.hist(
    amp_without_pulse_before,
    bins=30,
    alpha=0.5,
    label="No stim before",
    density=True,
)
plt.xlabel("Ripple Mean Z-Score")
plt.ylabel("Density (%)")
plt.legend()
plt.title("Jacob 06/17 03_s2: Ripple Amplitude: Stim v No Stim")
plt.show()
# -

# REVIEW: `dur_without_stim` is never defined in this notebook.
no_stim_ids = dur_without_stim.reset_index()["id"]  # noqa: F821
plt.hist(no_stim_ids, density=True)
plt.title("Distribution of No Stim SWR numbers")
plt.ylabel("Density %")
plt.xlabel("SWR Number")

# REVIEW: `dur_with_stim` is never defined in this notebook.
ids = dur_with_stim.reset_index()["id"]  # noqa: F821
plt.hist(ids)
plt.title("Distribution of Stim SWR numbers")
plt.ylabel("Density %")
plt.xlabel("SWR Number")

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- Parameters ---
# Assume stim timestamps are in df['stim_timestamp']
stim_times = df[
    "timestamp"
].values  # adjust if your stim column is named differently

# --- Convert ripple times ---
ripple_starts = ripple_times["start_time"].values
ripple_ends = ripple_times["end_time"].values

# --- Compute ripple duration ---
ripple_times["duration"] = ripple_times["end_time"] - ripple_times["start_time"]

# --- Mark ripples with stim DURING ripple ---
stim_during = (
    (stim_times[:, None] >= ripple_starts)
    & (stim_times[:, None] <= ripple_ends)
).any(axis=0)

ripple_times["stim_during"] = stim_during

# --- Split durations ---
dur_with_stim = ripple_times.loc[ripple_times["stim_during"], "duration"]
dur_without_stim = ripple_times.loc[~ripple_times["stim_during"], "duration"]

# --- Print statistics ---
print(f"N ripples with stim during: {len(dur_with_stim)}")
print(f"N ripples without stim during: {len(dur_without_stim)}")
print(f"Mean duration with stim during: {dur_with_stim.mean():.3f} s")
print(f"Mean duration without stim during: {dur_without_stim.mean():.3f} s")

# --- Statistical test ---
stat, p_value = mannwhitneyu(
    dur_with_stim, dur_without_stim, alternative="two-sided"
)
print(f"Mann-Whitney U test: stat={stat}, p={p_value:.4f}")

# --- Plot distributions ---
plt.hist(
    dur_with_stim, bins=30, alpha=0.5, label="Stim during SWR", density=True
)
plt.hist(
    dur_without_stim, bins=30, alpha=0.5, label="No stim during", density=True
)
plt.xlabel("Ripple duration (s)")
plt.ylabel("Density (%)")
plt.legend()
plt.title("Jacob 06/18 03_s2: Ripple Duration: Stim v No Stim")
plt.show()
# -

ripple_times

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- Parameters ---
stim_times = df["timestamp"].values

# --- Convert ripple times ---
ripple_starts = ripple_times["start_time"].values
ripple_ends = ripple_times["end_time"].values

# --- Add 10 ms window ---
post_window = 0.010  # 10 ms in seconds

# --- Compute ripple duration ---
ripple_times["duration"] = ripple_times["end_time"] - ripple_times["start_time"]

# --- Mark ripples with stim DURING or within 10 ms AFTER ---
stim_during_or_after = (
    (stim_times[:, None] >= ripple_starts)
    & (stim_times[:, None] <= (ripple_ends + post_window))
).any(axis=0)

ripple_times["stim_during_or_10ms_after"] = stim_during_or_after

# --- Split durations ---
dur_with_stim = ripple_times.loc[
    ripple_times["stim_during_or_10ms_after"], "duration"
]
dur_without_stim = ripple_times.loc[
    ~ripple_times["stim_during_or_10ms_after"], "duration"
]

# --- Print statistics ---
print(f"N ripples with stim (during or ≤10ms after): {len(dur_with_stim)}")
print(f"N ripples without stim: {len(dur_without_stim)}")
print(f"Mean duration with stim: {dur_with_stim.mean():.3f} s")
print(f"Mean duration without stim: {dur_without_stim.mean():.3f} s")

# --- Statistical test ---
stat, p_value = mannwhitneyu(
    dur_with_stim, dur_without_stim, alternative="two-sided"
)
print(f"Mann-Whitney U test: stat={stat}, p={p_value:.4f}")

# --- Plot distributions ---
plt.hist(
    dur_with_stim,
    bins=30,
    alpha=0.5,
    label="Stim during or ≤10ms after",
    density=True,
)
plt.hist(dur_without_stim, bins=30, alpha=0.5, label="No stim", density=True)
plt.xlabel("Ripple duration (s)")
plt.ylabel("Density")
plt.legend()
plt.title("Ripple Duration: Stim During / +10ms vs None")
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- Parameters ---
stim_times = df["timestamp"].values

# --- Convert ripple times ---
ripple_starts = ripple_times["start_time"].values
ripple_ends = ripple_times["end_time"].values

# --- Add 10 ms window ---
post_window = 0.010  # 10 ms in seconds

# --- Compute ripple duration ---
ripple_times["duration"] = ripple_times["end_time"] - ripple_times["start_time"]

# --- Mark ripples with stim DURING or within 10 ms AFTER ---
stim_during_or_after = (
    (stim_times[:, None] >= ripple_starts)
    & (stim_times[:, None] <= (ripple_ends + post_window))
).any(axis=0)

ripple_times["stim_during_or_10ms_after"] = stim_during_or_after

# --- Split durations ---
dur_with_stim = ripple_times.loc[
    ripple_times["stim_during_or_10ms_after"], "mean_zscore"
]
dur_without_stim = ripple_times.loc[
    ~ripple_times["stim_during_or_10ms_after"], "mean_zscore"
]

# --- Print statistics ---
print(f"N ripples with stim (during or ≤10ms after): {len(dur_with_stim)}")
print(f"N ripples without stim: {len(dur_without_stim)}")
print(f"Mean duration with stim: {dur_with_stim.mean():.3f} s")
print(f"Mean duration without stim: {dur_without_stim.mean():.3f} s")

# --- Statistical test ---
stat, p_value = mannwhitneyu(
    dur_with_stim, dur_without_stim, alternative="two-sided"
)
print(f"Mann-Whitney U test: stat={stat}, p={p_value:.4f}")

# --- Plot distributions ---
plt.hist(
    dur_with_stim,
    bins=30,
    alpha=0.5,
    label="Stim during or ≤10ms after",
    density=True,
)
plt.hist(dur_without_stim, bins=30, alpha=0.5, label="No stim", density=True)
plt.xlabel("Mean ZScore")
plt.ylabel("Density")
plt.legend()
plt.title("Ripple Amplitude: Stim During / +10ms vs None")
plt.show()
# -

len(stim_times)

# +
# --- Check if each stim falls inside ANY ripple ---
stim_in_ripple = (
    (stim_times[:, None] >= ripple_starts - 0.010)
    & (stim_times[:, None] <= ripple_ends)
).any(axis=1)

# --- Count ---
n_stim_with_ripple = np.sum(stim_in_ripple)
n_stim_without_ripple = len(stim_times) - n_stim_with_ripple

print(f"N stims WITH ripple: {n_stim_with_ripple}")
print(f"N stims WITHOUT ripple: {n_stim_without_ripple}")

# +
# ripple_power = (lfp_band.LFPBandV1() & {"lfp_merge_id": lfp_merge_id, "filter_name": filter_name}).compute_signal_power(
#     electrode_list=electrode_list
# )

# +
import numpy as np

# --- Parameters ---
post_window = 0.010  # 10 ms

# --- Arrays ---
stim_times = df["timestamp"].values
ripple_starts = ripple_times["start_time"].values
ripple_ends = ripple_times["end_time"].values

# --- Ripples WITH stim (during or +10ms) ---
ripple_has_stim = (
    (stim_times[:, None] >= ripple_starts)
    & (stim_times[:, None] <= ripple_ends + post_window)
).any(axis=0)

# --- Stims WITH ripple ---
stim_in_ripple = (
    (stim_times[:, None] >= ripple_starts)
    & (stim_times[:, None] <= ripple_ends + post_window)
).any(axis=1)

# --- Counts ---
TP = np.sum(ripple_has_stim)  # ripples with laser
FN = len(ripple_has_stim) - TP  # ripples without laser
FP = len(stim_times) - np.sum(stim_in_ripple)  # stims without ripple

print(f"TP (ripples with laser): {TP}")
print(f"FN (ripples without laser): {FN}")
print(f"FP (stims without ripple): {FP}")

# +
# --- Totals ---
n_ripples = len(ripple_has_stim)
n_stims = len(stim_times)

# --- Percentages ---
pct_ripples_with_stim = TP / n_ripples * 100
pct_ripples_without_stim = FN / n_ripples * 100

pct_stims_with_ripple = np.sum(stim_in_ripple) / n_stims * 100
pct_stims_without_ripple = FP / n_stims * 100

# --- Print nicely ---
print("\n--- Ripple-centered ---")
print(f"Ripples WITH laser: {TP} ({pct_ripples_with_stim:.2f}%)")
print(f"Ripples WITHOUT laser: {FN} ({pct_ripples_without_stim:.2f}%)")

print("\n--- Stim-centered ---")
print(
    f"Stims WITH ripple: {np.sum(stim_in_ripple)} ({pct_stims_with_ripple:.2f}%)"
)
print(f"Stims WITHOUT ripple: {FP} ({pct_stims_without_ripple:.2f}%)")

# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Custom matrix (no TN)
cm = np.array([[TP, FN], [FP, 0]])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Laser ON", "Laser OFF"],
    yticklabels=["SWR", "No SWR"],
)

plt.title("Signal Detection - 2/23 11_s6 2SD")
plt.xlabel("Category")
plt.ylabel("Event Type")
plt.show()

# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Totals ---
n_ripples = TP + FN
n_stims = FP + np.sum(stim_in_ripple)

# --- Convert to percentages (row-wise) ---
cm_pct = np.array(
    [[TP / n_ripples * 100, FN / n_ripples * 100], [FP / n_stims * 100, 0]]
)

# --- Labels with percentages + counts ---
labels = np.array(
    [
        [f"{TP}\n({cm_pct[0,0]:.1f}%)", f"{FN}\n({cm_pct[0,1]:.1f}%)"],
        [f"{FP}\n({cm_pct[1,0]:.1f}%)", "0\n(0%)"],
    ]
)

# --- Plot ---
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_pct,
    annot=labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Laser ON", "Laser OFF"],
    yticklabels=["SWR", "No SWR"],
)

plt.title("Signal Detection - 2/23 11_s6 2SD")
plt.xlabel("Category")
plt.ylabel("Event Type")
plt.show()

# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Totals ---
n_ripples = TP + FN
n_stims = FP + np.sum(stim_in_ripple)

# --- Percentages (rounded) ---
cm_pct = np.array(
    [
        [round(TP / n_ripples * 100), round(FN / n_ripples * 100)],
        [round(FP / n_stims * 100), 0],
    ]
)

# --- Labels (leave bottom-right blank) ---
labels = np.array(
    [
        [f"{TP}\n({cm_pct[0,0]}%)", f"{FN}\n({cm_pct[0,1]}%)"],
        [f"{FP}\n({cm_pct[1,0]}%)", ""],
    ]
)

# --- Plot ---
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_pct,
    annot=labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Laser ON", "Laser OFF"],
    yticklabels=["SWR", "No SWR"],
    cbar=False,
)  # optional: cleaner look

plt.title("Signal Detection - 2/23 05_s3 2SD")
plt.xlabel("Online SWR Detection")
plt.ylabel("Offline SWR Detection")
plt.show()
