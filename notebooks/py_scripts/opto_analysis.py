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

# First, we load our datajoint config
#

# +
from pathlib import Path
import datajoint as dj

dj.config.load(
    Path("dj_local_conf_2.json").absolute()
)  # replace with your config file name
# -

# I have put all of our spike sorted data into a spike sorting group

# +
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

SortedSpikesGroup()

# +
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

nwb_file_name = "Caius20260618_.nwb"

group_key = {
    "nwb_file_name": nwb_file_name,
    "sorted_spikes_group_name": "mPFC_opto_testing",
}
SortedSpikesGroup & group_key
# -

# here we can see that I put 3 sort groups (3 shanks in the mPFC) into this one sorted spikes group
SortedSpikesGroup.Units & group_key

# We can access the spikesorting results for this data using `SortedSpikesGroup.fetch_spike_data()`
#

# get the complete key
group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
# get the spike data, returns a list of unit spike times
SortedSpikesGroup().fetch_spike_data(group_key)

# Now we can get the cell spiking information

spike_times, unit_ids = SortedSpikesGroup().fetch_spike_data(
    group_key, return_unit_ids=True
)

# +
len(unit_ids)  # this is how many cells there are

# this is a lot probably because there are a lot of artifacts that looked like cells

# so if the results look weird, we might have to do another round of curation
# -

unit_ids  # so each cell has a number and a spikesorting_merge_id so we can reference it in the database

# Now we want laser times

# +
import numpy as np
import pynwb
import spyglass.common as sgc

# this reads the raw nwb file so we can get laser DIO times
io = pynwb.NWBHDF5IO("/stelmo/nwb/raw/Caius20260618.nwb", mode="r")
nwbf = io.read()

# +
epoch_number = 2  # this goes from 0-9 and we can reference the google sheet to figure what laser power it is

epoch_start_time, epoch_stop_time, epoch_name = nwbf.intervals["epochs"][
    epoch_number
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

# +
import matplotlib.pyplot as plt

# Parameters
# pre_time = 0.2
# post_time = 0.4
# bin_size = 0.005

# laser_on_times = first_laser_on_times

# # PSTH bins
# bins = np.arange(-pre_time, post_time + bin_size, bin_size)
# bin_centers = bins[:-1] + bin_size / 2

# for unit_index, spikes in enumerate(spike_times):

#     if len(spikes) == 0:
#         continue

#     aligned_spikes_all = []

#     for onset in laser_on_times:

#         start_t = onset - pre_time
#         stop_t  = onset + post_time

#         # ✅ DO NOT skip trials — just collect spikes if present
#         spikes_rel = spikes[(spikes >= start_t) & (spikes <= stop_t)] - onset

#         # Always append (even if empty)
#         aligned_spikes_all.append(spikes_rel)

#     if len(aligned_spikes_all) == 0:
#         continue

#     # Flatten spikes for PSTH
#     all_spikes_flat = np.concatenate(aligned_spikes_all) if any(len(s) > 0 for s in aligned_spikes_all) else np.array([])

#     counts, _ = np.histogram(all_spikes_flat, bins=bins)

#     # ✅ Normalize by TOTAL number of trials (not just non-empty ones)
#     n_trials = len(aligned_spikes_all)
#     psth = counts / (n_trials * bin_size)

#     # ----- Plot -----
#     fig, (ax_raster, ax_psth) = plt.subplots(
#         2, 1, figsize=(8, 6), sharex=True,
#         gridspec_kw={'height_ratios': [2, 1]}
#     )

#     # Raster (includes empty trials correctly)
#     for i, spikes_rel in enumerate(aligned_spikes_all):
#         if len(spikes_rel) > 0:
#             ax_raster.vlines(spikes_rel, i + 0.5, i + 1.5, color='black', linewidth=0.8)

#     ax_raster.axvspan(0, 0.1, color='red', alpha=0.2)
#     ax_raster.set_ylabel("Trial")
#     ax_raster.set_title(f"Unit {unit_ids[unit_index]['unit_id']}")
#     ax_raster.set_facecolor('white')
#     ax_raster.set_ylim(0.5, n_trials + 0.5)

#     # PSTH
#     ax_psth.bar(bin_centers, psth, width=bin_size, color='black')
#     ax_psth.axvspan(0, 0.1, color='red', alpha=0.2)

#     ax_psth.set_xlabel("Time from laser onset (s)")
#     ax_psth.set_ylabel("Firing rate (Hz)")
#     ax_psth.set_xlim([-pre_time, post_time])
#     ax_psth.set_facecolor('white')

#     plt.tight_layout()
#     plt.show()

# +
pre_time = 0.2
post_time = 0.20
laser_on_times = first_laser_on_times

modulation_scores = []

for u, spikes in enumerate(spike_times):
    spikes_before_total = 0
    spikes_after_total = 0
    valid_trials = 0

    for onset in laser_on_times:
        start_t = onset - pre_time
        stop_t = onset + post_time

        if spikes[0] > stop_t or spikes[-1] < start_t:
            continue

        # count spikes before and after
        spikes_before = np.sum((spikes >= onset - pre_time) & (spikes < onset))
        spikes_after = np.sum((spikes >= onset) & (spikes < onset + post_time))

        spikes_before_total += spikes_before
        spikes_after_total += spikes_after
        valid_trials += 1

    if valid_trials == 0:
        modulation_scores.append(np.nan)
        continue

    # average per trial
    spikes_before_avg = spikes_before_total / valid_trials
    spikes_after_avg = spikes_after_total / valid_trials

    # compute modulation index
    if (spikes_before_avg + spikes_after_avg) == 0:
        score = np.nan
    else:
        score = (spikes_before_avg - spikes_after_avg) / (
            spikes_before_avg + spikes_after_avg
        )

    modulation_scores.append(score)

modulation_scores = np.array(modulation_scores)


# Remove NaNs
valid_idx = ~np.isnan(modulation_scores)
scores_valid = modulation_scores[valid_idx]

unit_ids_valid = [unit_ids[i] for i in np.where(valid_idx)[0]]

fig, ax = plt.subplots(figsize=(10, 5))
unit_labels = [u["unit_id"] for u in unit_ids_valid]
ax.bar(unit_labels, scores_valid, color="gray", edgecolor="black")

ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Unit ID")
ax.set_ylabel("Modulation Index (before - after) / (before + after)")
ax.set_title("Laser Modulation per Unit")

ax.set_ylim([-1.1, 1.1])
plt.tight_layout()
plt.show()

# +
import seaborn as sns

plt.figure(figsize=(6, 5))

sns.histplot(scores_valid, bins=20, kde=True, color="gray")

plt.axvline(0.2, color="red", linestyle="--")

plt.xlabel("Modulation Index")
plt.ylabel("Count")
plt.title("Distribution of Laser Modulation Scores")

plt.tight_layout()
plt.show()

# +
# Convert unit_ids list of dicts to simple list of numbers
unit_id_numbers = np.array([u["unit_id"] for u in unit_ids])

# Define threshold
threshold = 0.2  # 20%

# Units strongly modulated
suppressed_units = unit_id_numbers[modulation_scores > threshold]
activated_units = unit_id_numbers[modulation_scores < -threshold]

print(f"Suppressed units (> +20% decrease): {suppressed_units}")
print(f"Activated units (> +20% increase): {activated_units}")
# -

len(suppressed_units)

len(activated_units)
