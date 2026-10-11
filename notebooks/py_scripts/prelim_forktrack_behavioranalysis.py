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
import pynwb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spyglass.position import PositionOutput
import spyglass.common as sgc
import pandas as pd

# +
# from wtrack table


def get_first_pokes_after_well_change(poke_df):
    """
    Given a DataFrame of poke events, return only the first poke (value == 1)
    at each well that follows a poke at a different well.

    Expected columns: ['time', 'well_name', 'value']
    """
    # Keep only "poke ON" events
    poke_on_df = poke_df[poke_df["value"] == 1].copy()

    # Sort by time
    poke_on_df = poke_on_df.sort_values("time").reset_index(drop=True)

    # Identify transitions between wells
    poke_on_df["prev_well"] = poke_on_df["well_name"].shift(1)
    transitions = poke_on_df["well_name"] != poke_on_df["prev_well"]

    # Keep only first pokes following a different well
    first_pokes = poke_on_df[transitions].copy().reset_index(drop=True)

    return first_pokes


def check_pump_after_pokes(first_pokes_df, pump_times_dict, max_delay=None):
    """
    Check whether the pump for the poked well turned on after each first poke.

    Parameters
    ----------
    first_pokes_df : pd.DataFrame
        DataFrame with at least columns ['time', 'well_name'] where well_name matches
        keys in pump_times_dict after replacing '_poke' with '' or similar.
    pump_times_dict : dict
        Mapping like {'Left_pump': np.array([...]), 'Center_pump': np.array([...]), ...}
        Each array should be sorted ascending and contain pump ON timestamps.
    max_delay : float or None
        If provided, require the pump ON to occur within max_delay seconds after the poke.
        If None, any pump after the poke counts.

    Returns
    -------
    pd.DataFrame
        Copy of first_pokes_df with added columns:
            'pump_triggered' (bool),
            'pump_time' (float or np.nan),
            'pump_delay' (float or np.nan)
    """
    df = first_pokes_df.copy().reset_index(drop=True)
    pump_triggered = []
    pump_times_found = []
    pump_delays = []

    # Ensure pump arrays are numpy and sorted
    pump_times_dict = {
        k: np.sort(np.asarray(v)) for k, v in pump_times_dict.items()
    }

    for t, well in zip(df["time"], df["well_name"]):
        # derive pump key from well name (common pattern Left_poke -> Left_pump)
        pump_key = well.replace("poke", "pump").replace("Poke", "Pump")
        pump_arr = pump_times_dict.get(pump_key, np.array([]))

        if pump_arr.size == 0:
            pump_triggered.append(False)
            pump_times_found.append(np.nan)
            pump_delays.append(np.nan)
            continue

        # find first pump_time >= poke time
        idx = np.searchsorted(pump_arr, t, side="left")
        if idx < len(pump_arr):
            pump_time = float(pump_arr[idx])
            delay = pump_time - t
            ok = delay >= 0 and (max_delay is None or delay <= max_delay)
            pump_triggered.append(bool(ok))
            pump_times_found.append(pump_time if ok else np.nan)
            pump_delays.append(delay if ok else np.nan)
        else:
            pump_triggered.append(False)
            pump_times_found.append(np.nan)
            pump_delays.append(np.nan)

    df["pump_triggered"] = pump_triggered
    df["pump_time"] = pump_times_found
    df["pump_delay"] = pump_delays
    return df


# -

nwb_file_name = "Caius20260623_.nwb"  # "Embry20260604_.nwb"
nwbf = pynwb.NWBHDF5IO(sgc.Nwbfile().get_abs_path(nwb_file_name), "r").read()

# REVIEW: `nwb` is never defined in this notebook, so this cell raises
# NameError. The NWB file object is presumably loaded under another name.
nwbf_dios = nwb.fields["processing"]["behavior"][  # noqa: F821
    "behavioral_events"
].fields["time_series"]

# +
import pandas as pd

# Define the epochs you're interested in
epochs_of_interest = [1, 3, 5, 7, 9]

# Initialize an empty list to store the data from selected epochs
epoch_data_list = []

# Loop through the specified epochs
for epoch_index in epochs_of_interest:
    # Extract the epoch data
    epoch_start_time, epoch_stop_time, epoch_name = nwbf.intervals["epochs"][
        epoch_index
    ].to_numpy()[0]

    # Append the data to the list
    epoch_data_list.append(
        {
            "epoch_start_time": epoch_start_time,
            "epoch_stop_time": epoch_stop_time,
            "epoch_name": epoch_name,
        }
    )

# Convert the list into a DataFrame
epochs_df = pd.DataFrame(epoch_data_list)

# +
# Define the mapping for DIO events
name_mapping = {
    "LeftWell_Poke": "Left_poke",
    "CenterWell_Poke": "Center_poke",
    "RightWell_Poke": "Right_poke",
    "LeftMilk_Pump": "Left_pump",
    "CenterMilk_Pump": "Center_pump",
    "RightMilk_Pump": "Right_pump",
    "HandleMilk_Pump": "Handle_pump",
    "HandleWell_Poke": "Handle_poke",
}

# Dictionary to store DIO events for each epoch
all_dio_events = {}

# Loop through the specified epochs
for epoch_index in epochs_of_interest:
    # Extract the start and stop times for the epoch
    epoch_start_time, epoch_stop_time, epoch_name = nwbf.intervals["epochs"][
        epoch_index
    ].to_numpy()[0]

    # Make sure epoch_name is a scalar (not a numpy.ndarray)
    epoch_name = (
        epoch_name.item() if isinstance(epoch_name, np.ndarray) else epoch_name
    )

    # Initialize a dictionary to store DIO events for this epoch
    dio_events = {}

    # Access the DIO data from the NWB file
    nwbf_dios = (
        nwbf.processing["behavior"]
        .data_interfaces["behavioral_events"]
        .time_series
    )

    # Loop through each DIO event in the name_mapping
    for i, (nwb_name, mapped_name) in enumerate(name_mapping.items()):
        if nwb_name in nwbf_dios:
            # Get timestamps and data for the current DIO event
            ts = np.asarray(nwbf_dios[nwb_name].timestamps[:])
            data = np.asarray(nwbf_dios[nwb_name].data[:])

            # Filter the DIO events based on the epoch's start and stop times
            mask = (ts > epoch_start_time) & (ts <= epoch_stop_time)

            # Store the filtered events for this DIO
            dio_events[i] = {
                "name": mapped_name,
                "times": ts[mask],
                "values": data[mask],
            }

    # Store the DIO events for this epoch in the all_dio_events dictionary
    all_dio_events[epoch_name] = dio_events

# +
nwbf_dios = (
    nwbf.processing["behavior"].data_interfaces["behavioral_events"].time_series
)

name_mapping = {
    "LeftWell_Poke": "Left_poke",
    "CenterWell_Poke": "Center_poke",
    "RightWell_Poke": "Right_poke",
    "LeftMilk_Pump": "Left_pump",
    "CenterMilk_Pump": "Center_pump",
    "RightMilk_Pump": "Right_pump",
    "HandleMilk_Pump": "Handle_pump",
    "HandleWell_Poke": "Handle_poke",
}

dio_events = {}
for i, (nwb_name, mapped_name) in enumerate(name_mapping.items()):
    if nwb_name in nwbf_dios:
        ts = np.asarray(nwbf_dios[nwb_name].timestamps[:])
        data = np.asarray(nwbf_dios[nwb_name].data[:])
        mask = (ts > epoch_start_time) & (ts <= epoch_stop_time)

        dio_events[i] = {
            "name": mapped_name,
            "times": ts[mask],
            "values": data[mask],
        }

# +
import pandas as pd
import numpy as np

rows = []

# Iterate over each epoch in all_dio_events
for epoch_key, epoch_data in all_dio_events.items():
    for well_id, well_data in epoch_data.items():
        well_name = well_data["name"]
        times = well_data["times"]
        values = well_data["values"]

        # Compute the distance for each timepoint (if have this)
        for time, value in zip(times, values):
            animal_x = np.nan
            animal_y = np.nan
            distance_to_well = np.nan
            well_x = np.nan
            well_y = np.nan

            # Store the data for this well at this time
            rows.append(
                {
                    "time": time,
                    "epoch": epoch_key,
                    "well_name": well_name,
                    "value": value,
                    "animal_x": animal_x,
                    "animal_y": animal_y,
                    "distance_to_well": distance_to_well,
                    "well_x": well_x,
                    "well_y": well_y,
                }
            )

# Convert the rows list to a DataFrame
df = pd.DataFrame(rows)
df
# -

df["well_name"].unique()

pump_df = df[
    (df["well_name"] == "Left_pump")
    | (df["well_name"] == "Right_pump")
    | (df["well_name"] == "Center_pump")
    | (df["well_name"] == "Handle_pump")
]

df = df[
    (df["well_name"] == "Left_poke")
    | (df["well_name"] == "Right_poke")
    | (df["well_name"] == "Center_poke")
    | (df["well_name"] == "Handle_pump")
]

valid_pokes = df

# +
# valid_pokes = results["valid_pokes"]

first_pokes = get_first_pokes_after_well_change(valid_pokes)

print(first_pokes[["time", "well_name"]])
# -

first_pokes

all_dio_events

# +
# Filter name mapping to pumps only
pump_name_mapping = {k: v for k, v in name_mapping.items() if "Pump" in k}

pump_events = {}
for i, (nwb_name, mapped_name) in enumerate(pump_name_mapping.items()):
    if nwb_name in nwbf_dios:
        ts = np.asarray(nwbf_dios[nwb_name].timestamps[:])
        data = np.asarray(nwbf_dios[nwb_name].data[:])
        mask = (ts > epoch_start_time) & (ts <= epoch_stop_time)

        pump_events[i] = {
            "name": mapped_name,
            "times": ts[mask],
            "values": data[mask],
        }

pump_times_dict = {v["name"]: v["times"] for v in pump_events.values()}
pump_times_dict = all

# +
import numpy as np

# Create flat pump_times_dict combining all sessions
pump_times_dict = {}

for session_id, session_data in all_dio_events.items():
    for channel_id, channel_data in session_data.items():
        event_name = channel_data["name"]

        # Check if this is a pump event
        if "pump" in event_name.lower():
            times = channel_data["times"]
            values = channel_data["values"]

            # Only keep times where value is 1 (pump activation)
            activation_times = times[values == 1]

            # Initialize list if this pump type hasn't been seen yet
            if event_name not in pump_times_dict:
                pump_times_dict[event_name] = []

            # Add activation times to the list
            pump_times_dict[event_name].extend(activation_times)

# Convert lists to sorted numpy arrays
for pump_name in pump_times_dict:
    pump_times_dict[pump_name] = np.sort(np.array(pump_times_dict[pump_name]))
# -

pump_times_dict


# +
def flag_close_pokes(pokes_df, min_interval=10.0):
    df = pokes_df.sort_values("time").reset_index(drop=True)
    df["time_diff"] = df["time"].diff()
    df["too_close"] = df["time_diff"] < min_interval
    return df


pump_results_df = check_pump_after_pokes(
    first_pokes,
    pump_times_dict,
    max_delay=0.5,  # optional, only count pumps within 0.5s of poke
)

# Initialize the 'time_between_rewards' column
pump_results_df["time_between_rewards"] = None

# Find indices of rows where a pump event triggered
pump_indices = pump_results_df[pump_results_df["pump_triggered"]].index

# Ensure there's at least one pump event to process
if len(pump_indices) > 0:
    # Set the time difference for the first reward (use epoch start time)
    epoch_start_time = pump_results_df["time"].iloc[
        0
    ]  # Assuming the first row contains the epoch start time
    pump_results_df.loc[pump_indices[0], "time_between_rewards"] = (
        pump_results_df.loc[pump_indices[0], "pump_time"] - epoch_start_time
    )

# Calculate the time difference between consecutive pump events
for i in range(1, len(pump_indices)):
    prev_idx = pump_indices[i - 1]
    curr_idx = pump_indices[i]

    # Calculate the time difference between consecutive pump events
    time_diff = (
        pump_results_df.loc[curr_idx, "pump_time"]
        - pump_results_df.loc[prev_idx, "pump_time"]
    )

    # Set the time difference for the current event
    pump_results_df.loc[curr_idx, "time_between_rewards"] = time_diff


# -

def calculate_time_between_rewards(pump_results_df, epoch_start_time):
    """
    Calculate time between rewards, using epoch start time for the first reward.

    Parameters:
    -----------
    pump_results_df : pd.DataFrame
        DataFrame with pump results including 'pump_triggered' and 'pump_time' columns
    epoch_start_time : float
        The start time of the current epoch

    Returns:
    --------
    pd.DataFrame with 'time_between_rewards' column added
    """
    # Initialize the column
    pump_results_df["time_between_rewards"] = None

    # Find indices of rows where a pump event triggered
    pump_indices = pump_results_df[pump_results_df["pump_triggered"]].index

    # Ensure there's at least one pump event to process
    if len(pump_indices) > 0:
        # Use the epoch start time for the first reward
        pump_results_df.loc[pump_indices[0], "time_between_rewards"] = (
            pump_results_df.loc[pump_indices[0], "pump_time"] - epoch_start_time
        )

        # Calculate the time difference between consecutive pump events
        for i in range(1, len(pump_indices)):
            prev_idx = pump_indices[i - 1]
            curr_idx = pump_indices[i]

            time_diff = (
                pump_results_df.loc[curr_idx, "pump_time"]
                - pump_results_df.loc[prev_idx, "pump_time"]
            )
            pump_results_df.loc[curr_idx, "time_between_rewards"] = time_diff

    return pump_results_df


start_time = epochs_df[epochs_df["epoch_name"] == "04_r2"][
    "epoch_start_time"
].values[0]
start_time


# +
def assign_trial_type(well_name):
    if well_name is None:
        return None  # or return a default value depending on your requirement
    if pd.isna(well_name):  # Check if the value is NaN or None
        return None  # Or return a default value if preferred
    if "Handle" in well_name:
        return "Outbound"
    elif "Left" in well_name or "Right" in well_name:
        return "Inbound"
    else:
        return None  # In case there are any unknown well names


# Apply the function to the 'prev_well' column
pump_results_df["trial_type"] = pump_results_df["prev_well"].apply(
    assign_trial_type
)
pump_results_df
# -

pump_results_df

# Usage:
epoch = "06_r4"
subset_df = pump_results_df[pump_results_df["epoch"] == "10_r5"]
start_time = epochs_df[epochs_df["epoch_name"] == "08_r4"][
    "epoch_start_time"
].values[0]
test = calculate_time_between_rewards(subset_df, start_time)

test

filtered_pump_results_df = test[test["time_between_rewards"] < 90]

df_well = filtered_pump_results_df
df_well

# +
import matplotlib.pyplot as plt

# Map well names to simplified labels
well_labels = ["Left", "Center", "Right", "Handle"]

fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

for ax, well in zip(axes, well_labels):
    plot_df = pump_results_df  # [pump_results_df['epoch'] == epoch]
    df_well = plot_df[plot_df["well_name"].str.contains(well)]

    ax.scatter(
        df_well.index,
        df_well["pump_triggered"],
        c=df_well["pump_triggered"].map({True: "green", False: "red"}),
        s=50,
    )
    ax.set_ylabel(f"{well} Well")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.grid(False)

axes[-1].set_xlabel("Trial number")
plt.suptitle("Rewarded Pokes by Well 6/23")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# -

rewarded_pokes = pump_results_df[pump_results_df["pump_triggered"]]
rewarded_pokes

len(rewarded_pokes)

# +
import pandas as pd
import numpy as np


def check_pump_for_pokes(
    processed_df, poke_name="Left_poke", pump_name="reward_pump", window=0.5
):
    """
    Check if pokes triggered the pump.

    Parameters
    ----------
    processed_df : pd.DataFrame
        Processed DIO events with columns: ['dio_name', 'dio_event_times', 'dio_event_values']
    poke_name : str
        Name of the poke to check
    pump_name : str
        Name of the pump DIO channel
    window : float
        Time window (seconds) after poke to consider pump ON as triggered

    Returns
    -------
    pd.DataFrame
        Columns: ['poke_time', 'pump_delivered', 'pump_times_after_poke']
    """

    # Extract poke times
    poke_times = processed_df[processed_df["dio_name"] == poke_name][
        "dio_event_times"
    ].values
    if len(poke_times) == 0:
        return pd.DataFrame(
            columns=["poke_time", "pump_delivered", "pump_times_after_poke"]
        )
    poke_times = np.concatenate(poke_times)

    # Extract pump times when it turns on (value==1)
    pump_rows = processed_df[processed_df["dio_name"] == pump_name]
    if len(pump_rows) == 0:
        pump_times = np.array([])
    else:
        pump_times = np.concatenate(
            [
                times[np.array(values) == 1]
                for times, values in zip(
                    pump_rows["dio_event_times"], pump_rows["dio_event_values"]
                )
            ]
        )

    # Check for pump within window after each poke
    results = []
    for poke in poke_times:
        triggered_times = pump_times[
            (pump_times >= poke) & (pump_times <= poke + window)
        ]
        results.append(
            {
                "poke_time": poke,
                "pump_delivered": len(triggered_times) > 0,
                "pump_times_after_poke": triggered_times.tolist(),
            }
        )

    return pd.DataFrame(results)


# -

from mpfc_opto.behavior.em_module import EM_main, RunEM

pump_results_df["resp"] = pump_results_df["pump_triggered"].astype(int)

pump_results_df["rat_name"] = "Caius"

# +
mapping = {"02_r1": 1, "04_r2": 2, "06_r3": 3, "08_r4": 4, "10_r5": 5}

pump_results_df["epoch_number"] = pump_results_df["epoch"].map(mapping)
# -

pump_results_df

outbound = pump_results_df[
    pump_results_df["trial_type"] == "Inbound"
].reset_index()
outbound

trial = outbound[
    (outbound["epoch_number"] > 22) & (outbound["epoch_number"] < 24)
]
trial

pump_results_df

agg_df = pump_results_df

agg_df["trial"] = agg_df.reset_index().index

agg_df

# +
df = agg_df
kernel_size = 25

# Compute rolling mean per trial_type
colors = {
    "Inbound": "#043D11",
    "Outbound": "#CE0CB4",
}

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

for ax, (ttype, color) in zip(axes, colors.items()):
    sub = (
        df[df["trial_type"] == ttype]
        .copy()
        .sort_values("trial")
        .reset_index(drop=True)
    )
    sub["trial_num"] = np.arange(1, len(sub) + 1)  # reset trial number

    smoothed = (
        sub["resp"]
        .rolling(window=kernel_size, center=False, min_periods=1)
        .mean()
    )
    sem = sub["resp"].rolling(
        window=kernel_size, center=False, min_periods=1
    ).std() / np.sqrt(kernel_size)

    # jitter = np.random.uniform(-0.01, 0.01, len(sub))
    ax.scatter(
        sub["trial_num"],
        sub["resp"],
        color=color,
        alpha=0.18,
        s=18,
        zorder=1,
        linewidths=0,
    )

    ax.plot(sub["trial_num"], smoothed, color=color, linewidth=2.5, zorder=3)

    ax.fill_between(
        sub["trial_num"],
        smoothed - sem,
        smoothed + sem,
        color=color,
        alpha=0.15,
        zorder=2,
    )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(
        ttype.capitalize(),
        fontsize=13,
        #  fontweight='bold',
        color=color,
    )
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

axes[0].set_ylabel("P(Correct)", fontsize=12)
axes[1].set_ylabel("P(Correct)", fontsize=12)
fig.suptitle(
    f"Fork Track Performance SWR-Triggered On Time (Smoothing Kernel = {kernel_size} trials)",
    fontsize=13,
    #  fontweight='bold',
    y=1.02,
)
plt.tight_layout()
plt.show()

# +
df = agg_df[agg_df["epoch_number"] > 3]
kernel_size = 25

# Compute rolling mean per trial_type
colors = {
    "Inbound": "#043D11",
    "Outbound": "#CE0CB4",
}

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

for ax, (ttype, color) in zip(axes, colors.items()):
    sub = (
        df[df["trial_type"] == ttype]
        .copy()
        .sort_values("trial")
        .reset_index(drop=True)
    )
    sub["trial_num"] = np.arange(1, len(sub) + 1)  # reset trial number

    smoothed = (
        sub["resp"]
        .rolling(window=kernel_size, center=True, min_periods=1)
        .mean()
    )
    sem = sub["resp"].rolling(
        window=kernel_size, center=True, min_periods=1
    ).std() / np.sqrt(kernel_size)

    # jitter = np.random.uniform(-0.01, 0.01, len(sub))
    ax.scatter(
        sub["trial_num"],
        sub["resp"],
        color=color,
        alpha=0.18,
        s=18,
        zorder=1,
        linewidths=0,
    )

    ax.plot(sub["trial_num"], smoothed, color=color, linewidth=2.5, zorder=3)

    ax.fill_between(
        sub["trial_num"],
        smoothed - sem,
        smoothed + sem,
        color=color,
        alpha=0.15,
        zorder=2,
    )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(
        ttype.capitalize(),
        fontsize=13,
        #  fontweight='bold',
        color=color,
    )
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

axes[0].set_ylabel("P(Correct)", fontsize=12)
axes[1].set_ylabel("P(Correct)", fontsize=12)
fig.suptitle(
    f"W-Track Performance SWR-Triggered On Time (Smoothing Kernel = {kernel_size} trials)",
    fontsize=13,
    #  fontweight='bold',
    y=1.02,
)
plt.tight_layout()
plt.show()

# +
df = agg_df
kernel_size = 25

# Compute rolling mean per trial_type
colors = {
    "Inbound": "#043D11",
    "Outbound": "#CE0CB4",
}

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

for ax, (ttype, color) in zip(axes, colors.items()):
    sub = (
        df[df["trial_type"] == ttype]
        .copy()
        .sort_values("trial")
        .reset_index(drop=True)
    )
    sub["trial_num"] = np.arange(1, len(sub) + 1)  # reset trial number

    epoch_ranges = sub.groupby("epoch_number")["trial_num"].agg(["min", "max"])

    # One purple block spanning epochs 18-21
    ax.axvspan(
        epoch_ranges.loc[4, "min"],
        epoch_ranges.loc[5, "max"],
        color="purple",
        alpha=0.08,
        zorder=0,
    )

    smoothed = (
        sub["resp"]
        .rolling(window=kernel_size, center=True, min_periods=1)
        .mean()
    )
    sem = sub["resp"].rolling(
        window=kernel_size, center=True, min_periods=1
    ).std() / np.sqrt(kernel_size)

    # jitter = np.random.uniform(-0.01, 0.01, len(sub))
    ax.scatter(
        sub["trial_num"],
        sub["resp"],
        color=color,
        alpha=0.18,
        s=18,
        zorder=1,
        linewidths=0,
    )

    ax.plot(sub["trial_num"], smoothed, color=color, linewidth=2.5, zorder=3)

    ax.fill_between(
        sub["trial_num"],
        smoothed - sem,
        smoothed + sem,
        color=color,
        alpha=0.15,
        zorder=2,
    )
    # ax.set_facecolor('#f0e6ff')

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(
        ttype.capitalize(),
        fontsize=13,
        #  fontweight='bold',
        color=color,
    )
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

axes[0].set_ylabel("P(Correct)", fontsize=12)
axes[1].set_ylabel("P(Correct)", fontsize=12)
fig.suptitle(
    "Caius Fork Track Performance On Time + Delayed",
    fontsize=13,
    #  fontweight='bold',
    y=1.02,
)
plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define p_init and subplot dimensions
p_init = 0.5
subplot_width = 6
subplot_height = 6

# Set the animal you want to plot
rat_name = "Caius"  # Replace with the specific rat's name
trial_types = ["Inbound", "Outbound"]  # Replace with your actual trial types

# Initialize figure
num_columns = 1  # Single animal, so one column
num_rows = len(trial_types)  # Number of trial types
fig, axes = plt.subplots(
    num_rows,
    num_columns,
    figsize=(subplot_width * num_columns, subplot_height * num_rows),
    sharex=True,
    sharey=True,
)

# Plot learning curves
data_list = []

# Loop over trial types and optogenetic conditions for the selected rat
for trial_type_idx, trial_type in enumerate(trial_types):
    # Filter the DataFrame for the specific rat and trial_type
    filtered_df = pump_results_df[
        (pump_results_df["rat_name"] == rat_name)
        & (pump_results_df["trial_type"] == trial_type)
    ].reset_index(drop=True)

    if filtered_df.empty:
        continue

    # Extract response values and trial numbers
    resp_values = filtered_df["resp"]
    trial_number = filtered_df.index

    if len(resp_values) == 0:
        continue

    # Set the figure and axis for plotting
    fig_ax_list = [
        fig,
        axes[trial_type_idx],
    ]  # Adjusted for one animal (axes is now 1D)

    # Call the main plotting function (EM_main) and get the plot elements
    fig, ax, pll, pul, pmode = EM_main(
        resp_values,
        p_init,
        fig_ax_list,
        color="blue",
        trial_number=trial_number,
    )

    # Store results for later use (data for learning trials)
    data_list.append((rat_name, trial_type, pll, trial_number, pmode))

    # Set the title and axis labels
    ax.set_title(
        f"{trial_type} - {rat_name}"
    )  # Title for each subplot (Trial Type + Rat Name)
    ylabel = "p(correct)" if trial_type_idx == len(trial_types) - 1 else ""
    xlabel = "trial number" if trial_type_idx == len(trial_types) - 1 else ""
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

# Adjust layout to avoid overlap
fig.tight_layout()


# Add learning trials to the dataframe
def get_first_entry_else_nan(x):
    return x[0] if len(x) > 0 else np.nan


df = pd.DataFrame(
    data_list,
    columns=["rat_name", "trial_type", "pll", "trial_number", "pmode"],
)

# Now you can index it correctly. For example:
pll_df = df[["rat_name", "trial_type", "pll", "trial_number", "pmode"]]

learning_trials = [
    get_first_entry_else_nan(np.where(pll > 0.5)[0]) for pll in pll_df.pll
]


def indxVal_else_nan(values, indx):
    return values[indx] if indx is not np.nan else np.nan


# Create the learning_trial column
learning_trials = [
    indxVal_else_nan(trial_number, ind)
    for trial_number, ind in zip(pll_df.trial_number, learning_trials)
]
pll_df["learning_trial"] = learning_trials

# Show legend

# axes[0].legend(loc='lower right', t/itle='optogenetics')

# Optionally, display the plot
plt.show()
# -

learning_trials

incorrect = pump_results_df[~pump_results_df["pump_triggered"]]
center_incorrect = incorrect[incorrect["well_name"] == "Center_poke"]
center_incorrect

# +
df = pump_results_df.copy()

# group by epoch_number
group = df.groupby("epoch_number")

# ----------------------------------------
# Trial counts (denominator)
# ----------------------------------------
inbound_trials = group.apply(lambda g: (g["trial_type"] == "Inbound").sum())
outbound_trials = group.apply(lambda g: (g["trial_type"] == "Outbound").sum())

# ----------------------------------------
# Reward counts (using resp column)
# ----------------------------------------
inbound_rewards = group.apply(
    lambda g: g[(g["trial_type"] == "Inbound")]["resp"].sum()
)
outbound_rewards = group.apply(
    lambda g: g[(g["trial_type"] == "Outbound")]["resp"].sum()
)

# ----------------------------------------
# Reward rate = proportion correct
# ----------------------------------------
inbound_reward_rate = inbound_rewards / inbound_trials.replace(0, pd.NA)
outbound_reward_rate = outbound_rewards / outbound_trials.replace(0, pd.NA)

# ----------------------------------------
# Total reward rate (all trials)
# ----------------------------------------
total_rewards = group["resp"].sum()
total_trials = group["resp"].count()
total_reward_rate = total_rewards / total_trials

# Build summary dataframe
summary = pd.DataFrame(
    {
        "inbound_trials": inbound_trials,
        "outbound_trials": outbound_trials,
        "inbound_rewards": inbound_rewards,
        "outbound_rewards": outbound_rewards,
        "inbound_reward_rate": inbound_reward_rate,
        "outbound_reward_rate": outbound_reward_rate,
        "total_reward_rate": total_reward_rate,
        "total_rewards": total_rewards,
    }
)

summary = summary.reset_index()
# -

summary

# +
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

# Total reward rate
axes[0].plot(
    summary["epoch_number"],
    summary["total_reward_rate"],
    marker="o",
    color="tab:red",
    label="Total Reward Rate",
)
axes[0].set_ylabel("Total Reward Rate")
axes[0].set_title("Reward Proportion Per Trial Type")
axes[0].legend()
axes[0].grid(False)
axes[0].axvspan(4.5, 5.5, color="purple", alpha=0.3)

# Inbound reward rate
axes[1].plot(
    summary["epoch_number"],
    summary["inbound_reward_rate"],
    marker="o",
    color="tab:blue",
    label="Inbound Reward Rate",
)
axes[1].set_ylabel("Inbound Reward Rate")
axes[1].legend()
axes[1].grid(False)
axes[1].axvspan(4.5, 5.5, color="purple", alpha=0.3)

# Outbound reward rate
axes[2].plot(
    summary["epoch_number"],
    summary["outbound_reward_rate"],
    marker="o",
    color="tab:orange",
    label="Outbound Reward Rate",
)
axes[2].set_ylabel("Outbound Reward Rate")
axes[2].set_xlabel("Epoch Number")
axes[2].legend()
axes[2].grid(False)
axes[2].axvspan(4.5, 5.5, color="purple", alpha=0.3)


x_vals = summary["epoch_number"]

for ax in axes:
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1)


plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt

sessions = summary["epoch_number"]  # epoch_number

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    sessions,
    summary["total_rewards"],
    marker="o",
    color="tab:red",
    label="Total Rewards",
)

ax.plot(
    sessions,
    summary["inbound_rewards"],
    marker="o",
    color="tab:blue",
    label="Inbound Rewards",
)

ax.plot(
    sessions,
    summary["outbound_rewards"],
    marker="o",
    color="tab:orange",
    label="Outbound Rewards",
)

ax.set_ylabel("Number of Rewards")
ax.set_xlabel("Epoch Number")
ax.set_title("Number of Rewards per Epoch")
ax.legend()
ax.grid(False)
ax.axvspan(4.5, 5.5, color="purple", alpha=0.3)

plt.tight_layout()
plt.show()
