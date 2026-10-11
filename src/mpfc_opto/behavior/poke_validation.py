"""Position-based validation of DIO poke events.

Shared by the fork-track and W-track pipelines, which ran byte-comparable
copies of this: the bodies agreed to 94% (`validate_poke_events`) and 89%
(`plot_validation`), differing only in whether the well positions and
thresholds arrived as arguments or as attributes of a validator object.

Imports no DataJoint or spyglass, so this is importable -- and testable --
without a database.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def interpolate_position(position_times, position_x, position_y, query_times):
    """
    Interpolate x/y position at arbitrary query times.

    Parameters
    ----------
    position_times : np.array
        Strictly increasing array of position sample times.
    position_x : np.array
        X coordinates at each sample time.
    position_y : np.array
        Y coordinates at each sample time.
    query_times : np.array
        Times at which to interpolate position.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with interpolated (x, y) positions.
    """
    query_times = np.asarray(query_times)
    position_times = np.asarray(position_times)

    # Guarded: `.min()` on an empty array raises, and an empty query is legal
    # -- a session can have no pokes to validate.
    if query_times.size:
        logger.debug(
            f"interpolating {query_times.size} times in "
            f"[{query_times.min()}, {query_times.max()}] from position "
            f"spanning [{position_times[0]}, {position_times[-1]}]"
        )
    # Not an assert: assertions are stripped under `python -O`, and np.interp
    # silently returns garbage for unsorted sample times.
    if not np.all(np.diff(position_times) > 0):
        raise ValueError("position_times is not strictly increasing")

    interp_x = np.interp(query_times, position_times, position_x)
    interp_y = np.interp(query_times, position_times, position_y)

    return np.column_stack((interp_x, interp_y))


def validate_poke_events(
    well_positions,
    poke_times,
    poke_names,
    poke_values=None,
    position_times=None,
    position_x=None,
    position_y=None,
    distance_threshold=15.0,
    max_speed=150.0,
    plot=True,
):
    """
    Validate poke events based on animal position.

    Parameters
    ----------
    well_positions : dict
        Map of well names to (x, y) coordinates.
    poke_times : np.array
        Timestamps of poke events.
    poke_names : np.array
        Names of poked wells.
    poke_values : np.array, optional
        Poke values (0 or 1). Defaults to all 1s if not provided.
    position_times : np.array
        Position sample times.
    position_x : np.array
        X coordinates.
    position_y : np.array
        Y coordinates.
    distance_threshold : float
        Maximum distance from well for a valid poke (in position units).
    max_speed : float
        Maximum plausible speed (position units per second).
    plot : bool
        Whether to create a validation plot.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'valid_pokes': DataFrame of valid pokes
        - 'invalid_pokes': DataFrame of rejected pokes
        - 'summary': dict with statistics
    """
    # ----------------------------------
    # EARLY EXIT: no position data
    # ----------------------------------
    if (
        position_times is None
        or position_x is None
        or position_y is None
        or len(position_times) == 0
    ):
        poke_df = pd.DataFrame(
            {
                "time": poke_times,
                "well_name": poke_names,
                "value": (
                    poke_values
                    if poke_values is not None
                    else np.ones(len(poke_times), dtype=int)
                ),
            }
        )

        summary = {
            "total_pokes": len(poke_df),
            "valid_pokes": len(poke_df),
            "invalid_pokes": 0,
            "percent_valid": 100.0 if len(poke_df) else 0,
            "note": "Position validation skipped (no position data)",
        }

        return {
            "valid_pokes": poke_df.reset_index(drop=True),
            "invalid_pokes": poke_df.iloc[0:0],
            "summary": summary,
        }

    if poke_values is None:
        poke_values = np.ones(len(poke_times), dtype=int)

    animal_positions = interpolate_position(
        position_times, position_x, position_y, poke_times
    )

    poke_df = pd.DataFrame(
        {
            "time": poke_times,
            "well_name": poke_names,
            "value": poke_values,
            "animal_x": animal_positions[:, 0],
            "animal_y": animal_positions[:, 1],
        }
    )

    # Compute distance to wells
    distances, well_x, well_y = [], [], []
    for _, row in poke_df.iterrows():
        if row["well_name"] in well_positions:
            wx, wy = well_positions[row["well_name"]]
            dist = np.sqrt(
                (row["animal_x"] - wx) ** 2 + (row["animal_y"] - wy) ** 2
            )
        else:
            wx, wy, dist = np.nan, np.nan, np.inf
        distances.append(dist)
        well_x.append(wx)
        well_y.append(wy)

    poke_df["distance_to_well"] = distances
    poke_df["well_x"] = well_x
    poke_df["well_y"] = well_y

    # Validate by distance
    valid_mask = poke_df["distance_to_well"] <= distance_threshold
    valid_pokes = poke_df[valid_mask].copy()
    invalid_pokes = poke_df[~valid_mask].copy()

    # Speed check for valid pokes
    if len(valid_pokes) > 1:
        speed_mask = np.ones(len(valid_pokes), dtype=bool)
        for i in range(1, len(valid_pokes)):
            prev = valid_pokes.iloc[i - 1]
            curr = valid_pokes.iloc[i]

            dist = np.sqrt(
                (curr["animal_x"] - prev["animal_x"]) ** 2
                + (curr["animal_y"] - prev["animal_y"]) ** 2
            )
            dt = curr["time"] - prev["time"]

            if dt > 0 and dist / dt > max_speed:
                speed_mask[i] = False

        invalid_speed_pokes = valid_pokes[~speed_mask]
        valid_pokes = valid_pokes[speed_mask]
        invalid_pokes = pd.concat([invalid_pokes, invalid_speed_pokes])

    summary = {
        "total_pokes": len(poke_df),
        "valid_pokes": len(valid_pokes),
        "invalid_pokes": len(invalid_pokes),
        "percent_valid": (
            (100 * len(valid_pokes) / len(poke_df)) if len(poke_df) > 0 else 0
        ),
    }

    logger.info("\nPosition validation summary:")
    logger.info(f"  Total pokes: {summary['total_pokes']}")
    logger.info(
        f"  Valid pokes: {summary['valid_pokes']} ({summary['percent_valid']:.1f}%)"
    )
    logger.info(f"  Invalid pokes: {summary['invalid_pokes']}")

    if len(invalid_pokes) > 0:
        logger.info("\nInvalid poke details:")
        for _, row in invalid_pokes.iterrows():
            logger.info(
                f"  {row['well_name']} at t={row['time']:.2f}s, distance={row['distance_to_well']:.1f}"
            )

    if plot:
        plot_validation(
            valid_pokes,
            invalid_pokes,
            position_x,
            position_y,
            well_positions,
            distance_threshold,
        )

    return {
        "valid_pokes": valid_pokes.reset_index(drop=True),
        "invalid_pokes": invalid_pokes.reset_index(drop=True),
        "summary": summary,
    }


def plot_validation(
    valid_pokes,
    invalid_pokes,
    position_x,
    position_y,
    well_positions,
    distance_threshold,
):
    """Create visualization of position-based poke validation."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        position_x,
        position_y,
        "-",
        alpha=0.2,
        linewidth=0.5,
        label="Trajectory",
    )

    for well_name, (wx, wy) in well_positions.items():
        ax.plot(wx, wy, "ko", markersize=10)
        ax.text(wx, wy, well_name, ha="center", fontsize=9)
        circle = plt.Circle((wx, wy), distance_threshold, alpha=0.2)
        ax.add_patch(circle)

    if len(valid_pokes) > 0:
        ax.plot(
            valid_pokes["animal_x"],
            valid_pokes["animal_y"],
            "go",
            markersize=8,
            label="Valid pokes",
            alpha=0.7,
        )

    if len(invalid_pokes) > 0:
        ax.plot(
            invalid_pokes["animal_x"],
            invalid_pokes["animal_y"],
            "rx",
            markersize=10,
            markeredgewidth=2,
            label="Invalid pokes",
        )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Position-Based DIO Validation")
    ax.legend()
    ax.axis("equal")
    plt.tight_layout()
    plt.show()
