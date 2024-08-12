import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter

log = logging.getLogger(__name__)


def plot_timeseries_with_breakouts(
    df: pd.DataFrame, breakout_indices: list[int], name: str
) -> None:
    """
    Plot the time series with highlighted breakout points.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'date' and 'value' columns
    - breakout_indices (list): List of indices where breakouts occur
    - name (str): Name of the plot for saving
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the time series
    ax.plot(df["date"], df["value"], label="Time Series")

    # Highlight breakout points
    breakout_dates = df["date"].iloc[breakout_indices]
    breakout_values = df["value"].iloc[breakout_indices]
    ax.scatter(
        breakout_dates, breakout_values, color="red", s=50, label="Breakout Points"
    )

    # Customize the plot
    ax.set_title("Time Series with Breakout Points")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    file_path = (
        Path(os.path.abspath(__file__)).parent.parent
        / f"plots/{name}_breakout_data.png"
    )
    plt.savefig(file_path)
    plt.clf()
    log.info(file_path)
