from typing import Literal

import numpy as np
import pandas as pd


def ma_breakout_detection(
    data: pd.DataFrame | np.ndarray,
    value_column: str | None = None,
    date_column: str | None = None,
    window_size: int = 20,
    threshold: float = 2.0,
    method: Literal["MA", "EMA"] = "MA",
) -> list[int]:
    # Input validation and conversion (unchanged)
    if isinstance(data, pd.DataFrame):
        if value_column is None or date_column is None:
            raise ValueError(
                "value_column and date_column must be specified for DataFrame input"
            )
        if value_column not in data.columns or date_column not in data.columns:
            raise ValueError(
                f"Columns '{value_column}' or '{date_column}' not found in DataFrame"
            )
        values = data[value_column].values
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("NumPy array input must be 1-dimensional")
        values = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array")

    if method not in ["MA", "EMA"]:
        raise ValueError("Method must be 'MA' or 'EMA'")

    if window_size > len(values):
        raise ValueError("Window size cannot be larger than the data length")

    breakouts = []

    if method == "MA":
        # Calculate Moving Average
        ma = np.convolve(values, np.ones(window_size), "valid") / window_size
        ma_std = np.array(
            [
                np.std(values[i : i + window_size])
                for i in range(len(values) - window_size + 1)
            ]
        )

        # Detect breakouts using Moving Average
        for i in range(len(ma)):
            if (
                values[i + window_size - 1] > ma[i] + threshold * ma_std[i]
                or values[i + window_size - 1] < ma[i] - threshold * ma_std[i]
            ):
                breakouts.append(i + window_size - 1)

    else:  # method == 'EMA'
        # Calculate Exponential Moving Average
        alpha = 2 / (window_size + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        ema_std = np.zeros_like(values)
        squared_diff = np.zeros_like(values)

        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
            squared_diff[i] = (values[i] - ema[i]) ** 2
            ema_std[i] = np.sqrt(
                alpha * squared_diff[i] + (1 - alpha) * (ema_std[i - 1] ** 2)
            )

            if i >= window_size - 1:
                if (
                    values[i] > ema[i] + threshold * ema_std[i]
                    or values[i] < ema[i] - threshold * ema_std[i]
                ):
                    breakouts.append(i)

    return breakouts
