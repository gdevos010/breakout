import numpy as np
import pandas as pd


def windowed_variance_breakout(
    data: pd.DataFrame | np.ndarray | list,
    column: str | None = None,
    window_size: int = 20,
    overlap: int = 10,
    threshold: float = 2.0,
) -> tuple[list[int], np.ndarray]:
    """
    Perform windowed variance breakout detection on a time series.

    This function detects changes in volatility by comparing variances across different time windows.
    It can work with both pandas DataFrames and NumPy arrays.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - column (str, optional): Name of the column to analyze if data is a DataFrame
    - window_size (int): Size of the rolling window for variance calculation
    - overlap (int): Number of overlapping points between consecutive windows
    - threshold (float): Number of standard deviations to consider as a breakout

    Returns:
    - Tuple[List[int], np.ndarray]:
        - List of indices where breakouts were detected
        - Array of variance ratios for each window
    """

    # Input validation and conversion to numpy array
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when input is a DataFrame")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column].values
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("NumPy array input must be 1-dimensional")
        series = data
    elif isinstance(data, list):
        series = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array")

    # Validate other parameters
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Invalid window_size or overlap parameters")

    # Calculate the number of windows
    n = len(series)
    step = window_size - overlap
    num_windows = (n - window_size) // step + 1

    # Calculate variances for each window
    variances = np.array(
        [np.var(series[i * step : i * step + window_size]) for i in range(num_windows)]
    )

    # Calculate variance ratios between consecutive windows
    variance_ratios = variances[1:] / variances[:-1]

    # Detect breakouts using the log of variance ratios
    log_variance_ratios = np.log(variance_ratios)
    mean_log_ratio = np.mean(log_variance_ratios)
    std_log_ratio = np.std(log_variance_ratios)

    breakouts = []
    for i in range(len(log_variance_ratios)):
        if abs(log_variance_ratios[i] - mean_log_ratio) > threshold * std_log_ratio:
            # Convert window index to data index
            breakout_index = (i + 1) * step + window_size // 2
            breakouts.append(breakout_index)

    return breakouts, variance_ratios
