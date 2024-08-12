import numpy as np
import pandas as pd


def adaptive_threshold_breakout(
    data: pd.DataFrame | np.ndarray | list,
    column: str = None,
    window_size: int = 20,
    n_sigmas: float = 3.0,
    overlap: int = 10,
) -> tuple[list[int], np.ndarray]:
    """
    Perform adaptive threshold breakout detection on a time series.

    This function detects breakouts by comparing each point to a threshold that
    adapts based on the local mean and standard deviation of the time series.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - column (str, optional): Name of the column to analyze if data is a DataFrame
    - window_size (int): Size of the sliding window for local statistics calculation
    - n_sigmas (float): Number of standard deviations to use for the threshold
    - overlap (int): Number of overlapping points between consecutive windows

    Returns:
    - Tuple[List[int], np.ndarray]:
        - List of indices where breakouts were detected
        - Array of thresholds for each point in the time series
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

    n = len(series)
    step = window_size - overlap

    # Initialize arrays to store local statistics and thresholds
    local_means = np.zeros(n)
    local_stds = np.zeros(n)
    thresholds = np.zeros(n)

    # Calculate local statistics using a sliding window
    for i in range(0, n - window_size + 1, step):
        window = series[i : i + window_size]
        local_mean = np.mean(window)
        local_std = np.std(window)

        # Fill the arrays with calculated statistics
        local_means[i : i + window_size] = local_mean
        local_stds[i : i + window_size] = local_std

    # Handle edge cases for the first and last windows
    local_means[: window_size // 2] = local_means[window_size // 2]
    local_means[-(window_size // 2) :] = local_means[-(window_size // 2) - 1]
    local_stds[: window_size // 2] = local_stds[window_size // 2]
    local_stds[-(window_size // 2) :] = local_stds[-(window_size // 2) - 1]

    # Calculate adaptive thresholds
    thresholds = local_means + n_sigmas * local_stds

    # Detect breakouts
    breakouts = np.where(np.abs(series - local_means) > thresholds - local_means)[0]

    return breakouts.tolist(), thresholds
