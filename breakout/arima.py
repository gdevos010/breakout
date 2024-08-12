import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_breakout_detection(
    data: pd.DataFrame | np.ndarray | list,
    value_column: str = "value",
    date_column: str = "date",
    order: tuple[int, int, int] = (1, 1, 1),
    threshold: float = 3.0,
    freq: str = "D",
) -> list[int]:
    """
    Perform ARIMA-based breakout detection on a time series.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - value_column (str): Name of the column containing the time series values (for DataFrame input)
    - date_column (str): Name of the column containing the dates (for DataFrame input)
    - order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model
    - threshold (float): Number of standard deviations to consider as a breakout
    - freq (str): Frequency of the time series. Default is 'D' for daily.

    Returns:
    - List[int]: List of indices where breakouts were detected
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        if value_column not in data.columns or date_column not in data.columns:
            raise ValueError(
                f"Columns '{value_column}' or '{date_column}' not found in DataFrame"
            )
        ts = data.set_index(date_column)[value_column]
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.to_datetime(ts.index)
        ts = ts.asfreq(freq)  # Ensure the time series has a frequency
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("NumPy array input must be 1-dimensional")
        # Create a date range for the numpy array
        date_rng = pd.date_range(start="1/1/2000", periods=len(data), freq=freq)
        ts = pd.Series(data, index=date_rng)
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array")

    # Suppress the frequency inference warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

        # Fit ARIMA model
        model = ARIMA(ts, order=order)
        results = model.fit()

    # Get residuals
    residuals = results.resid

    # Calculate mean and standard deviation of residuals
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Identify breakouts
    breakouts = []
    for i, residual in enumerate(residuals):
        if abs(residual - mean_residual) > threshold * std_residual:
            breakouts.append(i)

    return breakouts
