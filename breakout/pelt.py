import numpy as np
import pandas as pd
from ruptures import Pelt


def pelt_breakout_detection(
    data: pd.DataFrame | np.ndarray | list,
    column: str | None = None,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    penalty: float = 1.0,
) -> tuple[list[int], float]:
    """
    Perform PELT (Pruned Exact Linear Time) breakout detection on a time series.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - column (str, optional): Name of the column to analyze (only for DataFrame input)
    - model (str): Cost model ('l1', 'l2', 'rbf', etc.). Default is 'l2'.
    - min_size (int): Minimum segment length. Default is 2.
    - jump (int): Jump value for faster computation. Default is 5.
    - penalty (float): Penalty value for the PELT algorithm. Default is 1.0.

    Returns:
    - Tuple[List[int], float]:
        - List of indices where breakouts were detected
        - Total cost of the segmentation
    """

    # Input validation and extraction of signal
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when input is a DataFrame")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        signal = data[column].values
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("NumPy array input must be 1-dimensional")
        signal = data
    elif isinstance(data, list):
        signal = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array")

    # Initialize and fit the PELT model
    algo = Pelt(model=model, min_size=min_size, jump=jump).fit(signal)

    # Detect breakpoints
    breakpoints = algo.predict(pen=penalty)

    # Calculate the total cost of the segmentation
    total_cost = algo.cost.sum_of_costs(breakpoints)

    # Remove the last breakpoint (which is always the length of the signal)
    breakpoints = breakpoints[:-1]

    return breakpoints, total_cost
