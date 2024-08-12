from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd


class CUSUMBreakoutDetector:
    def __init__(
        self,
        threshold: float = 1.0,
        drift: float = 0.0,
        target_method: Literal["fixed", "moving", "exponential"] = "exponential",
        target_params: dict | None = None,
    ):
        """
        Initialize the CUSUM Breakout Detector.

        Parameters:
        - threshold (float): The threshold for detecting a breakout
        - drift (float): The acceptable drift in the process
        - target_method (str): Method to calculate the target value ('fixed', 'moving', or 'exponential')
        - target_params (dict): Parameters for the target value calculation method
        """
        self.threshold = threshold
        self.drift = drift
        self.target_method = target_method
        self.target_params = target_params or {"alpha": 0.1}

        self.target: None | float = None
        self.cumsum_pos = 0.0
        self.cumsum_neg = 0.0
        self.breakouts: list[int] = []
        self.cumsum_values = []
        self.data_buffer = []

    def update(self, value: float) -> tuple[bool, float]:
        """
        Update the detector with a new data point.

        Parameters:
        - value (float): The new data point

        Returns:
        - Tuple[bool, float]:
            - Boolean indicating if a breakout was detected
            - The current CUMSUM value
        """
        self.data_buffer.append(value)

        # Initialize or update target value
        if self.target is None:
            self.target = value
        elif self.target_method == "moving":
            window = self.target_params.get("window", 10)
            self.target = np.mean(self.data_buffer[-window:])
        elif self.target_method == "exponential":
            alpha = self.target_params.get("alpha", 0.1)
            self.target = alpha * value + (1 - alpha) * self.target
        elif self.target_method == "fixed":
            # TODO double check nothing is needed for fixed
            a = 0
        else:
            raise ValueError(
                "Invalid target_method. Choose 'fixed', 'moving', or 'exponential'"
            )
        # Calculate positive and negative cumsums
        self.cumsum_pos = max(0.0, self.cumsum_pos + value - self.target - self.drift)
        self.cumsum_neg = max(0.0, self.cumsum_neg - value + self.target - self.drift)

        # Check for breakouts
        cumsum_value = max(self.cumsum_pos, self.cumsum_neg)
        is_breakout = cumsum_value > self.threshold

        if is_breakout:
            self.breakouts.append(len(self.data_buffer) - 1)
            self.cumsum_pos = 0.0
            self.cumsum_neg = 0.0

        self.cumsum_values.append(cumsum_value)
        return is_breakout, cumsum_value

    def get_results(self) -> tuple[list[int], list[float]]:
        """
        Get the current results of the detector.

        Returns:
        - Tuple[List[int], List[float]]:
            - List of indices where breakouts were detected
            - List of CUMSUM values at each point
        """
        return self.breakouts, self.cumsum_values


def cumsum_breakout_detection_online(
    data: pd.DataFrame | pd.Series | np.ndarray | float,
    column: str | None = None,
    threshold: float = 1.0,
    drift: float = 0.0,
    target_method: Literal["fixed", "moving", "exponential"] = "exponential",
    target_params: dict | None = None,
    online: bool = False,
) -> tuple[list[int], list[float]] | CUSUMBreakoutDetector:
    """
    Perform CUMSUM (Cumulative Sum) breakout detection on a time series.
    Supports both batch and online/streaming detection.

    Parameters:
    - data (Union[pd.DataFrame, pd.Series, np.ndarray, float]): The time series data or a single data point
    - column (str, optional): Name of the column to analyze if data is a DataFrame
    - threshold (float): The threshold for detecting a breakout
    - drift (float): The acceptable drift in the process
    - target_method (str): Method to calculate the target value ('fixed', 'moving', or 'exponential')
    - target_params (dict): Parameters for the target value calculation method
    - online (bool): If True, returns a CUSUMBreakoutDetector object for streaming detection

    Returns:
    - Union[Tuple[List[int], List[float]], CUSUMBreakoutDetector]:
        - If online is False: Tuple containing list of breakout indices and list of CUMSUM values
        - If online is True: CUSUMBreakoutDetector object for streaming detection
    """
    detector = CUSUMBreakoutDetector(threshold, drift, target_method, target_params)

    if online:
        return detector

    # Input validation and conversion to numpy array
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when input is a DataFrame")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column].values
    elif isinstance(data, pd.Series):
        series = data.values
    elif isinstance(data, np.ndarray):
        series = data
    elif isinstance(data, (int, float)):
        series = np.array([data])
    else:
        raise ValueError(
            "Input data must be a pandas DataFrame, Series, numpy array, or a single numeric value"
        )

    # Process the data
    for value in series:
        detector.update(value)

    return detector.get_results()
