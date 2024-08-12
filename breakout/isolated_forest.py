import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def isolation_forest_breakout(
    data: pd.DataFrame | np.ndarray,
    column: str | None = None,
    contamination: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 100,
    max_samples: str | int = "auto",
) -> tuple[list[int], np.ndarray]:
    """
    Perform Isolation Forest breakout detection on a time series.

    This function uses the Isolation Forest algorithm to detect anomalies
    (potential breakouts) in the given time series data. It can work with
    both pandas DataFrames and NumPy arrays.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - column (str, optional): Name of the column to analyze if data is a DataFrame
    - contamination (float): The proportion of outliers in the data set
    - random_state (int): Controls the pseudo-randomness of the selection of samples
    - n_estimators (int): The number of base estimators in the ensemble
    - max_samples (Union[str, int]): The number of samples to draw to train each base estimator

    Returns:
    - Tuple[List[int], np.ndarray]:
        - List of indices where breakouts were detected
        - Array of anomaly scores (-1 for inliers, 1 for outliers)
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
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array")

    # Reshape the input for scikit-learn
    X = series.reshape(-1, 1)

    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples,
    )

    # Fit the model and predict anomalies
    y_pred = clf.fit_predict(X)

    # Get the anomaly scores
    # anomaly_scores = clf.score_samples(X)

    # Identify breakout points (where y_pred == -1)
    breakouts = np.where(y_pred == -1)[0].tolist()

    return breakouts, y_pred
