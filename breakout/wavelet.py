import numpy as np
import pandas as pd
import pywt


def wavelet_breakout_detection(
    data: pd.DataFrame | np.ndarray,
    column: None | str = None,
    wavelet: str = "db4",
    level: int = 5,
    threshold: float = 1.5,
) -> list[int]:
    """
    Perform breakout detection using Wavelet Transform.

    Parameters:
    - data (Union[pd.DataFrame, np.ndarray]): The time series data
    - column (str, optional): Name of the column to analyze if data is a DataFrame
    - wavelet (str): Wavelet to use for the transform (default: 'db4')
    - level (int): Decomposition level (default: 5)
    - threshold (float): Threshold for detecting breakouts (default: 1.5)

    Returns:
    - List[int]: List of indices where breakouts were detected
    """
    # Input validation and conversion to numpy array
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when input is a DataFrame")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column].values
    elif isinstance(data, np.ndarray):
        series = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or numpy array")

    # Ensure the input is 1D
    if series.ndim != 1:
        raise ValueError("Input series must be 1-dimensional")

    if series.size == 0:
        raise ValueError("Input data is size 0")

    # Check if the input is constant
    if np.all(series == series[0]):
        return []  # Return an empty list for constant input

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Calculate the detail coefficients' standard deviation
    detail_coeffs = coeffs[1:]
    std_dev = np.sqrt(np.sum([np.sum(d**2) for d in detail_coeffs]) / len(series))

    # Identify significant coefficients
    significant_coeffs = [np.abs(d) > threshold * std_dev for d in detail_coeffs]

    # Reconstruct the signal using only significant coefficients
    reconstructed = np.zeros_like(series)
    for i, sig_coeffs in enumerate(significant_coeffs):
        d_reconstructed = pywt.idwt(
            None, coeffs[i + 1] * sig_coeffs, wavelet, mode="symmetric"
        )
        # Pad or truncate d_reconstructed to match reconstructed length
        if len(d_reconstructed) > len(reconstructed):
            d_reconstructed = d_reconstructed[: len(reconstructed)]
        elif len(d_reconstructed) < len(reconstructed):
            d_reconstructed = np.pad(
                d_reconstructed,
                (0, len(reconstructed) - len(d_reconstructed)),
                "constant",
            )
        reconstructed += d_reconstructed

    # Detect breakouts
    breakouts = np.where(np.abs(reconstructed) > threshold * np.std(reconstructed))[0]

    return breakouts.tolist()
