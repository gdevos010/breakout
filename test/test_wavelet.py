import numpy as np
import pandas as pd
import pytest

from breakout import (
    wavelet_breakout_detection,
)  # Replace 'your_module' with the actual module name


def test_numpy_input():
    # Test with numpy array input
    data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
    breakouts = wavelet_breakout_detection(data)
    assert isinstance(breakouts, list)
    assert all(isinstance(b, int) for b in breakouts)


def test_pandas_input():
    # Test with pandas DataFrame input
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2021-01-01", periods=1000, freq="h"),
            "value": np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000),
        }
    )
    breakouts = wavelet_breakout_detection(df, column="value")
    assert isinstance(breakouts, list)
    assert all(isinstance(b, int) for b in breakouts)


def test_invalid_input():
    # Test with invalid input types
    with pytest.raises(ValueError):
        wavelet_breakout_detection([1, 2, 3])  # List input should raise ValueError

    with pytest.raises(ValueError):
        wavelet_breakout_detection(
            "not a valid input"
        )  # String input should raise ValueError


def test_missing_column():
    # Test with missing column name for DataFrame input
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Column name must be specified when input is a DataFrame"
    ):
        wavelet_breakout_detection(df)


def test_invalid_column():
    # Test with invalid column name for DataFrame input
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Column 'invalid_column' not found in DataFrame"
    ):
        wavelet_breakout_detection(df, column="invalid_column")


def test_multidimensional_input():
    # Test with multidimensional input
    data = np.random.rand(10, 10)
    with pytest.raises(ValueError, match="Input series must be 1-dimensional"):
        wavelet_breakout_detection(data)


def test_parameter_variations():
    # Test with different parameter combinations
    data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)

    breakouts1 = wavelet_breakout_detection(
        data, wavelet="haar", level=3, threshold=2.0
    )
    breakouts2 = wavelet_breakout_detection(
        data, wavelet="sym4", level=7, threshold=1.0
    )

    assert isinstance(breakouts1, list)
    assert isinstance(breakouts2, list)
    assert (
        breakouts1 != breakouts2
    )  # Different parameters should yield different results


def test_empty_input():
    # Test with empty input
    with pytest.raises(ValueError):
        wavelet_breakout_detection(np.array([]))


def test_constant_input():
    # Test with constant input
    data = np.ones(1000)
    breakouts = wavelet_breakout_detection(data)
    assert len(breakouts) == 0  # No breakouts should be detected in constant data


def test_result_consistency():
    # Test for consistency of results
    data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
    breakouts1 = wavelet_breakout_detection(data)
    breakouts2 = wavelet_breakout_detection(data)
    assert breakouts1 == breakouts2  # Results should be consistent for the same input
