import numpy as np
import pandas as pd
import pytest

from breakout import isolation_forest_breakout


@pytest.fixture()
def sample_data():
    return np.array([1, 2, 3, 10, 4, 5, 6, 100, 7, 8, 9])


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {
            "values": [1, 2, 3, 10, 4, 5, 6, 100, 7, 8, 9],
            "dates": pd.date_range(start="2023-01-01", periods=11),
        }
    )


def test_numpy_input(sample_data):
    breakouts, y_pred = isolation_forest_breakout(sample_data)
    assert isinstance(breakouts, list)
    assert isinstance(y_pred, np.ndarray)
    assert len(breakouts) > 0
    assert len(y_pred) == len(sample_data)


def test_dataframe_input(sample_df):
    breakouts, y_pred = isolation_forest_breakout(sample_df, column="values")
    assert isinstance(breakouts, list)
    assert isinstance(y_pred, np.ndarray)
    assert len(breakouts) > 0
    assert len(y_pred) == len(sample_df)


def test_contamination():
    data = np.random.rand(100)
    contamination = 0.05
    breakouts, _ = isolation_forest_breakout(data, contamination=contamination)
    assert len(breakouts) == int(contamination * len(data))


def test_random_state():
    data = np.random.rand(100)
    breakouts1, _ = isolation_forest_breakout(data, random_state=42)
    breakouts2, _ = isolation_forest_breakout(data, random_state=42)
    assert breakouts1 == breakouts2


def test_n_estimators():
    data = np.random.rand(100)
    _, y_pred = isolation_forest_breakout(data, n_estimators=50)
    assert len(np.unique(y_pred)) == 2  # Should only have -1 and 1


def test_max_samples():
    data = np.random.rand(100)
    breakouts, _ = isolation_forest_breakout(data, max_samples=50)
    assert isinstance(breakouts, list)


def test_invalid_dataframe_input():
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Column name must be specified when input is a DataFrame"
    ):
        isolation_forest_breakout(df)


def test_invalid_column_name():
    df = pd.DataFrame({"values": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Column 'wrong_column' not found in DataFrame"
    ):
        isolation_forest_breakout(df, column="wrong_column")


def test_invalid_numpy_input():
    data = np.random.rand(10, 2)  # 2D array
    with pytest.raises(ValueError, match="NumPy array input must be 1-dimensional"):
        isolation_forest_breakout(data)


def test_invalid_input_type():
    data = [1, 2, 3]  # List instead of np.array or pd.DataFrame
    with pytest.raises(
        ValueError, match="Input data must be a pandas DataFrame or NumPy array"
    ):
        isolation_forest_breakout(data)


def test_empty_input():
    data = np.array([])
    with pytest.raises(ValueError):  # Expecting ValueError from sklearn
        isolation_forest_breakout(data)


def test_all_same_values():
    data = np.ones(100)
    breakouts, y_pred = isolation_forest_breakout(data)
    assert len(breakouts) == 0
    assert np.all(y_pred == 1)
