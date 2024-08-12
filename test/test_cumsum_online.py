import numpy as np
import pandas as pd
import pytest

from breakout.cumsum_online import cumsum_breakout_detection_online


def stream_data(data, detector):
    # Process new data points as they arrive

    for idx, new_value in enumerate(data):
        is_breakout, cumsum_value = detector.update(new_value)
    return detector.get_results()


def test_cumsum_breakout_detection_online_numpy_array():
    data = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
    detector = cumsum_breakout_detection_online(data=None, threshold=2.0, online=True)
    breakouts, cumsum_values = stream_data(data, detector)
    assert isinstance(breakouts, list)
    assert isinstance(cumsum_values, list)
    assert len(breakouts) > 0
    assert len(cumsum_values) == len(data)


def test_cumsum_breakout_detection_online_pandas_series():
    data = pd.Series([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
    detector = cumsum_breakout_detection_online(data=None, threshold=2.0, online=True)
    breakouts, cumsum_values = stream_data(data, detector)
    assert isinstance(breakouts, list)
    assert isinstance(cumsum_values, list)
    assert len(breakouts) > 0
    assert len(cumsum_values) == len(data)


def test_cumsum_breakout_detection_online_invalid_input():
    with pytest.raises(ValueError):
        cumsum_breakout_detection_online(
            [1, 2, 3, 4, 5]
        )  # List input should raise ValueError


def test_cumsum_breakout_detection_online_missing_column():
    data = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        cumsum_breakout_detection_online(
            data
        )  # Missing column name should raise ValueError


def test_cumsum_breakout_detection_online_invalid_column():
    data = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        cumsum_breakout_detection_online(data, column="invalid_column")


def test_cumsum_breakout_detection_online_target_methods():
    data = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])

    # Test fixed target method
    breakouts_fixed, _ = cumsum_breakout_detection_online(
        data, threshold=2.0, target_method="fixed"
    )

    # Test moving target method
    breakouts_moving, _ = cumsum_breakout_detection_online(
        data, threshold=2.0, target_method="moving", target_params={"window": 3}
    )

    # Test exponential target method
    breakouts_exp, _ = cumsum_breakout_detection_online(
        data, threshold=2.0, target_method="exponential", target_params={"alpha": 0.1}
    )

    assert len(breakouts_fixed) > 0
    assert len(breakouts_moving) > 0
    assert len(breakouts_exp) > 0
    assert breakouts_fixed != breakouts_moving != breakouts_exp


def test_cumsum_breakout_detection_online_invalid_target_method():
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        cumsum_breakout_detection_online(data, target_method="invalid")


def test_cumsum_breakout_detection_online_no_breakouts():
    data = np.array([1, 2, 3, 4, 5])
    breakouts, cumsum_values = cumsum_breakout_detection_online(data, threshold=10.0)
    assert len(breakouts) == 0
    assert len(cumsum_values) == len(data)


def test_cumsum_breakout_detection_online_drift():
    data = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
    breakouts_no_drift, _ = cumsum_breakout_detection_online(
        data, threshold=2.0, drift=0.0
    )
    breakouts_with_drift, _ = cumsum_breakout_detection_online(
        data, threshold=2.0, drift=1.0
    )
    assert len(breakouts_no_drift) >= len(breakouts_with_drift)


# Add more tests as needed
