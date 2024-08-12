from typing import Any, List, Tuple

import numpy as np
from sklearn.model_selection import ParameterGrid

from breakout.adaptive_threshold import adaptive_threshold_breakout
from breakout.arima import arima_breakout_detection
from breakout.cumsum_online import cumsum_breakout_detection_online
from breakout.isolated_forest import isolation_forest_breakout
from breakout.ma import ma_breakout_detection
from breakout.pelt import pelt_breakout_detection
from breakout.seasonal_decomposition import seasonal_decomposition_breakout
from breakout.wavelet import wavelet_breakout_detection
from breakout.windowed_variance import windowed_variance_breakout
import warnings
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def evaluate_breakouts(
    true_breakouts_list: list[list[int]],
    detected_breakouts_list: list[list[int]],
    *,
    tolerance: int = 5,
) -> float:
    """
    Evaluate the performance of breakout detection using average F1 score across multiple time series.

    Args:
    - true_breakouts_list: List of lists containing true breakout indices for each time series
    - detected_breakouts_list: List of lists containing detected breakout indices for each time series
    - tolerance: Number of points to consider as a correct detection

    Returns:
    - Average F1 score across all time series
    """
    f1_scores = []

    for true_breakouts, detected_breakouts in zip(
        true_breakouts_list, detected_breakouts_list
    ):
        true_positives = 0
        matched_detected = set()

        for tb in true_breakouts:
            for i, db in enumerate(detected_breakouts):
                if abs(tb - db) <= tolerance:
                    true_positives += 1
                    matched_detected.add(i)
                    break

        # Remove matched detections and those within tolerance of any true breakout
        unmatched_detected = [
            db for i, db in enumerate(detected_breakouts)
            if i not in matched_detected and
            not any(abs(tb - db) <= tolerance for tb in true_breakouts)
        ]

        false_positives = len(unmatched_detected)
        false_negatives = len(true_breakouts) - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1_score)

    return np.mean(f1_scores)


def auto_tune_adaptive_threshold(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for adaptive threshold breakout detection for multiple time series.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "window_size": [10, 20, 50, 100],
        "n_sigmas": [2.0, 2.5, 3.0, 3.5],
        "overlap": [5, 10, 25],
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="adaptive_threshold"):
        if (
            params["window_size"] <= 0
            or params["overlap"] < 0
            or params["overlap"] >= params["window_size"]
        ):
            continue

        detected_breakouts_list = [
            adaptive_threshold_breakout(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_arima(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for ARIMA breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "order": [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (2, 2, 2)],
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="arima"):
        detected_breakouts_list = [
            arima_breakout_detection(ts, **params) for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_cumsum(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for CUMSUM breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
        "drift": [0.01, 0.05, 0.1],
        "target_method": ["fixed", "moving", "exponential"],
        "target_params": [
            {"alpha": 0.1},
            {"alpha": 0.2},
            {"window": 10},
            {"window": 20},
        ],
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="cumsum"):
        detected_breakouts_list = [
            cumsum_breakout_detection_online(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_isolation_forest(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for Isolation Forest breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "contamination": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "max_samples": ["auto", 100, 200],
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="isolation_forest"):
        detected_breakouts_list = [
            isolation_forest_breakout(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_ma(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for Moving Average breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "window_size": [10, 20, 50],
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
        "method": ["MA", "EMA"],
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="ma"):
        detected_breakouts_list = [
            ma_breakout_detection(ts, **params) for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_pelt(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for PELT breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "model": ["l1", "l2", "rbf"],
        "min_size": [2, 5, 10],
        "jump": [1, 5, 10],
        "penalty": [0.5, 1.0, 2.0],
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), "pelt"):
        detected_breakouts_list = [
            pelt_breakout_detection(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_seasonal_decomposition(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for Seasonal Decomposition breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "period": [7, 30, 365],  # Assuming daily data, adjust as needed
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="seasonal_decomposition"):
        detected_breakouts_list = [
            seasonal_decomposition_breakout(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_wavelet(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for Wavelet breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "wavelet": ["db4", "sym4", "coif4"],
        "level": [2, 3, 4, 5, 6],
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="wavelet"):
        detected_breakouts_list = [
            wavelet_breakout_detection(ts, **params) for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_windowed_variance(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, Any]:
    """
    Automatically tune parameters for Windowed Variance breakout detection.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters
    """
    param_grid = {
        "window_size": [10, 20, 50],
        "overlap": [5, 10, 25],
        "threshold": np.arange(0.5, 10.5, 0.5).tolist(),
    }

    best_score = 0.0
    best_params = {}

    for params in tqdm(ParameterGrid(param_grid), desc="windowed_variance"):
        if (
            params["window_size"] <= 0
            or params["overlap"] < 0
            or params["overlap"] >= params["window_size"]
        ):
            continue
        detected_breakouts_list = [
            windowed_variance_breakout(ts, **params)[0] for ts in ts_list
        ]
        score = evaluate_breakouts(true_breakouts_list, detected_breakouts_list)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def auto_tune_all(
    ts_list: list[np.ndarray], true_breakouts_list: list[list[int]]
) -> dict[str, dict[str, Any]]:
    """
    Automatically tune parameters for all breakout detection methods.

    Args:
    - ts_list: List of time series data
    - true_breakouts_list: List of lists containing true breakout indices for each time series

    Returns:
    - Dictionary of optimal parameters for each method
    """
    return {
        "adaptive_threshold": auto_tune_adaptive_threshold(
            ts_list, true_breakouts_list
        ),
        "arima": auto_tune_arima(ts_list, true_breakouts_list),
        "cumsum": auto_tune_cumsum(ts_list, true_breakouts_list),
        "isolation_forest": auto_tune_isolation_forest(ts_list, true_breakouts_list),
        "ma": auto_tune_ma(ts_list, true_breakouts_list),
        "pelt": auto_tune_pelt(ts_list, true_breakouts_list),
        "seasonal_decomposition": auto_tune_seasonal_decomposition(
            ts_list, true_breakouts_list
        ),
        "wavelet": auto_tune_wavelet(ts_list, true_breakouts_list),
        "windowed_variance": auto_tune_windowed_variance(ts_list, true_breakouts_list),
    }
