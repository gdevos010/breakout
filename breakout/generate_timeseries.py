import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def generate_timeseries_with_breakouts(
    n_points: int = 1000,
    n_breakouts: int = 3,
    noise_level: float = 0.1,
    trend: float = 0.01,
    seasonality: bool = True,
    seasonal_period: float = 365.25,
    seasonal_amplitude: float = 0.5,
    breakout_min: float = 0.5,
    breakout_max: float = 2.0,
    start_date: str = "2020-01-01",
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Generate a time series with specified breakout points.

    Parameters:
    - n_points (int): Total number of points in the time series
    - n_breakouts (int): Number of breakout points to introduce
    - noise_level (float): Standard deviation of the noise
    - trend (float): Slope of the overall trend
    - seasonality (bool): Whether to include a seasonal component
    - seasonal_period (float): Period of the seasonal component in days
    - seasonal_amplitude (float): Amplitude of the seasonal component
    - breakout_min (float): Minimum magnitude of breakouts
    - breakout_max (float): Maximum magnitude of breakouts
    - start_date (str): Start date for the time series in 'YYYY-MM-DD' format
    - random_seed (int, optional): Seed for random number generation

    Returns:
    - pd.DataFrame: A dataframe with 'date' and 'value' columns
    - List[int]: Indices of breakout points
    """

    # Input validation
    if (
        n_points <= 0
        or n_breakouts < 0
        or noise_level < 0
        or breakout_min < 0
        or breakout_max < breakout_min
    ):
        raise ValueError("Invalid input parameters")

    if random_seed is not None:
        np.random.seed(random_seed)

    log.info("Generating time series with breakouts")

    # Generate base time series with trend
    x = np.arange(n_points)
    y = trend * x + np.random.normal(0, noise_level, n_points)

    # Add seasonality if requested
    if seasonality:
        season = seasonal_amplitude * np.sin(2 * np.pi * x / seasonal_period)
        y += season
        log.info(
            f"Added seasonal component with period {seasonal_period} and amplitude {seasonal_amplitude}"
        )

    # Introduce breakout points
    breakout_indices = np.sort(np.random.choice(n_points, n_breakouts, replace=False))
    for idx in breakout_indices:
        # Randomly choose direction and magnitude of breakout
        breakout = np.random.choice([-1, 1]) * np.random.uniform(
            breakout_min, breakout_max
        )
        y[idx:] += breakout

    log.info(f"Introduced {n_breakouts} breakouts at indices: {breakout_indices}")

    # Create DataFrame
    try:
        dates = pd.date_range(start=start_date, periods=n_points)
    except ValueError as e:
        log.error(f"Invalid start date: {start_date}")
        raise ValueError(f"Invalid start date: {start_date}") from e

    df = pd.DataFrame({"date": dates, "value": y})

    return df, breakout_indices.tolist()
