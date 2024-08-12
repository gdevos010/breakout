import logging

from rich.console import Console
from rich.logging import RichHandler

from breakout.adaptive_threshold import adaptive_threshold_breakout
from breakout.arima import arima_breakout_detection
from breakout.cumsum_online import cumsum_breakout_detection_online
from breakout.generate_timeseries import generate_timeseries_with_breakouts
from breakout.isolated_forest import isolation_forest_breakout
from breakout.ma import ma_breakout_detection
from breakout.parameter_tuning import auto_tune_all
from breakout.pelt import pelt_breakout_detection
from breakout.seasonal_decomposition import (
    plot_seasonal_decomposition_breakout,
    seasonal_decomposition_breakout,
)
from breakout.visualization import plot_timeseries_with_breakouts
from breakout.wavelet import wavelet_breakout_detection
from breakout.windowed_variance import windowed_variance_breakout

# Set up logging with rich
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger(__name__)
console = Console()


def main() -> None:
    # Example usage
    log.info("Starting breakout detection example")

    ts, breakouts = generate_timeseries_with_breakouts(
        n_points=1000, n_breakouts=3, noise_level=0.1, trend=0.01, seasonality=True
    )
    ts_np = ts["value"].values
    plot_timeseries_with_breakouts(ts, breakouts, "truth")
    log.info(f"Generated time series shape: {ts.shape}")
    log.info(f"Truth breakout indices: {breakouts}")

    # CUMSUM
    log.info("Performing CUMSUM breakout detection")
    detected_breakouts, cumsum_values = cumsum_breakout_detection_online(
        ts,
        column="value",
        threshold=3.0,
        drift=0.05,
        target_method="exponential",
        target_params={"alpha": 0.1},
    )
    plot_timeseries_with_breakouts(ts, detected_breakouts, "cumsum")
    log.info(f"CUMSUM detected breakout indices: {detected_breakouts}")

    # CUMSUM online
    detector = cumsum_breakout_detection_online(
        data=None, threshold=3.0, drift=0.05, online=True
    )

    # Process new data points as they arrive
    online_breakouts = []
    for idx, new_value in enumerate(ts_np):
        is_breakout, cumsum_value = detector.update(new_value)
        if is_breakout:
            online_breakouts.append(idx)

    log.info(f"CUMSUM online   breakout indices: {online_breakouts}")
    # Get the final results
    online_breakouts2, cumsum_values = detector.get_results()
    log.info(f"CUMSUM detected breakout indices: {online_breakouts2}")

    # Moving Average
    log.info("Performing Moving Average breakout detection")
    ma_detected = ma_breakout_detection(
        ts,
        value_column="value",
        date_column="date",
        window_size=20,
        threshold=2.5,
        method="MA",
    )
    plot_timeseries_with_breakouts(ts, ma_detected, "ma_breakout")
    log.info(f"MA detected breakout indices: {ma_detected}")

    # Exponential Moving Average
    log.info("Performing Exponential Moving Average breakout detection")
    ema_detected = ma_breakout_detection(
        ts,
        value_column="value",
        date_column="date",
        window_size=20,
        threshold=2.5,
        method="EMA",
    )
    plot_timeseries_with_breakouts(ts, ema_detected, "ema_breakout")
    log.info(f"EMA detected breakout indices: {ema_detected}")

    # PELT
    log.info("Performing PELT breakout detection")
    detected_breakpoints, cost_values = pelt_breakout_detection(
        ts, column="value", penalty=1.0
    )
    log.info(f"PELT detected breakpoint indices: {detected_breakpoints}")
    log.info(f"Cost function values at breakpoints: {cost_values}")
    plot_timeseries_with_breakouts(ts, detected_breakpoints, "pelt")

    # ARIMA
    log.info("Performing ARIMA breakout detection")
    detected_breakouts_df = arima_breakout_detection(
        ts, value_column="value", date_column="date", order=(1, 1, 1), threshold=3.0
    )
    log.info(f"ARIMA detected breakout indices: {detected_breakouts_df}")
    plot_timeseries_with_breakouts(ts, detected_breakouts_df, "ARIMA")

    # Seasonal Decomposition
    log.info("Performing Seasonal Decomposition breakout detection")
    detected_breakouts_df, decomposed_data_df = seasonal_decomposition_breakout(
        ts, value_column="value", date_column="date", period=365, threshold=2.5
    )
    log.info(
        f"Seasonal Decomposition detected breakout indices: {detected_breakouts_df}"
    )
    plot_seasonal_decomposition_breakout(decomposed_data_df)
    # log.info("Seasonal Decomposition data head:")
    # console.print(decomposed_data_df.head())

    # Windowed Variance
    log.info("Performing Windowed Variance breakout detection")
    detected_breakouts_df, variance_ratios_df = windowed_variance_breakout(
        ts, column="value", window_size=50, overlap=25, threshold=2.5
    )
    detected_breakouts_np, variance_ratios_np = windowed_variance_breakout(
        ts_np, window_size=50, overlap=25, threshold=2.5
    )
    log.info(
        f"Windowed Variance detected breakouts (DataFrame): {detected_breakouts_df}"
    )
    log.info(f"Windowed Variance detected breakouts (NumPy): {detected_breakouts_np}")
    plot_timeseries_with_breakouts(ts, detected_breakouts_df, "windowed_variance")

    # Wavelet
    log.info("Performing Wavelet breakout detection")
    detected_breakouts_df = wavelet_breakout_detection(
        ts, column="value", threshold=3.5
    )
    log.info(f"Wavelet detected breakouts (DataFrame): {detected_breakouts_df}")
    detected_breakouts_np = wavelet_breakout_detection(ts_np, threshold=3.5)
    log.info(f"Wavelet detected breakouts (NumPy array): {detected_breakouts_np}")

    df_breakouts, df_scores = isolation_forest_breakout(
        ts, column="value", contamination=0.01
    )
    log.info(f"Detected isolation breakouts (DataFrame): {df_breakouts}")
    plot_timeseries_with_breakouts(ts, df_breakouts, "isolation_forest")
    np_breakouts, np_scores = isolation_forest_breakout(ts_np, contamination=0.01)
    log.info(f"Detected isolation breakouts (NumPy): {np_breakouts}")

    detected_breakouts, thresholds = adaptive_threshold_breakout(
        ts, column="value", window_size=50, n_sigmas=3.0, overlap=25
    )
    log.info(f"Detected adaptive threshold breakouts (DataFrame): {detected_breakouts}")
    plot_timeseries_with_breakouts(ts, detected_breakouts, "adaptive_threshold")
    np_breakouts, np_scores = adaptive_threshold_breakout(
        ts_np, window_size=50, n_sigmas=3.0, overlap=25
    )
    log.info(f"Detected isolation breakouts (NumPy): {np_breakouts}")


def tuning_example() -> None:
    ts1, breakouts1 = generate_timeseries_with_breakouts(
        n_points=1000, n_breakouts=3, noise_level=0.1, trend=0.01, seasonality=True
    )
    ts1_np = ts1["value"].values
    log.info(f"Truth breakout indices: {breakouts1}")

    ts2, breakouts2 = generate_timeseries_with_breakouts(
        n_points=1000, n_breakouts=5, noise_level=0.2, trend=0.01, seasonality=True
    )
    ts2_np = ts2["value"].values

    ts_list = [ts1_np, ts2_np]
    breakouts_list = [breakouts1, breakouts2]

    # Tune parameters for all methods
    optimal_params = auto_tune_all(ts_list, breakouts_list)

    # Print optimal parameters
    for method, params in optimal_params.items():
        log.info(f"Optimal parameters for {method}:")
        log.info(params)

    detected_breakouts, _ = cumsum_breakout_detection_online(ts1_np)
    log.info(f"CUMSUM detected breakouts (NumPy array): {detected_breakouts}")
    detected_breakouts, _ = cumsum_breakout_detection_online(ts1_np, **optimal_params["cumsum"])
    plot_timeseries_with_breakouts(ts1, detected_breakouts, "cumsum")
    log.info(f"CUMSUM detected breakouts (NumPy array): {detected_breakouts}")
    plot_timeseries_with_breakouts(ts1, detected_breakouts, "cumsum_tuned")


if __name__ == "__main__":
    main()
    tuning_example()
