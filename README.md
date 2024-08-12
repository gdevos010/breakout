# Breakout Detection Library

This library provides various methods for detecting breakouts in time series data. It includes implementations of several algorithms and utilities for generating test data, parameter tuning, and visualization.

## Features

- Multiple breakout detection algorithms:
  - Adaptive Threshold
  - ARIMA-based
  - CUMSUM (Cumulative Sum)
  - Isolation Forest
  - Moving Average (MA and EMA)
  - PELT (Pruned Exact Linear Time)
  - Seasonal Decomposition
  - Wavelet Transform
  - Windowed Variance
- Automatic parameter tuning
- Time series generation with controlled breakouts
- Visualization utilities

## Installation

This project uses Poetry for dependency management. To install the library, follow these steps:

1. Make sure you have Poetry installed. If not, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/breakout-detection.git
   cd breakout-detection
   ```

3. Install the dependencies using Poetry:

   ```bash
   poetry install
   ```

This will create a virtual environment and install all the necessary dependencies.

## Usage

Then you can use the library in your Python scripts. Here's a basic example:

```python
from breakout import cumsum_breakout_detection_online
from breakout.visualization import plot_timeseries_with_breakouts
from breakout.generate_timeseries import generate_timeseries_with_breakouts

# Generate a time series with breakouts
ts, true_breakouts = generate_timeseries_with_breakouts(
    n_points=1000, n_breakouts=3, noise_level=0.1, trend=0.01, seasonality=True
)

# Detect breakouts using CUMSUM
detected_breakouts, _ = cumsum_breakout_detection_online(
    ts,
    column="value",
    threshold=3.0,
    drift=0.05,
    target_method="exponential",
    target_params={"alpha": 0.1},
)

# Visualize the results
plot_timeseries_with_breakouts(ts, detected_breakouts, "cumsum_breakouts")
```

### CUMSUM Online Example

Here's an example of using the CUMSUM online breakout detection:

```python
import numpy as np
from breakout import cumsum_breakout_detection_online
from breakout.generate_timeseries import generate_timeseries_with_breakouts

# Generate a time series with breakouts
ts, true_breakouts = generate_timeseries_with_breakouts(
    n_points=1000, n_breakouts=3, noise_level=0.1, trend=0.01, seasonality=True
)

# Initialize the CUMSUM online detector
detector = cumsum_breakout_detection_online(
    data=None,
    threshold=3.0,
    drift=0.05,
    target_method="exponential",
    target_params={"alpha": 0.1},
    online=True
)

# Process the time series data point by point
online_breakouts = []
for idx, value in enumerate(ts['value']):
    is_breakout, cumsum_value = detector.update(value)
    if is_breakout:
        online_breakouts.append(idx)

# Get the final results
final_breakouts, cumsum_values = detector.get_results()

print(f"True breakouts: {true_breakouts}")
print(f"Detected breakouts: {final_breakouts}")

# Calculate detection accuracy
correct_detections = sum(1 for b in final_breakouts if min(abs(np.array(true_breakouts) - b)) <= 5)
precision = correct_detections / len(final_breakouts) if final_breakouts else 0
recall = correct_detections / len(true_breakouts) if true_breakouts else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
```

This example demonstrates how to use the CUMSUM online detector to process a time series point by point, which is particularly useful for real-time or streaming data applications.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## TODO

- Implement Bayesian changepoint detection

## License

This project is licensed under the MIT License.
