# Breakout Detection Library

## Overview

This library provides a collection of algorithms for detecting breakouts (sudden changes or anomalies) in time series data. It includes implementations of several popular breakout detection methods, making it useful for various applications in data analysis, finance, and signal processing.

## Features

- ARIMA-based breakout detection
- CUMSUM (Cumulative Sum) breakout detection
- Isolation Forest anomaly detection
- Moving Average (MA) and Exponential Moving Average (EMA) breakout detection
- PELT (Pruned Exact Linear Time) breakout detection
- Seasonal Decomposition breakout detection
- Wavelet Transform breakout detection
- Windowed Variance breakout detection

## Installation

To install the library, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/breakout-detection-library.git
cd breakout-detection-library
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the library:

```python
import pandas as pd
from breakout import arima, cumsum, ma

# Load your time series data
data = pd.read_csv('your_data.csv')

# Perform ARIMA-based breakout detection
arima_breakouts = arima.arima_breakout_detection(data, value_column='value', date_column='date')

# Perform CUMSUM breakout detection
cumsum_breakouts, cumsum_values = cumsum.cumsum_breakout_detection(data, column='value')

# Perform Moving Average breakout detection
ma_breakouts = ma.ma_breakout_detection(data, value_column='value', date_column='date')

print("ARIMA breakouts:", arima_breakouts)
print("CUMSUM breakouts:", cumsum_breakouts)
print("MA breakouts:", ma_breakouts)
```

For more detailed usage examples, please refer to the `breakout.py` file in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## TODO

- Implement Bayesian changepoint detection

## Contact

For any questions or feedback, please open an issue on the GitHub repository.