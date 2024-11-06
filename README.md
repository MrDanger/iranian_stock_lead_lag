
# Iranian Stock Lead-Lag Analysis

This repository provides tools for identifying and analyzing lead-lag relationships in the Iranian stock market, using mathematical and computational approaches. By leveraging concepts from network theory and optimization, this project aims to determine the directional relationships among stocks, which can help in forecasting price movements and developing trading strategies.

## Features

1. **Lead-Lag Relationship Detection**: Utilizes various mathematical tools to identify stocks that lead or lag others based on time-series data.
2. **Network Analysis**: Employs `networkx` to visualize and analyze stock relationships, constructing a directed graph where edges represent lead-lag connections.
3. **Optimization and Modeling**: Uses `pyomo` for solving optimization problems related to stock relationships and lagging minimization, ensuring efficient computational performance.
4. **Visualization**: Integrates `matplotlib` for clear, detailed plots of stock behaviors and network structures, with additional support for bidirectional Arabic text rendering.
5. **Statistical Computations**: With `scipy`, the project applies various statistical tests and analyses, providing insights into stock price correlations and their significance.

## Mathematical Concepts

### 1. Time-Series Analysis
   - Utilizes data-driven techniques for analyzing stock price series over time. Methods include cross-correlation and autocorrelation for identifying temporal dependencies and directional influence.

### 2. Network Theory
   - Constructs a directed graph to model lead-lag relationships. Each stock is represented as a node, and directed edges indicate causative or predictive relationships, which are inferred through correlation and causality tests.

### 3. Optimization Techniques
   - Implements mathematical optimization to solve lead-lag minimization problems, leveraging `pyomo` for linear and nonlinear programming. This helps in determining the optimal structure for lead-lag networks.

### 4. Signal Processing
   - Signal processing tools are employed for smoothing and filtering stock price data, enhancing the detection of subtle lead-lag relationships within the noise of market fluctuations.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:MrDanger/iranian_stock_lead_lag.git
   cd iranian_stock_lead_lag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run Analysis**: Use `main.py` to start the analysis. This script fetches stock data, applies the mathematical models, and outputs a lead-lag relationship graph.
   ```bash
   python main.py
   ```

2. **Visualize Results**: The output includes network visualizations of lead-lag relationships, with customizable parameters for enhanced readability.

3. **Font Customization**: To properly render Farsi (Persian) labels, `Vazirmatn-Regular.ttf` is included for compatibility with bidirectional and reshaped text.

## Dependencies

- **pytse-client**: Interface for accessing Tehran Stock Exchange (TSE) data.
- **scipy**: Provides essential tools for statistical and mathematical analysis.
- **networkx**: Used for building and analyzing stock relationship networks.
- **matplotlib**: Visualization library for plotting lead-lag graphs and data distributions.
- **arabic-reshaper & python-bidi**: Libraries for handling bidirectional Farsi text in plots.
- **pyomo**: Mathematical optimization library used for building models of stock relationships.

## License

This project is licensed under the MIT License.
