# Equity Valuation Dashboard: DCF, Black–Scholes & ML Forecasting

## Overview

This project is an interactive Dash web application that applies several core models from quantitative finance to equity valuation and risk analysis. The app combines discounted cash flow (DCF) valuation, Black–Scholes option pricing, Monte Carlo simulation, and machine-learning forecasts into a single dashboard driven by live market data. Users can input a ticker, adjust key assumptions (growth rates, discount rates, volatility, forecast horizon, strike price, etc.), and immediately see how intrinsic value, option prices, and simulated price paths respond.

The project showcases how Python, data science, and financial theory can be integrated into a practical decision-support tool. It emphasizes end-to-end workflow: from pulling raw data with `yfinance`, to cleaning and transforming it with Pandas, to computing valuation metrics and visualizing results with Plotly and Dash.

> ⚠️ **Disclaimer:** This application is for educational and research purposes only and does **not** constitute financial advice or a recommendation to buy or sell any security.

---

## 1. Theoretical Background

The dashboard is built around four main components of modern finance.

### 1.1 Discounted Cash Flow (DCF)

DCF valuation estimates a firm’s intrinsic value by projecting future free cash flows and discounting them back to the present at a rate that reflects risk (often a weighted average cost of capital, WACC).

If $FCF_t$ is the forecast free cash flow in year $t$ and $r$ is the discount rate, the present value of cash flows over $T$ years is

$$
PV_{\text{FCF}} = \sum_{t=1}^{T} \frac{FCF_t}{(1 + r)^t}.
$$

A terminal value captures cash flows beyond the explicit forecast horizon, typically using a constant-growth perpetuity:

$$
TV = \frac{FCF_{T+1}}{(r - g)},
$$

where $g$ is the long-run growth rate. Enterprise value and equity value per share are derived from these components.

### 1.2 Black–Scholes Option Pricing

The Black–Scholes model provides a closed-form solution for pricing European call and put options under standard assumptions such as log-normal returns and constant volatility. For a stock price $S$, strike $K$, risk-free rate $r$, dividend yield $q$, volatility $\sigma$, and time to maturity $T$, the call and put prices are

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \quad
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),
$$

with

$$
d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}.
$$

Here $N(\cdot)$ is the cumulative distribution function of the standard normal distribution. These formulas link model inputs (price, volatility, time, rates) to option values and sensitivities.

### 1.3 Monte Carlo Simulation (GBM)

The stock price dynamics are modeled as geometric Brownian motion (GBM),

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t,
$$

which leads to the simulated terminal price

$$
S_T = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z\right],
$$

where $Z \sim N(0,1)$. By simulating many such paths, the dashboard visualizes the distribution of potential future prices and option payoffs.

### 1.4 Machine-Learning Forecasting

For the ML component, historical price data are transformed into features such as lagged log-returns, rolling volatility, and moving averages. A regression model (e.g., `RandomForestRegressor`) is trained to predict the future price $S_{t+h}$ for a user-selected horizon $h$. Model accuracy is evaluated using metrics like RMSE and MAPE and compared to a naïve benchmark (for example, “price stays constant”).

---

## 2. Application Features

The Equity Valuation Dashboard provides a highly interactive experience focused on both intuition and quantitative analysis:

- **Ticker-driven workflow**
  - User enters a stock ticker (e.g., `AAPL`, `MSFT`, `TSLA`) and runs the analysis.
  - Historical OHLCV data, dividends, and basic company fundamentals are fetched automatically via `yfinance`.

- **DCF Valuation Module**
  - Computes intrinsic value per share using projected FCFs, user-controlled WACC, forecast horizon, and terminal growth.
  - Displays both the forecasted cash flow stream and the present value of each year’s cash flow.
  - Shows key KPIs such as enterprise value, equity value, and upside/downside relative to the current market price.

- **Black–Scholes Option Pricer**
  - Calculates theoretical call and put prices given $S$, $K$, $T$, $r$, $q$, and $\sigma$.
  - Volatility can be estimated from historical returns or overridden manually.
  - Includes an option payoff chart at expiry to visualize profit/loss for different terminal stock prices.

- **Monte Carlo Price Simulation**
  - Simulates many future price paths using GBM with user-specified drift, volatility, and horizon.
  - Produces a histogram of simulated terminal prices and overlays the current price for comparison.
  - Can be used to explore risk, probability of breaching certain levels, and option payoff distributions.

- **Machine-Learning Forecast Panel**
  - Trains a model on historical data to predict prices $h$ days ahead (with an adjustable horizon).
  - Plots predicted vs. actual prices and reports error metrics.
  - Provides a quantitative comparison of ML forecasts against simple baselines.

- **TradingView-style Charting**
  - Interactive Plotly price chart with range sliders/selectors.
  - Optional indicators: simple/exp moving averages, Bollinger Bands, RSI, volume, and more.
  - Helps connect valuation outputs with recent market behavior.

This combination of tools makes the app useful both as a learning environment for students and as a prototype analysis dashboard for practitioners.

---

## 3. Technical Implementation

The project is implemented entirely in Python and builds on several core libraries:

- **Dash** for the web framework and reactive UI components.
- **Plotly** for interactive charts and visualizations (price series, DCF bars, Monte Carlo histograms, payoff diagrams).
- **yfinance** to pull historical prices, dividends, and fundamental data directly from Yahoo Finance.
- **Pandas & NumPy** for data wrangling, time-series handling, and numerical computation.
- **SciPy / math** for statistical and mathematical utilities (e.g., normal CDF for Black–Scholes).
- **scikit-learn** for machine-learning models and evaluation metrics.

The app is organized so that financial logic (DCF, option pricing, simulations, feature engineering) lives in reusable helper functions, while Dash callbacks are responsible for wiring user inputs to outputs. This separation makes it easier to test, extend, and reuse the computational parts in other projects.

---

## 4. Key Learning Outcomes

Working with this dashboard supports several learning goals:

- **Connecting theory and practice**  
  See how textbook formulas like DCF and Black–Scholes behave with real market data and realistic inputs.

- **Understanding model sensitivity**  
  Experiment with volatility, discount rates, growth assumptions, and forecast horizons to observe how valuations and risk metrics respond.

- **Building end-to-end data applications**  
  Gain experience turning raw market data into a polished, interactive web app using Dash, Plotly, and Python.

- **Exploring machine learning in finance**  
  Learn how basic ML models can be applied to financial time series and how to interpret forecast performance critically.

Overall, the project serves as a portfolio-ready example of quantitative finance, data science, and Python engineering, relevant for roles in quantitative research, risk management, and investment analytics.

---

## 5. Author

**Author:** *Nail Mammadzada*  
*University of Arizona , Statistics & Data Science , Finance Dual Degree Student*
 
- LinkedIn: https://www.linkedin.com/in/nailmammadzada/



---

## 6. Running the Project

To run the Equity Valuation Dashboard locally, open a terminal or command prompt and execute the following commands:

```bash
# Clone the repository
git clone https://github.com/your-github-handle/dash-dcf-blackscholes-ml-dashboard.git
cd dash-dcf-blackscholes-ml-dashboard

# (Optional) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
python App_ML.py
