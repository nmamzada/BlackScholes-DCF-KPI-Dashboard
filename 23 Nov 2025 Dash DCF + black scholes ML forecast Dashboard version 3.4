"""
Dash DCF + Black-Scholes KPI Dashboard with ML Forecast

Features
- User enters a stock ticker (e.g., AAPL, MSFT, TSLA) and clicks "Run Analysis".
- Pulls market data & financials via yfinance.
- Computes:
  • DCF (based on historical Free Cash Flow with user-controlled assumptions)
  • Black-Scholes (call/put prices using estimated or user-overridden inputs)
  • KPIs: Current Price, DCF Value/Share, Upside %, Annualized Volatility, BS Call/Put
- Visuals:
  • Price history (TradingView-style chart)
  • Forecast FCF & discounted FCF (DCF)
  • Monte-Carlo distribution of future price (GBM/Black-Scholes dynamics)
  • Option payoff at expiry
  • ML price forecast vs actuals (RandomForestRegressor, log-return model)
    + a true future forecast point (last date + h days)

Notes
- Internet is required at runtime for live data.
- DCF uses Free Cash Flow (FCF). yfinance’s 'FreeCashFlow' is generally Free Cash Flow to Equity (FCFE).
- If FCF is missing, the app estimates FCF = CFO - CapEx when available.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from plotly.subplots import make_subplots

import dash
from dash import Dash, dcc, html, Input, Output, State, no_update

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# External data / ML libs
# -------------------------------

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise SystemExit("Please install yfinance: pip install yfinance")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------------
# Indicator helpers (TradingView-style)
# -------------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = sma(close, n)
    sd = close.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def vwap(df: pd.DataFrame) -> pd.Series:
    # df must have High, Low, Close, Volume
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# -------------------------------
# Utility models & math helpers
# -------------------------------


@dataclass
class DCFResult:
    per_share_value: Optional[float]
    enterprise_value: Optional[float]
    equity_value: Optional[float]
    pv_fcfs: Optional[float]
    pv_terminal: Optional[float]
    fcf_forecast: pd.DataFrame
    message: str = ""


def ann_vol_from_history(prices: pd.Series, days: int = 252) -> Optional[float]:
    if prices is None or len(prices) < 30:
        return None
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return None
    daily_std = returns.std()
    return float(daily_std * math.sqrt(252))


def black_scholes(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> Tuple[float, float]:
    """Return (call, put) using Black-Scholes with continuous dividend yield q."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return (np.nan, np.nan)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return float(call), float(put)


def gbm_terminal_prices(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n: int = 10000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    drift = (r - q - 0.5 * sigma**2) * T
    diff = sigma * math.sqrt(T) * z
    ST = S0 * np.exp(drift + diff)
    return ST


def safe_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100 * x:.2f}%"


# -------------------------------
# Auto-parameter helpers
# -------------------------------


def auto_rf() -> float:
    # Simple RF approximation using 10y yield via yfinance ^TNX if available.
    # Fallback to 3% if not.
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
        if not tnx.empty:
            return float(tnx.iloc[-1] / 100.0)
    except Exception:
        pass
    return 0.03


def auto_wacc_from_beta(beta: Optional[float], rf: float) -> float:
    # Simple CAPM-based cost of equity: rf + beta * ERP, with ERP ~ 5.5%
    erp = 0.055
    if beta is None or not np.isfinite(beta):
        beta = 1.0
    wacc = rf + beta * erp
    return float(np.clip(wacc, 0.04, 0.14))


# -------------------------------
# yf client wrapper
# -------------------------------


class YFClient:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        self.t = yf.Ticker(self.ticker)

    def price_history(self, period: str = "5y") -> pd.Series:
        hist = self.t.history(period=period)
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        return hist["Close"].dropna()

    def ohlcv_history(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        df = self.t.history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            ).astype(float)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].dropna()

    def last_price(self) -> Optional[float]:
        try:
            p = self.t.fast_info.get("last_price")
            if p:
                return float(p)
        except Exception:
            pass
        try:
            hist = self.t.history(period="5d")
            if not hist.empty:
                return float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass
        return None

    def beta(self) -> Optional[float]:
        """Estimate beta vs SPY if possible, else use yf info."""
        try:
            info = self.t.get_info()
            b = info.get("beta")
            if b is not None:
                return float(b)
        except Exception:
            pass

        try:
            asset = self.t.history(period="5y")["Close"].dropna()
            spy = yf.Ticker("SPY").history(period="5y")["Close"].dropna()
            df = pd.DataFrame({"asset": asset, "spy": spy}).dropna()
            if len(df) < 60:
                return None
            ra = np.log(df["asset"] / df["asset"].shift(1)).dropna()
            rs = np.log(df["spy"] / df["spy"].shift(1)).dropna()
            n = min(len(ra), len(rs))
            ra, rs = ra.iloc[-n:], rs.iloc[-n:]
            cov = np.cov(ra, rs)[0, 1]
            var = np.var(rs)
            if var <= 0:
                return None
            return float(cov / var)
        except Exception:
            return None

    def dividend_yield(self) -> float:
        dy = None
        try:
            info = self.t.get_info()
            for key in ("trailingAnnualDividendYield", "dividendYield"):
                v = info.get(key)
                if v:
                    dy = v
                    break
        except Exception:
            dy = None

        if dy is None:
            try:
                div = self.t.dividends
                if div is not None and not div.empty:
                    last12 = div.iloc[-252:].sum() if len(div) > 252 else div.sum()
                    p = self.last_price()
                    if p and p > 0:
                        dy = float(last12 / p)
            except Exception:
                dy = None
        return float(dy) if dy is not None else 0.0

    def shares_outstanding(self) -> Optional[float]:
        try:
            info = self.t.get_info()
            for key in ("sharesOutstanding", "floatShares"):
                v = info.get(key)
                if v:
                    return float(v)
        except Exception:
            pass
        return None

    def free_cash_flows(self) -> pd.Series:
        """Return yearly FCF series (most recent first). Robust to naming/shape changes."""

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in str(s) if ch.isalnum())

        # Pull cashflow DF
        cf = None
        for attr in ("cashflow", "get_cashflow"):
            try:
                cf = getattr(self.t, attr)
                if callable(cf):
                    cf = cf()
                if cf is not None and not cf.empty:
                    break
            except Exception:
                cf = None
        if cf is None or cf.empty:
            return pd.Series(dtype=float)

        # Normalize index/columns (yfinance sometimes flips orientation)
        if isinstance(cf.index, pd.Index) and not isinstance(
            cf.index, pd.DatetimeIndex
        ):
            idx_norm = {_norm(i): i for i in cf.index}
        else:
            idx_norm = {}
        cf_idx_map = idx_norm

        candidates_fcf = [
            "freecashflow",
            "fcf",
            "freecashflowtoequity",
            "freecashflowtothefirm",
        ]
        candidates_cfo = [
            "totalcashfromoperatingactivities",
            "cashfromoperatingactivities",
            "operatingcashflow",
            "cashfromoperations",
        ]
        candidates_capex = [
            "capitalexpenditures",
            "capitalexpenditure",
            "capex",
            "investmentsincapitalexpenditures",
        ]

        # 1) Direct FCF row
        for key in candidates_fcf:
            if key in cf_idx_map:
                s = cf.loc[cf_idx_map[key]].dropna()
                if not s.empty:
                    s.name = "FCF"
                    return s

        # 2) CFO - CapEx fallback
        cfo_row = None
        for key in candidates_cfo:
            if key in cf_idx_map:
                cfo_row = cf_idx_map[key]
                break
        capex_row = None
        for key in candidates_capex:
            if key in cf_idx_map:
                capex_row = cf_idx_map[key]
                break
        if cfo_row and capex_row:
            s = (cf.loc[cfo_row] - cf.loc[capex_row]).dropna()
            if not s.empty:
                s.name = "FCF"
                return s

        # 3) Orientation fallback: items in columns
        norm_cols = {_norm(c): c for c in cf.columns}
        for key in candidates_fcf:
            if key in norm_cols:
                s = cf[norm_cols[key]].dropna()
                if not s.empty:
                    s.name = "FCF"
                    return s

        return pd.Series(dtype=float)


# -------------------------------
# DCF engine
# -------------------------------


def dcf_valuation(
    fcf_series: pd.Series,
    wacc: float,
    terminal_g: float,
    forecast_years: int,
    fcf_growth: Optional[float],
    shares_out: Optional[float],
) -> DCFResult:
    msg = []
    if fcf_series is None or fcf_series.empty:
        return DCFResult(
            per_share_value=None,
            enterprise_value=None,
            equity_value=None,
            pv_fcfs=None,
            pv_terminal=None,
            fcf_forecast=pd.DataFrame(
                columns=["Year", "FCF_Forecast", "Discount_Factor", "PV_FCF"]
            ),
            message="No FCF history available; DCF not computed.",
        )

    # Use the most recent annual FCF as base
    # sort by date if possible (oldest->newest)
    try:
        idx = pd.to_datetime(fcf_series.index, errors="coerce")
        fcf_series = pd.Series(fcf_series.values, index=idx).sort_index()
    except Exception:
        fcf_series = fcf_series.sort_index()  # chronological
    fcf0 = float(fcf_series.iloc[-1])

    # If no user growth provided, estimate a capped CAGR from history
    if fcf_growth is None:
        try:
            first, last = float(fcf_series.iloc[0]), float(fcf_series.iloc[-1])
            n = max(1, len(fcf_series) - 1)
            cagr = (abs(last) / max(1e-9, abs(first))) ** (1 / n) - 1
            # Cap in a reasonable band [-20%, +25%]
            fcf_growth = float(
                np.clip(cagr * np.sign(np.mean(fcf_series)), -0.20, 0.25)
            )
            msg.append(
                f"Estimated FCF CAGR from history at {fcf_growth*100:.1f}%, capped into [-20%, +25%]."
            )
        except Exception:
            fcf_growth = 0.03
            msg.append("Could not infer FCF growth; using default 3%.")
    else:
        msg.append(f"User-specified FCF growth: {fcf_growth*100:.1f}%.")

    # Forecast and discount
    years = list(range(1, forecast_years + 1))
    fcfs = [fcf0 * ((1 + fcf_growth) ** t) for t in years]
    disc_factors = [(1 / ((1 + wacc) ** t)) for t in years]
    pv_fcfs = float(np.dot(fcfs, disc_factors))

    fcfN = fcfs[-1] if fcfs else fcf0
    if wacc <= terminal_g:
        pv_terminal = np.nan
        msg.append("Terminal growth must be less than WACC; terminal value omitted.")
    else:
        tv = fcfN * (1 + terminal_g) / (wacc - terminal_g)
        pv_terminal = float(tv / ((1 + wacc) ** forecast_years))

    equity_value = pv_fcfs + (pv_terminal if np.isfinite(pv_terminal) else 0.0)
    shares_out = shares_out or 0.0
    per_share = (equity_value / shares_out) if shares_out and shares_out > 0 else None

    fdf = pd.DataFrame(
        {
            "Year": years,
            "FCF_Forecast": fcfs,
            "Discount_Factor": disc_factors,
            "PV_FCF": [fcfs[i] * disc_factors[i] for i in range(len(years))],
        }
    )

    return DCFResult(
        per_share_value=per_share,
        enterprise_value=None,  # FCFE-based DCF -> already equity value
        equity_value=equity_value,
        pv_fcfs=pv_fcfs,
        pv_terminal=pv_terminal if np.isfinite(pv_terminal) else None,
        fcf_forecast=fdf,
        message="; ".join(msg),
    )


# -------------------------------
# ML price forecasting engine (log-return model with future point)
# -------------------------------


@dataclass
class MLForecastResult:
    fig: go.Figure
    mape: Optional[float]
    rmse: Optional[float]
    horizon_days: int
    message: str = ""


def _build_features(prices: pd.Series, lookback: int = 20):
    """
    Build feature matrix (no targets): lagged log-returns, vol, moving averages.
    Used both for training and for the 'live' future forecast.
    """
    df = prices.to_frame("Close").copy()
    df["log_close"] = np.log(df["Close"])
    df["ret_1d"] = df["log_close"].diff()

    # Lagged returns
    for lag in range(1, lookback + 1):
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)

    # Volatility & moving averages
    df["vol_10"] = df["ret_1d"].rolling(10).std()
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    feature_cols = [c for c in df.columns if c.startswith("ret_lag_")] + [
        "vol_10",
        "vol_20",
        "ma_10",
        "ma_20",
    ]
    return df, feature_cols


def _build_ml_dataset(
    prices: pd.Series,
    horizon: int = 5,
    lookback: int = 20,
):
    """
    Build supervised dataset:

      Features: lagged returns + vol + MAs
      Target:   future log-return log(S_{t+h} / S_t)
    """
    if prices is None or len(prices) < lookback + horizon + 40:
        return None, None, None, None, None, "Not enough history for ML model."

    df_feat, feature_cols = _build_features(prices, lookback=lookback)

    # Targets
    df_feat["target_price"] = df_feat["Close"].shift(-horizon)
    df_feat["target_ret"] = np.log(df_feat["target_price"] / df_feat["Close"])

    # Keep only rows with full features + targets
    needed_cols = feature_cols + ["target_price", "target_ret"]
    df_model = df_feat.dropna(subset=needed_cols)
    if df_model.empty:
        return None, None, None, None, None, "Not enough rows after feature construction."

    X = df_model[feature_cols]
    y_ret = df_model["target_ret"]
    y_price = df_model["target_price"]
    base_price = df_model["Close"]

    return X, y_ret, y_price, base_price, feature_cols, ""


def train_ml_price_model(
    prices: pd.Series,
    horizon: int = 5,
) -> MLForecastResult:
    """
    Train RandomForest on log-returns h days ahead, backtest vs a random-walk
    baseline AND generate one true future forecast point (last date + h days).
    """
    if not SKLEARN_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(
            text="Install scikit-learn to enable ML forecasts:\n pip install scikit-learn",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return MLForecastResult(
            fig=fig,
            mape=None,
            rmse=None,
            horizon_days=horizon,
            message="Install scikit-learn to enable ML forecasts: pip install scikit-learn",
        )

    # Build training / backtest dataset
    X, y_ret, y_price, base_price, feature_cols, msg = _build_ml_dataset(
        prices, horizon=horizon
    )
    if X is None:
        fig = go.Figure()
        fig.add_annotation(
            text=msg,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return MLForecastResult(
            fig=fig,
            mape=None,
            rmse=None,
            horizon_days=horizon,
            message=msg,
        )

    n = len(X)
    base_msg = ""
    if n < 150:
        base_msg = f"ML: only {n} data points available; forecasts may be noisy."

    # Time-series split (no shuffle)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_ret = y_ret.iloc[:split]
    y_test_price = y_price.iloc[split:]
    base_price_test = base_price.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_ret)

    # ---- Backtest on the test window ----
    y_pred_ret = model.predict(X_test)
    y_pred_price = base_price_test * np.exp(y_pred_ret)

    mse = mean_squared_error(y_test_price, y_pred_price)
    rmse = float(math.sqrt(mse))
    mape = float(
        np.mean(np.abs((y_test_price - y_pred_price) / y_test_price)) * 100.0
    )

    # Baseline: random-walk (S_{t+h} ≈ S_t)
    baseline_pred = base_price_test
    baseline_mse = mean_squared_error(y_test_price, baseline_pred)
    baseline_rmse = float(math.sqrt(baseline_mse))
    baseline_mape = float(
        np.mean(np.abs((y_test_price - baseline_pred) / y_test_price)) * 100.0
    )

    improvement = (
        (baseline_mape - mape) / baseline_mape * 100.0 if baseline_mape > 0 else 0.0
    )
    if improvement >= 0:
        imp_text = f"≈{improvement:.1f}% lower error vs baseline."
    else:
        imp_text = f"≈{abs(improvement):.1f}% higher error than baseline."

    # ---- Build a true future forecast point ----
    df_feat_live, feat_cols_live = _build_features(prices)
    df_feat_live = df_feat_live.dropna(subset=feat_cols_live)
    future_price = None
    future_date = None
    last_date = None
    last_price = None

    if not df_feat_live.empty:
        latest_row = df_feat_live.iloc[-1]
        last_price = float(latest_row["Close"])
        last_date = latest_row.name
        X_future = latest_row[feat_cols_live].values.reshape(1, -1)
        future_ret = model.predict(X_future)[0]
        future_price = float(last_price * math.exp(future_ret))
        # Use calendar days for the x-axis; trading days would require a market calendar
        future_date = last_date + pd.Timedelta(days=horizon)

    # ---- Figure: backtest curves + future point/segment ----
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test_price.index,
            y=y_test_price.values,
            mode="lines",
            name="Actual future close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test_price.index,
            y=y_pred_price,
            mode="lines",
            name=f"ML forecast (t+{horizon}d)",
        )
    )

    # Add future forecast as a dashed segment and marker
    if future_price is not None and last_date is not None and future_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_date],
                y=[last_price, future_price],
                mode="lines",
                name="Future forecast path",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[future_date],
                y=[future_price],
                mode="markers",
                name=f"Forecast @ {future_date.date()}",
                marker=dict(size=10, symbol="star"),
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        yaxis_title="Price",
        xaxis_title="Date (target index)",
        height=420,
    )

    full_msg_parts = []
    if base_msg:
        full_msg_parts.append(base_msg)
    full_msg_parts.append(
        f"ML horizon {horizon} days — RMSE: {rmse:.2f}, MAPE: {mape:.2f}% "
        f"(baseline MAPE: {baseline_mape:.2f}%, {imp_text})"
    )
    if future_price is not None and future_date is not None and last_date is not None:
        full_msg_parts.append(
            f"Model forecast from {last_date.date()}: "
            f"≈${future_price:.2f} on {future_date.date()} (horizon {horizon} days)."
        )

    return MLForecastResult(
        fig=fig,
        mape=mape,
        rmse=rmse,
        horizon_days=horizon,
        message=" ".join(full_msg_parts),
    )


# -------------------------------
# Dash app
# -------------------------------

app: Dash = dash.Dash(__name__)
app.title = "DCF + Black-Scholes Dashboard"

card_style = {
    "border": "1px solid #e5e7eb",
    "borderRadius": "16px",
    "padding": "16px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
    "background": "white",
}

label_style = {"fontWeight": 600, "marginBottom": "6px", "display": "block"}

app.layout = html.Div(
    style={
        "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto",
        "padding": "20px",
        "background": "#f8fafc",
    },
    children=[
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": 16,
            },
            children=[
                html.Div(
                    [
                        html.H2(
                            "DCF + Black-Scholes KPI Dashboard", style={"margin": 0}
                        ),
                        html.Div(
                            "Enter a ticker, set assumptions, then Run Analysis.",
                            style={"color": "#64748b"},
                        ),
                    ]
                ),
                html.Div(
                    style={"display": "flex", "gap": 8},
                    children=[
                        dcc.Input(
                            id="ticker",
                            type="text",
                            placeholder="e.g., AAPL",
                            value="AAPL",
                            debounce=True,
                            style={
                                "height": 40,
                                "borderRadius": 12,
                                "padding": "0 12px",
                                "border": "1px solid #d1d5db",
                                "minWidth": 160,
                            },
                        ),
                        html.Button(
                            "Run Analysis",
                            id="run",
                            n_clicks=0,
                            style={
                                "height": 40,
                                "borderRadius": 12,
                                "padding": "0 16px",
                                "border": "none",
                                "background": "#111827",
                                "color": "white",
                                "fontWeight": 600,
                            },
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": 16},
            children=[
                # ---------------- LEFT SIDEBAR: Controls ----------------
                html.Div(
                    style=card_style,
                    children=[
                        html.H4("DCF Assumptions", style={"marginTop": 0}),
                        html.Label("Forecast Years", style=label_style),
                        dcc.Slider(
                            id="years",
                            min=3,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(3, 11)},
                        ),
                        html.Label("WACC (Discount Rate)", style=label_style),
                        dcc.Slider(
                            id="wacc",
                            min=0.04,
                            max=0.14,
                            step=0.005,
                            value=0.08,
                            marks={
                                0.04: "4%",
                                0.06: "6%",
                                0.08: "8%",
                                0.10: "10%",
                                0.12: "12%",
                                0.14: "14%",
                            },
                        ),
                        html.Div(
                            id="wacc_hint",
                            style={"fontSize": 12, "color": "#64748b", "marginTop": 4},
                        ),
                        html.Label("Terminal Growth Rate", style=label_style),
                        dcc.Slider(
                            id="terminal_g",
                            min=0.0,
                            max=0.06,
                            step=0.0025,
                            value=0.02,
                            marks={
                                0.0: "0%",
                                0.02: "2%",
                                0.03: "3%",
                                0.04: "4%",
                                0.05: "5%",
                                0.06: "6%",
                            },
                        ),
                        html.Label("FCF Growth (Optional Override)", style=label_style),
                        dcc.Slider(
                            id="fcf_growth",
                            min=-0.20,
                            max=0.25,
                            step=0.01,
                            value=0.05,
                            marks={
                                -0.20: "-20%",
                                -0.10: "-10%",
                                0.0: "0%",
                                0.10: "10%",
                                0.20: "20%",
                                0.25: "25%",
                            },
                        ),
                        html.Div(
                            "Leave unchanged to auto-estimate from history.",
                            style={"fontSize": 12, "color": "#64748b", "marginTop": 4},
                        ),
                        html.Hr(),
                        html.H4("Black-Scholes Assumptions"),
                        html.Label("Strike Price (K)", style=label_style),
                        dcc.Input(
                            id="strike",
                            type="number",
                            value=150,
                            style={"width": "100%"},
                        ),
                        html.Label("Time to Expiry (Years, T)", style=label_style),
                        dcc.Slider(
                            id="T",
                            min=0.1,
                            max=2.0,
                            step=0.05,
                            value=1.0,
                            marks={0.25: "0.25", 0.5: "0.5", 1.0: "1", 2.0: "2"},
                        ),
                        html.Label("Risk-Free Rate (r)", style=label_style),
                        dcc.Input(
                            id="rf",
                            type="number",
                            value=0.03,
                            step=0.001,
                            style={"width": "100%"},
                        ),
                        html.Label("Dividend Yield (q)", style=label_style),
                        dcc.Input(
                            id="div_yield",
                            type="number",
                            value=0.0,
                            step=0.001,
                            style={"width": "100%"},
                        ),
                        html.Label(
                            "Volatility Override (σ, optional)", style=label_style
                        ),
                        dcc.Input(
                            id="sigma_override",
                            type="number",
                            value=None,
                            step=0.01,
                            placeholder="Auto from history if empty",
                            style={"width": "100%"},
                        ),
                        html.Label(
                            "Monte Carlo Simulations (for GBM)", style=label_style
                        ),
                        dcc.Slider(
                            id="n_sims",
                            min=1000,
                            max=20000,
                            step=1000,
                            value=5000,
                            marks={
                                1000: "1k",
                                5000: "5k",
                                10000: "10k",
                                20000: "20k",
                            },
                        ),
                        html.Hr(),
                        html.Label("Trading Chart Type", style=label_style),
                        dcc.Dropdown(
                            id="chart_type",
                            options=[
                                {"label": "Candlestick + Volume", "value": "candles"},
                                {"label": "Line Chart", "value": "line"},
                            ],
                            value="candles",
                            clearable=False,
                        ),
                        html.Label("Indicators", style=label_style),
                        dcc.Checklist(
                            id="indicators",
                            options=[
                                {"label": "SMA(50)", "value": "sma50"},
                                {"label": "SMA(200)", "value": "sma200"},
                                {"label": "Bollinger Bands", "value": "bbands"},
                                {"label": "RSI", "value": "rsi"},
                                {"label": "MACD", "value": "macd"},
                                {"label": "VWAP", "value": "vwap"},
                                {"label": "ATR", "value": "atr"},
                            ],
                            value=["sma50", "sma200", "bbands"],
                            inline=True,
                        ),
                        html.Label("Chart Options", style=label_style),
                        dcc.Checklist(
                            id="chart_opts",
                            options=[
                                {"label": "Log Scale", "value": "log"},
                                {"label": "Show Crosshair", "value": "crosshair"},
                            ],
                            value=["log"],
                            inline=True,
                        ),
                    ],
                ),

                # ---------------- RIGHT MAIN AREA ----------------
                html.Div(
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(5, minmax(0, 1fr))",
                                "gap": 12,
                                "marginBottom": 16,
                            },
                            children=[
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.Div(
                                            "Current Price",
                                            style={
                                                "fontSize": 12,
                                                "color": "#64748b",
                                            },
                                        ),
                                        html.H3(
                                            id="kpi_price", style={"margin": 0}
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.Div(
                                            "DCF Value / Share",
                                            style={
                                                "fontSize": 12,
                                                "color": "#64748b",
                                            },
                                        ),
                                        html.H3(id="kpi_dcf", style={"margin": 0}),
                                    ],
                                ),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.Div(
                                            "Upside vs Market",
                                            style={
                                                "fontSize": 12,
                                                "color": "#64748b",
                                            },
                                        ),
                                        html.H3(
                                            id="kpi_upside", style={"margin": 0}
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.Div(
                                            "Annualized Volatility",
                                            style={
                                                "fontSize": 12,
                                                "color": "#64748b",
                                            },
                                        ),
                                        html.H3(id="kpi_vol", style={"margin": 0}),
                                    ],
                                ),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.Div(
                                            "BS Call / Put",
                                            style={
                                                "fontSize": 12,
                                                "color": "#64748b",
                                            },
                                        ),
                                        html.H3(id="kpi_bs", style={"margin": 0}),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style=card_style,
                            children=[
                                html.H4("TradingView-style Chart"),
                                dcc.Graph(
                                    id="tv_chart",
                                    config={"displayModeBar": True},
                                    style={"height": 520},
                                ),
                            ],
                        ),

                        # -------- DCF & Monte Carlo (2 columns) + wide ML card ------
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": 16,
                                "marginTop": 16,
                            },
                            children=[
                                # DCF card
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.H4("DCF: Forecast & Discounted FCF"),
                                        dcc.Graph(
                                            id="dcf_chart",
                                            config={"displayModeBar": False},
                                            style={"height": 320},
                                        ),
                                        html.Div(
                                            id="dcf_msg",
                                            style={
                                                "color": "#64748b",
                                                "fontSize": 12,
                                                "marginTop": 6,
                                            },
                                        ),
                                    ],
                                ),
                                # Monte Carlo card
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.H4("Monte Carlo: 1Y Price Distribution"),
                                        dcc.Graph(
                                            id="mc_hist",
                                            config={"displayModeBar": False},
                                            style={"height": 320},
                                        ),
                                    ],
                                ),
                                # ML forecast card – full width
                                html.Div(
                                    style={**card_style, "gridColumn": "1 / span 2"},
                                    children=[
                                        html.H4("ML Price Forecast (t + h days)"),
                                        html.Label(
                                            "ML forecast horizon (days)",
                                            style=label_style,
                                        ),
                                        dcc.Input(
                                            id="ml_horizon",
                                            type="number",
                                            value=5,
                                            min=1,
                                            step=1,
                                            style={"width": "120px"},
                                        ),
                                        dcc.Graph(
                                            id="ml_chart",
                                            config={"displayModeBar": False},
                                            style={"height": 420},
                                        ),
                                        html.Div(
                                            id="ml_metrics",
                                            style={
                                                "color": "#64748b",
                                                "fontSize": 12,
                                                "marginTop": 6,
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                html.H4("Option Payoff at Expiry"),
                                dcc.Graph(
                                    id="payoff_chart",
                                    config={"displayModeBar": False},
                                    style={"height": 320},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="hidden_state", style={"display": "none"}),
    ],
)

# -------------------------------
# Callbacks
# -------------------------------

# Auto-prefill parameters when ticker changes (and on first load)
@app.callback(
    Output("rf", "value"),
    Output("div_yield", "value"),
    Output("strike", "value"),
    Output("wacc", "value"),
    Output("fcf_growth", "value"),
    Input("ticker", "value"),
    prevent_initial_call=False,
)
def prefill_params(ticker):
    if not ticker:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    client = YFClient(ticker)
    S = client.last_price() or 100.0
    q = client.dividend_yield()
    beta = client.beta()
    rf = auto_rf()
    wacc = auto_wacc_from_beta(beta, rf)
    return (
        rf,
        q,
        S,  # strike default at-the-money
        wacc,
        0.05,
    )  # default fcf growth 5%


@app.callback(
    Output("kpi_price", "children"),
    Output("kpi_dcf", "children"),
    Output("kpi_upside", "children"),
    Output("kpi_vol", "children"),
    Output("kpi_bs", "children"),
    Output("tv_chart", "figure"),
    Output("dcf_chart", "figure"),
    Output("dcf_msg", "children"),
    Output("mc_hist", "figure"),
    Output("payoff_chart", "figure"),
    Input("run", "n_clicks"),
    Input("ticker", "value"),
    Input("years", "value"),
    Input("wacc", "value"),
    Input("terminal_g", "value"),
    Input("fcf_growth", "value"),
    Input("strike", "value"),
    Input("T", "value"),
    Input("rf", "value"),
    Input("div_yield", "value"),
    Input("sigma_override", "value"),
    Input("n_sims", "value"),
    Input("chart_type", "value"),
    Input("indicators", "value"),
    Input("chart_opts", "value"),
    prevent_initial_call=False,
)
def run_analysis(
    n_clicks,
    ticker,
    years,
    wacc,
    terminal_g,
    fcf_growth,
    K,
    T,
    r,
    q,
    sigma_override,
    n_sims,
    chart_type,
    indicators,
    chart_opts,
):
    if not ticker:
        return (
            "—",
            "—",
            "—",
            "—",
            "—",
            go.Figure(),
            go.Figure(),
            "",
            go.Figure(),
            go.Figure(),
        )

    yf_client = YFClient(ticker)

    # --- Market data ---
    price_series = yf_client.price_history("5y")
    S = yf_client.last_price()
    if S is None or not np.isfinite(S):
        S = price_series.iloc[-1] if not price_series.empty else np.nan

    # Dividend yield default from data if not set
    if q is None or (isinstance(q, (int, float)) and q <= 0):
        q = yf_client.dividend_yield()

    # Annualized vol from history if override not provided
    hist_sigma = ann_vol_from_history(price_series)
    sigma = sigma_override if sigma_override is not None else hist_sigma or 0.25

    # DCF data
    fcf_series = yf_client.free_cash_flows()
    shares_out = yf_client.shares_outstanding()

    dcf_res = dcf_valuation(
        fcf_series,
        wacc=float(wacc),
        terminal_g=float(terminal_g),
        forecast_years=int(years),
        fcf_growth=fcf_growth,
        shares_out=shares_out,
    )

    # KPIs
    price_kpi = f"${S:,.2f}" if np.isfinite(S) else "—"
    if dcf_res.per_share_value is not None and np.isfinite(dcf_res.per_share_value):
        dcf_kpi = f"${dcf_res.per_share_value:,.2f}"
        if np.isfinite(S) and S > 0:
            upside = (dcf_res.per_share_value / S) - 1
            upside_kpi = safe_pct(upside)
        else:
            upside_kpi = "—"
    else:
        dcf_kpi = "—"
        upside_kpi = "—"

    vol_kpi = safe_pct(hist_sigma)

    # Black-Scholes prices
    call_price, put_price = black_scholes(
        S=S, K=K, T=T, r=r, q=q, sigma=sigma or 0.25
    )
    if np.isfinite(call_price) and np.isfinite(put_price):
        bs_kpi = f"C: ${call_price:,.2f} / P: ${put_price:,.2f}"
    else:
        bs_kpi = "—"

    # ----------------------------
    # Trading-view style chart
    # ----------------------------
    ohlcv = yf_client.ohlcv_history("5y", "1d")

    tv_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    log_scale = "log" in (chart_opts or [])
    crosshair = "crosshair" in (chart_opts or [])

    if chart_type == "candles":
        tv_fig.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["Open"],
                high=ohlcv["High"],
                low=ohlcv["Low"],
                close=ohlcv["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
    else:
        tv_fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                name="Price",
            ),
            row=1,
            col=1,
        )

    closes = ohlcv["Close"]
    if "sma50" in (indicators or []):
        tv_fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=sma(closes, 50),
                mode="lines",
                name="SMA 50",
            ),
            row=1,
            col=1,
        )
    if "sma200" in (indicators or []):
        tv_fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=sma(closes, 200),
                mode="lines",
                name="SMA 200",
            ),
            row=1,
            col=1,
        )

    if "bbands" in (indicators or []):
        ma, upper, lower = bbands(closes, 20, 2.0)
        tv_fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=upper,
                mode="lines",
                name="BB Upper",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )
        tv_fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=lower,
                mode="lines",
                name="BB Lower",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

    if "vwap" in (indicators or []):
        tv_fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=vwap(ohlcv),
                mode="lines",
                name="VWAP",
            ),
            row=1,
            col=1,
        )

    # Volume
    tv_fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"] / 1e6,
            name="Volume (M)",
        ),
        row=2,
        col=1,
    )

    # RSI overlay
    if "rsi" in (indicators or []):
        tv_fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=rsi(closes),
                mode="lines",
                name="RSI(14)",
            ),
            row=2,
            col=1,
        )

    tv_fig.update_yaxes(
        type="log" if log_scale else "linear",
        row=1,
        col=1,
        title="Price",
    )
    tv_fig.update_yaxes(title="Volume (M)", row=2, col=1)
    tv_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified" if crosshair else "x",
        xaxis=dict(
            showspikes=crosshair,
            spikemode="across",
            spikesnap="cursor",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1, label="1M", step="month", stepmode="backward"
                    ),
                    dict(
                        count=3, label="3M", step="month", stepmode="backward"
                    ),
                    dict(
                        count=6, label="6M", step="month", stepmode="backward"
                    ),
                    dict(
                        count=1, label="YTD", step="year", stepmode="todate"
                    ),
                    dict(
                        count=1, label="1Y", step="year", stepmode="backward"
                    ),
                    dict(
                        count=5, label="5Y", step="year", stepmode="backward"
                    ),
                    dict(step="all"),
                ]
            ),
        ),
        yaxis=dict(
            showspikes=crosshair, spikemode="across", spikesnap="cursor"
        ),
    )

    # ----------------------------
    # DCF chart
    # ----------------------------
    dcf_fig = go.Figure()
    if not dcf_res.fcf_forecast.empty:
        fdf = dcf_res.fcf_forecast
        dcf_fig.add_trace(
            go.Bar(
                x=fdf["Year"],
                y=fdf["PV_FCF"],
                name="PV of FCF",
            )
        )
        dcf_fig.add_trace(
            go.Scatter(
                x=fdf["Year"],
                y=fdf["FCF_Forecast"],
                mode="lines+markers",
                name="FCF Forecast",
            )
        )
        dcf_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="Year",
            yaxis_title="FCF / PV",
        )
    dcf_msg = dcf_res.message

    # ----------------------------
    # Monte Carlo distribution
    # ----------------------------
    mc_fig = go.Figure()
    if np.isfinite(S) and np.isfinite(sigma) and T > 0:
        ST = gbm_terminal_prices(S0=S, T=T, r=r, q=q, sigma=sigma, n=n_sims)
        mc_fig.add_trace(
            go.Histogram(
                x=ST,
                nbinsx=60,
                name="Terminal Price",
            )
        )
        mc_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="Price after T years",
            yaxis_title="Frequency",
        )

    # ----------------------------
    # Option payoff chart
    # ----------------------------
    payoff_fig = go.Figure()
    Ks = np.linspace(max(1, 0.2 * S), 2.0 * S, 200) if np.isfinite(S) else np.linspace(
        10, 200, 200
    )
    call_payoff = np.maximum(Ks - K, 0)  # intrinsic of long call at expiry
    put_payoff = np.maximum(K - Ks, 0)  # long put
    payoff_fig.add_trace(
        go.Scatter(
            x=Ks,
            y=call_payoff,
            mode="lines",
            name="Call Payoff",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=Ks,
            y=put_payoff,
            mode="lines",
            name="Put Payoff",
        )
    )
    payoff_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="Underlying Price at Expiry",
        yaxis_title="Payoff",
    )

    return (
        price_kpi,
        dcf_kpi,
        upside_kpi,
        vol_kpi,
        bs_kpi,
        tv_fig,
        dcf_fig,
        dcf_msg,
        mc_fig,
        payoff_fig,
    )


@app.callback(
    Output("ml_chart", "figure"),
    Output("ml_metrics", "children"),
    Input("run", "n_clicks"),
    State("ticker", "value"),
    State("ml_horizon", "value"),
    prevent_initial_call=True,
)
def run_ml_forecast(n_clicks, ticker, ml_horizon):
    """Trigger ML training / forecast when the user clicks Run Analysis."""
    if not ticker:
        return go.Figure(), ""

    client = YFClient(ticker)
    prices = client.price_history("5y")

    if prices is None or prices.empty:
        return go.Figure(), "No price history available for ML forecast."

    h = int(ml_horizon or 5)
    result = train_ml_price_model(prices, horizon=h)
    metrics_text = (
        result.message or f"ML forecast horizon: {result.horizon_days} days."
    )
    return result.fig, metrics_text


if __name__ == "__main__":
    app.run(debug=True)
