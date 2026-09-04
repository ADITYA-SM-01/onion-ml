"""
price_forecast.py
──────────────────
Forecasts onion prices for the next N days.

Strategy (graceful degradation chain):
  1. Prophet   — if installed (needs C++ compiler: pip install prophet)
  2. sklearn   — Ridge regression on lag features (always available)
  3. Linear trend — fallback using numpy polyfit
"""

import os
import numpy as np
from datetime import date, timedelta
from typing import Optional

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))
HISTORY_DAYS  = int(os.getenv("HISTORY_DAYS", "30"))


# ── Main entry ────────────────────────────────────────────────────────────────

async def get_price_forecast(
    history: list[dict],         # list of {"date": "YYYY-MM-DD", "modal_price": int}
    forecast_days: int = FORECAST_DAYS,
) -> dict:
    """
    Forecast onion prices using the best available model.
    history: list of records sorted oldest → newest.
    Returns: { history, forecast, trend, trend_pct, recommendation, model_used }
    """
    if not history or len(history) < 3:
        return _linear_forecast(history or [], forecast_days)

    if HAS_PROPHET:
        return _prophet_forecast(history, forecast_days)
    elif HAS_SKLEARN:
        return _sklearn_forecast(history, forecast_days)
    else:
        return _linear_forecast(history, forecast_days)


# ── Prophet forecast ──────────────────────────────────────────────────────────

def _prophet_forecast(history: list[dict], n: int) -> dict:
    """Facebook Prophet — best quality seasonal + trend forecast."""
    try:
        import pandas as pd
        df = pd.DataFrame([{"ds": r["date"], "y": r["modal_price"]} for r in history])
        df["ds"] = pd.to_datetime(df["ds"])

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            seasonality_prior_scale=10,
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=n)
        forecast = m.predict(future)

        forecast_rows = forecast.tail(n)
        last_actual = df["y"].iloc[-1]

        fc_list = []
        for _, row in forecast_rows.iterrows():
            fc_list.append({
                "date":            row["ds"].strftime("%Y-%m-%d"),
                "predicted_price": max(700, int(row["yhat"])),
                "lower_bound":     max(700, int(row["yhat_lower"])),
                "upper_bound":     max(700, int(row["yhat_upper"])),
                "is_forecast":     True,
            })

        trend_pct = int((fc_list[-1]["predicted_price"] - last_actual) / last_actual * 100)
        trend = "rising" if trend_pct > 3 else ("falling" if trend_pct < -3 else "stable")

        return {
            "history":         history,
            "forecast":        fc_list,
            "trend":           trend,
            "trend_pct":       abs(trend_pct),
            "recommendation":  _recommendation(trend, trend_pct),
            "model_used":      "prophet",
        }
    except Exception as e:
        print(f"[Forecast] Prophet failed: {e}. Falling back to sklearn.")
        return _sklearn_forecast(history, n)


# ── scikit-learn Ridge forecast ───────────────────────────────────────────────

def _sklearn_forecast(history: list[dict], n: int) -> dict:
    """Ridge regression on lag features — no C++ needed, always available."""
    prices = [r["modal_price"] for r in history]
    lags   = 7  # number of lag features

    if len(prices) <= lags:
        return _linear_forecast(history, n)

    X, y = [], []
    for i in range(lags, len(prices)):
        X.append(prices[i - lags:i])
        y.append(prices[i])

    X_arr = np.array(X)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_arr)

    # Iteratively predict next n days
    recent = list(prices[-lags:])
    fc_list = []
    for i in range(n):
        feat = scaler.transform([recent[-lags:]])
        pred = int(max(700, model.predict(feat)[0]))
        next_date = (date.today() + timedelta(days=i + 1)).isoformat()
        # Simple uncertainty band ±8%
        fc_list.append({
            "date":            next_date,
            "predicted_price": pred,
            "lower_bound":     max(700, int(pred * 0.92)),
            "upper_bound":     int(pred * 1.08),
            "is_forecast":     True,
        })
        recent.append(pred)

    last_actual = prices[-1]
    trend_pct   = int((fc_list[-1]["predicted_price"] - last_actual) / last_actual * 100)
    trend       = "rising" if trend_pct > 3 else ("falling" if trend_pct < -3 else "stable")

    return {
        "history":        history,
        "forecast":       fc_list,
        "trend":          trend,
        "trend_pct":      abs(trend_pct),
        "recommendation": _recommendation(trend, trend_pct),
        "model_used":     "ridge-regression",
    }


# ── Linear trend forecast ─────────────────────────────────────────────────────

def _linear_forecast(history: list[dict], n: int) -> dict:
    """Numpy polyfit linear trend — last resort, zero dependencies."""
    prices = [r["modal_price"] for r in history] if history else [2000]
    xs = np.arange(len(prices))
    coeffs = np.polyfit(xs, prices, 1) if len(prices) > 1 else [0, prices[0]]
    slope, intercept = coeffs[0], coeffs[1]

    last_actual = prices[-1]
    fc_list = []
    for i in range(n):
        pred = int(max(700, intercept + slope * (len(prices) + i)))
        next_date = (date.today() + timedelta(days=i + 1)).isoformat()
        fc_list.append({
            "date":            next_date,
            "predicted_price": pred,
            "lower_bound":     max(700, int(pred * 0.90)),
            "upper_bound":     int(pred * 1.10),
            "is_forecast":     True,
        })

    trend_pct = int((fc_list[-1]["predicted_price"] - last_actual) / last_actual * 100) if last_actual else 0
    trend     = "rising" if trend_pct > 3 else ("falling" if trend_pct < -3 else "stable")

    return {
        "history":        history,
        "forecast":       fc_list,
        "trend":          trend,
        "trend_pct":      abs(trend_pct),
        "recommendation": _recommendation(trend, trend_pct),
        "model_used":     "linear-trend",
    }


# ── Recommendation text ───────────────────────────────────────────────────────

def _recommendation(trend: str, trend_pct: int) -> str:
    abs_pct = abs(trend_pct)
    if trend == "rising":
        if abs_pct >= 15:
            return f"📈 Strong upward trend (+{abs_pct}% projected). If you can wait, hold stock — prices should peak soon. Consider locking forward contracts."
        return f"📈 Prices trending up {abs_pct}%. Consider delaying sale by 1–2 weeks to capture higher margin."
    elif trend == "falling":
        if abs_pct >= 15:
            return f"📉 Sharp price decline expected (-{abs_pct}%). Sell as soon as possible to minimise loss."
        return f"📉 Mild downward trend ({abs_pct}%). Monitor daily. Consider selling a portion now to reduce risk."
    else:
        return "📊 Market is stable. Monitor Lasalgaon mandi daily for the right selling window."
