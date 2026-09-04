"""
shelf_life.py
─────────────
Wraps the trained LightGBM / XGBoost models from BASIC-ML/onion_ml/models/
to provide shelf-life regression and risk classification via REST.

If models are not yet trained, falls back to a physiologically-grounded
heuristic (same relationships as the training simulator).
"""

import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
# Look for models in this directory first, then fall back to BASIC-ML
MODEL_DIRS = [
    HERE / "models",
    HERE.parent / "BASIC-ML" / "onion_ml" / "models",
]

RISK_ORDER = ["Low", "Medium", "High"]
FEATURE_COLUMNS = [
    "temp_mean", "temp_var",
    "rh_mean", "rh_var",
    "vpd_kpa",
    "co2_mean", "co2_roc",
    "ethylene_mean", "ethylene_roc",
    "o2_depletion_rate",
    "cum_weight_loss_pct",
    "door_open_freq", "door_open_duration_hrs",
    "light_exposure_hrs",
    "time_since_loading_days",
]

# ── Load models (lazy, cached) ───────────────────────────────────────────────
_clf = None
_reg = None
_models_loaded = False


def _find_models():
    for d in MODEL_DIRS:
        clf_path = d / "risk_classifier.joblib"
        reg_path = d / "shelf_life_regressor.joblib"
        if clf_path.exists() and reg_path.exists():
            return clf_path, reg_path
    return None, None


def _load_models():
    global _clf, _reg, _models_loaded
    if _models_loaded:
        return
    clf_path, reg_path = _find_models()
    if clf_path:
        try:
            _clf = joblib.load(clf_path)
            _reg = joblib.load(reg_path)
            print(f"[ShelfLife] Models loaded from {clf_path.parent}")
        except Exception as e:
            print(f"[ShelfLife] Failed to load models: {e}. Using heuristic.")
    else:
        print("[ShelfLife] No trained models found. Using heuristic.")
    _models_loaded = True


# ── Feature engineering ───────────────────────────────────────────────────────

def _vapor_pressure_deficit(temp_c: float, rh_pct: float) -> float:
    svp = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    return svp * (1 - rh_pct / 100.0)


def _compute_features(
    temperature: list[float],
    humidity: list[float],
    co2: list[float],
    ammonia: list[float],
    light: list[float],
    vibration: list[float],
    time_since_loading_days: float,
    initial_load: float = 250.0,
    current_load: float = 240.0,
    prev_co2_mean: Optional[float] = None,
) -> dict:
    temp_arr = np.array(temperature, dtype=float)
    rh_arr = np.array(humidity, dtype=float)
    co2_arr = np.array(co2, dtype=float)

    temp_mean = float(np.mean(temp_arr))
    temp_var = float(np.var(temp_arr))
    rh_mean = float(np.mean(rh_arr))
    rh_var = float(np.var(rh_arr))
    vpd = _vapor_pressure_deficit(temp_mean, rh_mean)
    co2_mean = float(np.mean(co2_arr))
    co2_roc = 0.0 if prev_co2_mean is None else (co2_mean - prev_co2_mean)

    # Ammonia as proxy for ethylene (both indicate organic decomposition)
    eth_mean = float(np.mean(ammonia))
    eth_roc = 0.0

    # O2 depletion rate estimated from CO2 increase (respiration stoichiometry)
    o2_dep_rate = max(0.0, co2_roc * 0.93)

    cum_weight_loss_pct = 100.0 * (initial_load - current_load) / initial_load if initial_load > 0 else 0.0

    # Light as lux array — count hours with lux > 20
    light_arr = np.array(light, dtype=float)
    light_exposure_hrs = int(np.sum(light_arr > 20))

    return {
        "temp_mean": temp_mean, "temp_var": temp_var,
        "rh_mean": rh_mean, "rh_var": rh_var,
        "vpd_kpa": vpd,
        "co2_mean": co2_mean, "co2_roc": co2_roc,
        "ethylene_mean": eth_mean, "ethylene_roc": eth_roc,
        "o2_depletion_rate": o2_dep_rate,
        "cum_weight_loss_pct": cum_weight_loss_pct,
        "door_open_freq": 0, "door_open_duration_hrs": 0,
        "light_exposure_hrs": light_exposure_hrs,
        "time_since_loading_days": time_since_loading_days,
    }


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_shelf_life(features: dict) -> dict:
    temp = features["temp_mean"]
    rh = features["rh_mean"]
    ammonia = features["ethylene_mean"]
    days = features["time_since_loading_days"]
    light = features["light_exposure_hrs"]

    max_days = 180.0  # Baseline 6 months for well-stored onions

    # Temperature penalties (optimal: 10-18°C)
    if temp > 25:
        max_days -= (temp - 25) * 5
    elif temp > 18:
        max_days -= (temp - 18) * 2
    elif temp < 5:
        max_days -= (5 - temp) * 8

    # Humidity penalties (optimal: 60-80%)
    if rh > 80:
        max_days -= (rh - 80) * 3
    elif rh < 60:
        max_days -= (60 - rh) * 2

    # Gas penalties
    if ammonia > 5:
        max_days -= (ammonia - 5) * 10

    # Light exposure penalty
    if light > 2:
        max_days -= light * 1.5

    remaining = max(0.0, round(max_days - days))
    spoilage = 0.8 if remaining < 30 else (0.4 if remaining < 90 else 0.1)
    risk = "High" if remaining < 30 else ("Medium" if remaining < 90 else "Low")

    return {
        "shelf_life_days": remaining,
        "spoilage_probability": spoilage,
        "risk_class": risk,
        "source": "heuristic",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    temperature: list[float],
    humidity: list[float],
    co2: list[float],
    ammonia: list[float],
    light: list[float],
    vibration: list[float],
    time_since_loading_days: float,
    initial_load: float = 250.0,
    current_load: float = 240.0,
) -> dict:
    _load_models()

    features = _compute_features(
        temperature, humidity, co2, ammonia, light, vibration,
        time_since_loading_days, initial_load, current_load,
    )

    if _clf is None or _reg is None:
        return _heuristic_shelf_life(features)

    try:
        import pandas as pd
        x = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        proba = _clf.predict_proba(x)[0]
        high_idx = RISK_ORDER.index("High")
        spoilage_prob = float(proba[high_idx])
        risk_class = RISK_ORDER[int(np.argmax(proba))]
        shelf_life = float(max(0, _reg.predict(x)[0]))
        return {
            "shelf_life_days": round(shelf_life, 1),
            "spoilage_probability": round(spoilage_prob, 3),
            "risk_class": risk_class,
            "source": "ml-model",
            "features": features,
        }
    except Exception as e:
        print(f"[ShelfLife] ML inference error: {e}. Falling back to heuristic.")
        return _heuristic_shelf_life(features)
