"""
shelf_life.py
─────────────

Wraps the trained LightGBM / XGBoost models from BASIC-ML/onion_ml/models/
to provide shelf-life regression and risk classification via REST.

If models are not yet trained, falls back to a physiologically-grounded
heuristic calibrated to the real sensor hardware.

REAL HARDWARE SENSORS (calibrated to live serial readings and datasheets):
  - DHT22 x2  : temperature (°C), relative humidity (%RH)
                DHT22 #1 may return NaN — backend falls back to DHT22 #2.
  - MQ-135    : Air-quality / NH3 sensor
                  Clean air baseline : ~400 PPM (CO2-equivalent floor)
                  Good ventilation   : 400–1,000 PPM
                  Spoilage gases     : > 1,000 PPM; up to 3,000 PPM
                  (firmware: MQ135_MIN_PPM=400, MQ135_MAX_PPM=3000)
  - MQ-4      : CH4 (methane) sensor
                  Atmospheric CH4 ≈ 2 PPM — BELOW MQ-4 detection range
                  Sensor floor in clean air: ~326 PPM (hardware confirmed)
                  Fermentation onset: rises well above 1,000 PPM
                  (firmware: MQ4_MIN_PPM=300, MQ4_MAX_PPM=10000)
  - ESP32-CAM : irradiance (W/m²) — optional, defaults to 0.0
  - SW-420    : vibration (bool → 0.0/1.0)

NO CO2 sensor. NO ethylene sensor. NO MQ-6 sensor.
"""

import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
MODEL_DIRS = [
    HERE / "models",
    HERE.parent / "BASIC-ML" / "onion_ml" / "models",
]

RISK_ORDER = ["Low", "Medium", "High"]

# Feature columns — must match BASIC-ML/onion_ml/features.py exactly
FEATURE_COLUMNS = [
    "temp_mean", "temp_var",
    "rh_mean", "rh_var",
    "vpd_kpa",
    "nh3_mean", "nh3_roc",
    "ch4_mean", "ch4_roc",
    "irradiance_mean",
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
    nh3: list[float],
    ch4: list[float],
    light: list[float] | None,
    vibration: list[float],
    time_since_loading_days: float,
    initial_load: float = 250.0,
    current_load: float = 240.0,
    prev_nh3_mean: Optional[float] = None,
    prev_ch4_mean: Optional[float] = None,
) -> dict:
    temp_arr  = np.array(temperature, dtype=float)
    rh_arr    = np.array(humidity, dtype=float)
    nh3_arr   = np.array(nh3, dtype=float)
    ch4_arr   = np.array(ch4, dtype=float)

    # light is OPTIONAL — the camera node pushes frames independently.
    # Default to [0.0] when absent so the feature is always present
    # (0 W/m² = no irradiance detected, no light-stress penalty applied).
    if not light:
        light_arr = np.array([0.0], dtype=float)
    else:
        light_arr = np.array(light, dtype=float)

    temp_mean = float(np.mean(temp_arr))
    temp_var  = float(np.var(temp_arr))
    rh_mean   = float(np.mean(rh_arr))
    rh_var    = float(np.var(rh_arr))
    vpd       = _vapor_pressure_deficit(temp_mean, rh_mean)

    nh3_mean  = float(np.mean(nh3_arr))
    ch4_mean  = float(np.mean(ch4_arr))
    irradiance_mean = float(np.mean(light_arr))

    # Rate-of-change vs previous window
    nh3_roc = 0.0 if prev_nh3_mean is None else (nh3_mean - prev_nh3_mean)
    ch4_roc = 0.0 if prev_ch4_mean is None else (ch4_mean - prev_ch4_mean)

    cum_weight_loss_pct = (
        100.0 * (initial_load - current_load) / initial_load
        if initial_load > 0 else 0.0
    )

    # Light exposure hours (irradiance > 20 W/m² counts as light stress)
    light_exposure_hrs = int(np.sum(light_arr > 20.0))

    return {
        "temp_mean": temp_mean, "temp_var": temp_var,
        "rh_mean": rh_mean, "rh_var": rh_var,
        "vpd_kpa": vpd,
        "nh3_mean": nh3_mean, "nh3_roc": nh3_roc,
        "ch4_mean": ch4_mean, "ch4_roc": ch4_roc,
        "irradiance_mean": irradiance_mean,
        "cum_weight_loss_pct": cum_weight_loss_pct,
        "door_open_freq": 0, "door_open_duration_hrs": 0,
        "light_exposure_hrs": light_exposure_hrs,
        "time_since_loading_days": time_since_loading_days,
    }


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_shelf_life(features: dict) -> dict:
    temp         = features["temp_mean"]
    rh           = features["rh_mean"]
    nh3          = features["nh3_mean"]     # ppm from MQ-135 (baseline ~400 ppm clean air)
    ch4          = features["ch4_mean"]     # ppm from MQ-4  (sensor floor ~326 ppm clean air)
    days         = features["time_since_loading_days"]
    irr          = features["irradiance_mean"]  # W/m² from camera (0 if no frame)

    max_days = 180.0  # Baseline 6 months for well-stored onions

    # Temperature penalties (optimal: 10–18°C; Indian shed: often 28–35°C)
    if temp > 30:
        max_days -= (temp - 30) * 6
    elif temp > 18:
        max_days -= (temp - 18) * 2
    elif temp < 5:
        max_days -= (5 - temp) * 8

    # Humidity penalties (optimal: 60–75% RH)
    if rh > 80:
        max_days -= (rh - 80) * 3
    elif rh < 55:
        max_days -= (55 - rh) * 2

    # NH3 penalty (MQ-135):
    #   Baseline in clean storage air: ~400 PPM (sensor floor).
    #   400–800 PPM  = normal storage air, no penalty
    #   800–1,500 PPM = elevated, mild spoilage signal
    #   > 1,500 PPM  = heavy ammonia buildup, active spoilage
    if nh3 > 1500:
        max_days -= (nh3 - 1500) * 0.04
    elif nh3 > 800:
        max_days -= (nh3 - 800) * 0.01

    # CH4 penalty (MQ-4):
    #   Sensor floor in clean air: ~326 PPM (not atmospheric; MQ-4 can’t read 2 PPM).
    #   330–1,000 PPM  = normal / sensor baseline range, no penalty
    #   1,000–2,000 PPM = fermentation starting
    #   > 2,000 PPM     = active anaerobic fermentation
    if ch4 > 2000:
        max_days -= (ch4 - 2000) * 0.005
    elif ch4 > 1000:
        max_days -= (ch4 - 1000) * 0.002

    # Irradiance penalty (light stress causes sprouting)
    # 0 W/m² = no camera frame or dark storage — no penalty applied
    if irr > 50:
        max_days -= (irr - 50) * 0.05

    remaining  = max(0.0, round(max_days - days))
    spoilage   = 0.8 if remaining < 30 else (0.4 if remaining < 90 else 0.1)
    risk       = "High" if remaining < 30 else ("Medium" if remaining < 90 else "Low")

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
    nh3: list[float],
    ch4: list[float],
    light: list[float],
    vibration: list[float],
    time_since_loading_days: float,
    initial_load: float = 250.0,
    current_load: float = 240.0,
) -> dict:
    _load_models()

    features = _compute_features(
        temperature, humidity, nh3, ch4, light, vibration,
        time_since_loading_days, initial_load, current_load,
    )

    if _clf is None or _reg is None:
        return _heuristic_shelf_life(features)

    try:
        import pandas as pd
        x = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        proba          = _clf.predict_proba(x)[0]
        high_idx       = RISK_ORDER.index("High")
        spoilage_prob  = float(proba[high_idx])
        risk_class     = RISK_ORDER[int(np.argmax(proba))]
        shelf_life     = float(max(0, _reg.predict(x)[0]))
        return {
            "shelf_life_days":      round(shelf_life, 1),
            "spoilage_probability": round(spoilage_prob, 3),
            "risk_class":           risk_class,
            "source":               "ml-model",
            "features":             features,
        }
    except Exception as e:
        print(f"[ShelfLife] ML inference error: {e}. Falling back to heuristic.")
        return _heuristic_shelf_life(features)
