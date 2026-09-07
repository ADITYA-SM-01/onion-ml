"""
main.py — OnionGuard ML Microservice
──────────────────────────────────────
FastAPI application exposing ML inference endpoints for the Node.js backend.

Endpoints:
  POST /ml/shelf-life       — Predict shelf life from sensor window
  GET  /ml/market-price     — Current onion prices from AGMARKNET
  GET  /ml/price-forecast   — 7-day Prophet/ARIMA price forecast
  GET  /ml/health           — Health check

REAL HARDWARE SENSORS (used in shelf-life prediction):
  - DHT22 x2  : temperature (°C), relative humidity (%RH)
  - MQ-135    : nh3 (ammonia ppm) — spoilage gas indicator
  - MQ-4      : ch4 (methane ppm) — fermentation gas indicator
  - ESP32-CAM : light (irradiance W/m²) — from camera image brightness
  - SW-420    : vibration (0/1)

NO CO2 sensor. NO ethylene sensor.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, AliasChoices

load_dotenv()

import shelf_life as sl
import market_price as mp
import price_forecast as pf

# ── Lifespan: pre-load ML models at startup ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] OnionGuard ML Microservice starting...")
    sl._load_models()  # Pre-warm model cache
    yield
    print("[INFO] Shutting down ML Microservice.")

app = FastAPI(
    title="OnionGuard ML Service",
    description="Shelf life prediction + onion market price forecasting",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/ml/docs",
    redoc_url=None,
)

# ── CORS (internal service, restrict to backend only in production) ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to backend URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ShelfLifeRequest(BaseModel):
    """
    Sensor window for shelf-life prediction.
    Send the last N readings (typically 6–24 hours worth).

    Fields map directly to ESP32 sensor hardware:
      temperature  — DHT22 average (°C)
      humidity     — DHT22 average (%RH)
      nh3          — MQ-135 ammonia ppm  [was: co2/ammonia — removed]
      ch4          — MQ-4 methane ppm    [was: co — renamed for clarity]
      light        — ESP32-CAM irradiance W/m² (camera-derived)
      vibration    — SW-420 (0.0 or 1.0)
    """
    temperature: List[float] = Field(..., min_length=1, max_length=168)
    humidity:    List[float] = Field(..., min_length=1, max_length=168)

    # MQ-135 → NH3 (ammonia) ppm — primary spoilage gas indicator
    nh3: List[float] = Field(
        ..., min_length=1, max_length=168,
        validation_alias=AliasChoices('nh3', 'ammonia'),
        description="NH3 ammonia ppm from MQ-135 sensor"
    )

    # MQ-4 → CH4 (methane) ppm — fermentation indicator
    ch4: List[float] = Field(
        ..., min_length=1, max_length=168,
        validation_alias=AliasChoices('ch4', 'co'),
        description="CH4 methane ppm from MQ-4 sensor"
    )

    # Irradiance W/m² from ESP32-CAM image
    light: List[float] = Field(..., min_length=1, max_length=168,
                                description="Irradiance W/m² from camera")

    vibration: List[float] = Field(..., min_length=1, max_length=168)

    time_since_loading_days: float = Field(
        default=0.0, ge=0, le=365,
        validation_alias=AliasChoices('time_since_loading_days', 'timeSinceLoadingDays')
    )
    initial_load: float = Field(
        default=250.0, ge=1,
        validation_alias=AliasChoices('initial_load', 'initialLoad')
    )
    current_load: float = Field(
        default=240.0, ge=0,
        validation_alias=AliasChoices('current_load', 'currentLoad')
    )

    @field_validator("temperature", "humidity", "nh3", "ch4", "light", "vibration", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, (int, float)):
            return [v]
        return v


class ShelfLifeResponse(BaseModel):
    shelf_life_days:      float
    spoilage_probability: float
    risk_class:           str
    source:               str
    recommendation:       str


class PriceEntry(BaseModel):
    date:      str
    modal_price: int
    min_price:   int
    max_price:   int
    market:    str
    district:  str
    state:     str
    commodity: str
    unit:      str


class ForecastEntry(BaseModel):
    date:            str
    predicted_price: int
    lower_bound:     int
    upper_bound:     int
    is_forecast:     bool


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/ml/health")
async def health_check():
    return {
        "status": "ok",
        "service": "onionguard-ml",
        "models_loaded": sl._models_loaded and sl._clf is not None,
        "model_source": "ml-model" if sl._clf else "heuristic",
        "agmarknet_configured": bool(os.getenv("AGMARKNET_API_KEY")),
        "prophet_available": pf.HAS_PROPHET,
        "sklearn_available": pf.HAS_SKLEARN,
        "sensor_schema": "v2 (nh3+ch4+irradiance — no co2/ethylene)",
    }


@app.post("/ml/shelf-life", response_model=ShelfLifeResponse)
async def shelf_life_endpoint(body: ShelfLifeRequest):
    """
    Predict residual shelf life and spoilage risk from a sensor window.
    Send the last N hours of sensor readings (typically 6–24 hours).

    Accepts both the new field names (nh3, ch4) and legacy aliases
    (ammonia, co) for backward compatibility.
    """
    try:
        result = sl.predict(
            temperature=body.temperature,
            humidity=body.humidity,
            nh3=body.nh3,
            ch4=body.ch4,
            light=body.light,
            vibration=body.vibration,
            time_since_loading_days=body.time_since_loading_days,
            initial_load=body.initial_load,
            current_load=body.current_load,
        )

        # Add human-readable recommendation
        days = result["shelf_life_days"]
        risk = result["risk_class"]
        if risk == "High" or days < 30:
            rec = "⚠️ Urgent: Sell or move onions within 30 days. Check ventilation immediately."
        elif risk == "Medium" or days < 90:
            rec = "🟡 Monitor closely. Consider selling within 2-3 months for best returns."
        else:
            rec = "✅ Storage conditions are good. Onions are safe for extended storage."

        result["recommendation"] = rec
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/market-price")
async def market_price_endpoint(district: Optional[str] = Query(default=None)):
    """
    Returns current onion prices from AGMARKNET.
    Set AGMARKNET_API_KEY in .env for live data; falls back to realistic mock.
    """
    data = await mp.fetch_current_prices(district)
    return data


@app.get("/ml/price-forecast")
async def price_forecast_endpoint():
    """
    Returns 7-day onion price forecast using Ridge regression / Prophet + last 30 days history.
    Model priority: Prophet → Ridge Regression → Linear Trend
    """
    history = await mp.fetch_price_history(days=30)
    result  = await pf.get_price_forecast(history, forecast_days=7)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
