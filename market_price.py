"""
market_price.py
────────────────
Fetches onion market prices from data.gov.in AGMARKNET API.

Resource: Variety-wise Daily Market Prices Data of Commodity
Endpoint: GET /resource/35985678-0d79-46b4-9ed6-6f13308a1d24

Confirmed field names from data.gov.in API response (verified live):
  Arrival_Date, Commodity, Commodity_Code, District, Grade,
  Market, Max_Price, Min_Price, Modal_Price, State, Variety

API Parameters:
  api-key              → your API key (required)
  format               → json (default)
  offset               → pagination skip
  limit                → max records
  filters[State]       → state name
  filters[District]    → district name
  filters[Commodity]   → commodity name (capital C)
  filters[Arrival_Date]→ date filter DD/MM/YYYY
"""

import os
import httpx
import random
from datetime import date, timedelta
from typing import Optional

AGMARKNET_BASE = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
COMMODITY = os.getenv("AGMARKNET_COMMODITY", "Onion")

# Representative mandis for mock data
MAJOR_MANDIS = [
    # Maharashtra
    {"market": "Lasalgaon",        "district": "Nashik",       "state": "Maharashtra"},
    {"market": "Pimpalgaon",       "district": "Nashik",       "state": "Maharashtra"},
    {"market": "Yeola",            "district": "Nashik",       "state": "Maharashtra"},
    {"market": "Solapur",          "district": "Solapur",      "state": "Maharashtra"},
    {"market": "Ahmednagar",       "district": "Ahmednagar",   "state": "Maharashtra"},
    {"market": "Pune",             "district": "Pune",         "state": "Maharashtra"},
    {"market": "Kolhapur",         "district": "Kolhapur",     "state": "Maharashtra"},
    # Karnataka
    {"market": "Bangalore",        "district": "Bangalore",    "state": "Karnataka"},
    {"market": "Hubli (Amaragol)", "district": "Dharwad",      "state": "Karnataka"},
    {"market": "Bellary",          "district": "Ballari",      "state": "Karnataka"},
    {"market": "Davanagere",       "district": "Davanagere",   "state": "Karnataka"},
    {"market": "Gadag",            "district": "Gadag",        "state": "Karnataka"},
    # Madhya Pradesh
    {"market": "Indore",           "district": "Indore",       "state": "Madhya Pradesh"},
    {"market": "Neemuch",          "district": "Neemuch",      "state": "Madhya Pradesh"},
    {"market": "Mandsaur",         "district": "Mandsaur",     "state": "Madhya Pradesh"},
    {"market": "Ujjain",           "district": "Ujjain",       "state": "Madhya Pradesh"},
    # Gujarat
    {"market": "Mahuva",           "district": "Bhavnagar",    "state": "Gujarat"},
    {"market": "Gondal",           "district": "Rajkot",       "state": "Gujarat"},
    {"market": "Rajkot",           "district": "Rajkot",       "state": "Gujarat"},
    {"market": "Surat",            "district": "Surat",        "state": "Gujarat"},
    # Rajasthan
    {"market": "Alwar",            "district": "Alwar",        "state": "Rajasthan"},
    {"market": "Jaipur",           "district": "Jaipur",       "state": "Rajasthan"},
    {"market": "Sikar",            "district": "Sikar",        "state": "Rajasthan"},
    # Andhra Pradesh & Telangana
    {"market": "Kurnool",          "district": "Kurnool",      "state": "Andhra Pradesh"},
    {"market": "Mahabubnagar",     "district": "Mahabubnagar", "state": "Telangana"},
    {"market": "Bowenpally",       "district": "Hyderabad",    "state": "Telangana"},
    # Delhi & Northern States
    {"market": "Azadpur",          "district": "Delhi",        "state": "Delhi"},
    {"market": "Kanpur",           "district": "Kanpur",       "state": "Uttar Pradesh"},
    {"market": "Kolkata",          "district": "Kolkata",      "state": "West Bengal"},
]

# ── Field normalisation ───────────────────────────────────────────────────────

def _normalize_record(r: dict) -> dict:
    """
    Normalise a data.gov.in record to our standard schema.
    Handles the exact casing returned by the API:
      Arrival_Date, Modal_Price, Min_Price, Max_Price, Market, District, State, Variety
    """
    def get(*keys) -> str:
        for k in keys:
            v = r.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    # Price fields — data.gov.in uses capital-case
    try:
        modal = int(float(get("Modal_Price",  "modal_price",  "modal price")  or 0))
        min_p = int(float(get("Min_Price",    "min_price",    "min price")    or 0))
        max_p = int(float(get("Max_Price",    "max_price",    "max price")    or 0))
    except (ValueError, TypeError):
        modal = min_p = max_p = 0

    # Date — API returns DD/MM/YYYY, convert to ISO
    arrival = get("Arrival_Date", "arrival_date", "Arrival Date") or date.today().isoformat()
    if "/" in arrival:
        parts = arrival.split("/")
        if len(parts) == 3:
            arrival = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

    return {
        "market":    get("Market",    "market"),
        "district":  get("District",  "district"),
        "state":     get("State",     "state"),
        "commodity": get("Commodity", "commodity"),
        "variety":   get("Variety",   "variety"),
        "grade":     get("Grade",     "grade"),
        "modal_price": modal,
        "min_price":   min_p,
        "max_price":   max_p,
        "date":        arrival,
        "unit":        "Quintal",
    }


# ── Mock data generators ──────────────────────────────────────────────────────

def _generate_mock_prices(num_days: int = 30) -> list[dict]:
    """Synthetic but seasonally realistic onion prices."""
    random.seed(42)
    today = date.today()
    month = today.month
    base = 1200 if month in (10, 11, 12) else (2800 if month in (4, 5, 6) else 2000)

    prices, price = [], float(base)
    for i in range(num_days):
        d = today - timedelta(days=num_days - i - 1)
        price = max(700.0, min(5000.0, price + random.gauss(0, 80)))
        prices.append({
            "date":        d.isoformat(),
            "modal_price": round(price),
            "min_price":   round(price * 0.85),
            "max_price":   round(price * 1.15),
            "market":    "Lasalgaon",
            "district":  "Nashik",
            "state":     "Maharashtra",
            "commodity": COMMODITY,
            "variety":   "Other",
            "unit":      "Quintal",
        })
    return prices


def _generate_mock_current() -> list[dict]:
    random.seed(int(date.today().strftime("%Y%m%d")))
    base = random.randint(1400, 2600)
    result = []
    for m in MAJOR_MANDIS:
        mp = max(700, base + random.randint(-300, 300))
        result.append({
            **m,
            "commodity":   COMMODITY,
            "variety":     "Other",
            "modal_price": mp,
            "min_price":   round(mp * 0.85),
            "max_price":   round(mp * 1.15),
            "date":        date.today().isoformat(),
            "unit":        "Quintal",
        })
    return result


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_current_prices(district: Optional[str] = None) -> dict:
    """Fetch today's onion prices from AGMARKNET. Falls back to mock if API unavailable."""
    api_key = os.getenv("AGMARKNET_API_KEY", "")
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

    if not api_key or api_key == "YOUR_AGMARKNET_API_KEY" or use_mock:
        prices = _generate_mock_current()
        if district:
            prices = [p for p in prices if district.lower() in p["district"].lower()] or prices
        return {
            "source": "mock",
            "note":   "Set AGMARKNET_API_KEY in .env for live data",
            "prices": prices,
            "count":  len(prices),
        }

    try:
        # Get today's date and yesterday as filter (today may not be available yet)
        today_str = date.today().strftime("%d/%m/%Y")
        yesterday_str = (date.today() - timedelta(days=1)).strftime("%d/%m/%Y")

        params = {
            "api-key":             api_key,
            "format":              "json",
            "filters[Commodity]":  COMMODITY,
            "limit":               100,
            "offset":              0,
        }
        if district:
            params["filters[District]"] = district

        all_records: list[dict] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try today first, then yesterday
            for date_filter in [today_str, yesterday_str]:
                p = dict(params)
                p["filters[Arrival_Date]"] = date_filter
                resp = await client.get(AGMARKNET_BASE, params=p)
                resp.raise_for_status()
                data = resp.json()
                records = data.get("records", [])
                if records:
                    all_records = records
                    print(f"[MarketPrice] Got {len(records)} records for {date_filter}")
                    break

        if not all_records:
            # No recent records — get latest available without date filter
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(AGMARKNET_BASE, params=params)
                resp.raise_for_status()
                all_records = resp.json().get("records", [])

        prices = [_normalize_record(r) for r in all_records]
        prices = [p for p in prices if p["modal_price"] > 0]

        return {
            "source": "agmarknet",
            "prices": prices,
            "count":  len(prices),
        }

    except Exception as e:
        print(f"[MarketPrice] AGMARKNET fetch failed: {e}. Using mock data.")
        prices = _generate_mock_current()
        if district:
            prices = [p for p in prices if district.lower() in p["district"].lower()] or prices
        return {
            "source": "mock",
            "error":  str(e),
            "prices": prices,
            "count":  len(prices),
        }


async def fetch_price_history(days: int = 30) -> list[dict]:
    """
    Fetch onion price history for chart data.
    Queries AGMARKNET for Lasalgaon (the major reference mandi for onions).
    Fills missing days with realistic mock interpolation.
    """
    api_key = os.getenv("AGMARKNET_API_KEY", "")
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

    if not api_key or api_key == "YOUR_AGMARKNET_API_KEY" or use_mock:
        return _generate_mock_prices(days)

    # Fetch a batch from AGMARKNET — Lasalgaon as reference mandi
    history_by_date: dict[str, int] = {}
    try:
        params = {
            "api-key":            api_key,
            "format":             "json",
            "filters[Commodity]": COMMODITY,
            "filters[State]":     "Maharashtra",
            "filters[District]":  "Nashik",
            "limit":              100,
            "offset":             0,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(AGMARKNET_BASE, params=params)
            resp.raise_for_status()
            records = resp.json().get("records", [])
            for r in records:
                norm = _normalize_record(r)
                if norm["modal_price"] > 0 and norm["date"]:
                    history_by_date[norm["date"]] = norm["modal_price"]

        print(f"[MarketPrice] History: {len(history_by_date)} data points from AGMARKNET")
    except Exception as e:
        print(f"[MarketPrice] History fetch failed: {e}. Using full mock.")
        return _generate_mock_prices(days)

    # Merge with mock to fill any gaps
    mock = _generate_mock_prices(days)
    result = []
    for entry in mock:
        d = entry["date"]
        result.append({
            **entry,
            "modal_price": history_by_date.get(d, entry["modal_price"]),
            "source": "agmarknet" if d in history_by_date else "interpolated",
        })
    return result
