"""
Aloft AI Backend
----------------
Real ML-powered weather intelligence layer built on top of Open-Meteo's
free public data (no API key required).

What this actually does (no fake/hardcoded numbers):

1. /api/climate-normal
   Pulls the last N years of real historical daily data for a location
   from the Open-Meteo Archive API and computes the true climate normal
   (mean/min/max) for "today's" day-of-year. This is genuine statistics
   over real observed data.

2. /api/predict
   Trains a small RandomForestRegressor on that historical data
   (features: day-of-year cyclic encoding, recent rolling averages,
   humidity, precipitation) to predict tomorrow's max/min temperature.
   This is a real, freshly-trained model per request — not a canned
   number — and it also reports the official Open-Meteo forecast next
   to it so the two can be compared.

3. /api/anomaly
   Flags whether current conditions are a statistical outlier
   (heatwave / cold snap / unusually wet) using a z-score against the
   real climate normal computed in step 1. This solves a genuine
   problem: raw forecasts don't tell you "is this normal for here?"

4. /api/advisory
   Rule-based expert system (heat-index/WHO-style thresholds, real
   formulas) that turns raw numbers into actionable, human guidance:
   health risk, clothing, outdoor-activity and irrigation advice.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Deploy: Render / Railway (same as your other FastAPI projects).
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, date
import requests
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = FastAPI(title="Aloft AI Weather Backend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Simple in-memory cache: avoids retraining the model on every request for
# the same location. Keyed by (lat, lon) rounded to ~1km precision.
_predict_cache = {}
CACHE_TTL_SECONDS = 45 * 60  # 45 minutes — roughly matches how often forecasts change

# Generic cache reused by the raw data-fetch helpers below. Multiple
# endpoints (predict, anomaly, alerts) each independently needed the same
# 10-year historical dataset, tripling calls to Open-Meteo's Archive API
# for a single page load — enough to trip their free-tier rate limiting.
# Caching the raw fetch (not just the final /api/predict payload) means
# only the first caller per location actually hits the network.
_generic_cache = {}
_stale_fallback = {}  # never-expiring last-known-good data, per cache key


def _generic_cache_get(key, ttl):
    entry = _generic_cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _generic_cache_set(key, data):
    _generic_cache[key] = {"ts": time.time(), "data": data}
    # also keep a copy that never expires on its own TTL, purely as an
    # emergency fallback for when Open-Meteo is rate-limiting us and a
    # fresh fetch fails — better to serve slightly stale real data than
    # nothing at all.
    _stale_fallback[key] = data


def _stale_fallback_get(key):
    return _stale_fallback.get(key)


def _cache_get(key):
    entry = _predict_cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key, data):
    _predict_cache[key] = {"ts": time.time(), "data": data}


HISTORY_YEARS = 10        # how many past years of real data to pull (more years = better seasonal signal)
NORMAL_WINDOW_DAYS = 7     # +/- days around today's day-of-year for the "normal"

HISTORY_CACHE_TTL = 12 * 60 * 60   # 12 hours — historical archive data for a date range doesn't change intra-day
RECENT_CACHE_TTL = 20 * 60         # 20 minutes — "recent actual" days update daily, short cache is enough


# ---------------------------------------------------------------------------
# Data fetch helpers
# ---------------------------------------------------------------------------

def _get_with_retry(url, params, timeout=30, retries=2):
    """GET with a longer timeout and a couple of retries, so one slow
    response from the weather API doesn't take down the whole request.

    429 (rate limited) gets special handling: Render's free tier shares
    outbound IPs across many unrelated customers, so Open-Meteo's rate
    limit can trip even from light usage on our side. These are
    self-clearing within seconds, so we back off and retry rather than
    failing immediately."""
    last_err = None
    r = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))  # 2s, then 4s
                    continue
                return r  # out of retries, let caller report the 429
            return r
        except requests.exceptions.RequestException as e:
            last_err = e
    if r is not None:
        return r
    raise HTTPException(504, f"Weather service timed out after {retries + 1} attempts: {last_err}")


def _raise_for_bad_status(r, service_name):
    """Turn a non-200 upstream response into an accurate, honest error
    instead of a generic 'unavailable' — rate limiting is transient and
    self-clearing, a genuine outage is not, and the person deserves to
    know which one they're looking at."""
    if r.status_code == 429:
        raise HTTPException(
            503,
            f"{service_name} is temporarily rate-limited (this can happen on shared free-tier "
            f"hosting) — please wait ~30 seconds and try again."
        )
    raise HTTPException(502, f"{service_name} unavailable (status {r.status_code})")


def fetch_history(lat: float, lon: float, years: int = HISTORY_YEARS):
    """Fetch real daily historical weather from Open-Meteo Archive API.
    Pulls temperature, precipitation, humidity, pressure, wind and solar
    radiation — more signal for the model than temperature alone.
    Cached per-location for 12h since predict/anomaly/alerts all need this
    same dataset — without caching, a single page load could trigger 3x
    calls to the heaviest external endpoint."""
    cache_key = ("history", round(lat, 2), round(lon, 2), years)
    cached = _generic_cache_get(cache_key, HISTORY_CACHE_TTL)
    if cached is not None:
        return cached

    end = date.today() - timedelta(days=2)  # archive lags ~2 days behind
    start = end.replace(year=end.year - years)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "relative_humidity_2m_mean,weather_code,"
                 "surface_pressure_mean,wind_speed_10m_max,shortwave_radiation_sum",
        "timezone": "auto",
    }
    r = _get_with_retry(ARCHIVE_URL, params)
    if r.status_code != 200:
        stale = _stale_fallback_get(cache_key)
        if stale is not None:
            return stale
        _raise_for_bad_status(r, "Historical weather service")
    data = r.json()
    _generic_cache_set(cache_key, data)
    return data


def fetch_recent(lat: float, lon: float, past_days: int = 5):
    """The Archive API lags ~2 days behind 'today'. To build an accurate
    'recent trend' feature for tomorrow's prediction we instead pull the
    last few *actual* days from the Forecast API's past_days option,
    which reports real observed values, not a forecast, for those days.
    Cached for 20 minutes per location."""
    cache_key = ("recent", round(lat, 2), round(lon, 2), past_days)
    cached = _generic_cache_get(cache_key, RECENT_CACHE_TTL)
    if cached is not None:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "relative_humidity_2m_mean,surface_pressure_mean,"
                 "wind_speed_10m_max,shortwave_radiation_sum",
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": 1,
    }
    r = _get_with_retry(FORECAST_URL, params)
    if r.status_code != 200:
        stale = _stale_fallback_get(cache_key)
        if stale is not None:
            return stale
        _raise_for_bad_status(r, "Recent weather data service")
    data = r.json()
    _generic_cache_set(cache_key, data)
    return data


FORECAST_CACHE_TTL = 10 * 60  # 10 minutes — predict/anomaly/alerts/air-quality all call this separately


def fetch_forecast(lat: float, lon: float):
    cache_key = ("forecast", round(lat, 2), round(lon, 2))
    cached = _generic_cache_get(cache_key, FORECAST_CACHE_TTL)
    if cached is not None:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,surface_pressure,precipitation,is_day",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "timezone": "auto",
        "forecast_days": 3,
    }
    r = _get_with_retry(FORECAST_URL, params)
    if r.status_code != 200:
        stale = _stale_fallback_get(cache_key)
        if stale is not None:
            return stale
        _raise_for_bad_status(r, "Forecast service")
    data = r.json()
    _generic_cache_set(cache_key, data)
    return data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_frame(hist_json):
    daily = hist_json["daily"]
    dates = daily["time"]
    tmax = np.array(daily["temperature_2m_max"], dtype=float)
    tmin = np.array(daily["temperature_2m_min"], dtype=float)
    precip = np.array(daily["precipitation_sum"], dtype=float)
    humid = np.array(daily["relative_humidity_2m_mean"], dtype=float)
    pressure = np.array(daily["surface_pressure_mean"], dtype=float)
    wind = np.array(daily["wind_speed_10m_max"], dtype=float)
    radiation = np.array(daily["shortwave_radiation_sum"], dtype=float)

    doy = np.array([datetime.fromisoformat(d).timetuple().tm_yday for d in dates])
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    # rolling mean as a "recent trend" feature (shifted so no leakage)
    def rolling_prev(arr, window=3):
        out = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            out[i] = np.nanmean(arr[i - window:i])
        return out

    tmax_roll3 = rolling_prev(tmax, 3)
    tmin_roll3 = rolling_prev(tmin, 3)
    humid_roll3 = rolling_prev(humid, 3)
    pressure_roll3 = rolling_prev(pressure, 3)
    wind_roll3 = rolling_prev(wind, 3)
    radiation_roll3 = rolling_prev(radiation, 3)
    tmax_roll7 = rolling_prev(tmax, 7)
    pressure_roll7 = rolling_prev(pressure, 7)

    # pressure trend: is pressure rising or falling? (drop often precedes rain/storms)
    pressure_trend = np.full(len(pressure), np.nan)
    for i in range(3, len(pressure)):
        pressure_trend[i] = pressure[i - 1] - pressure[i - 3]

    # same day last year — a direct real-world seasonal anchor, not just sin/cos
    tmax_year_ago = np.full(len(tmax), np.nan)
    for i in range(365, len(tmax)):
        tmax_year_ago[i] = tmax[i - 365]

    return {
        "dates": dates,
        "doy": doy,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "tmax": tmax,
        "tmin": tmin,
        "precip": precip,
        "humid": humid,
        "pressure": pressure,
        "wind": wind,
        "radiation": radiation,
        "tmax_roll3": tmax_roll3,
        "tmin_roll3": tmin_roll3,
        "humid_roll3": humid_roll3,
        "pressure_roll3": pressure_roll3,
        "wind_roll3": wind_roll3,
        "radiation_roll3": radiation_roll3,
        "tmax_roll7": tmax_roll7,
        "pressure_roll7": pressure_roll7,
        "pressure_trend": pressure_trend,
        "tmax_year_ago": tmax_year_ago,
    }


FEATURE_NAMES = [
    "day_of_year_sin", "day_of_year_cos", "tmax_3day_avg", "tmin_3day_avg",
    "humidity_3day_avg", "pressure_3day_avg", "wind_3day_avg", "radiation_3day_avg",
    "tmax_7day_avg", "pressure_7day_avg", "pressure_trend_3day", "tmax_same_day_last_year",
]


def make_X(feat):
    return np.column_stack([
        feat["doy_sin"], feat["doy_cos"], feat["tmax_roll3"], feat["tmin_roll3"],
        feat["humid_roll3"], feat["pressure_roll3"], feat["wind_roll3"], feat["radiation_roll3"],
        feat["tmax_roll7"], feat["pressure_roll7"], feat["pressure_trend"], feat["tmax_year_ago"],
    ])


def climate_normal_for_doy(feat, target_doy, window=NORMAL_WINDOW_DAYS):
    """Real statistic: mean/std of tmax & tmin across all historical years
    within +/- window days of target_doy (handles year wraparound)."""
    doy = feat["doy"]
    diff = np.minimum(np.abs(doy - target_doy), 365 - np.abs(doy - target_doy))
    mask = diff <= window
    if mask.sum() < 5:
        mask = diff <= window * 2  # widen if too little data

    return {
        "sample_size": int(mask.sum()),
        "tmax_mean": float(np.nanmean(feat["tmax"][mask])),
        "tmax_std": float(np.nanstd(feat["tmax"][mask])),
        "tmin_mean": float(np.nanmean(feat["tmin"][mask])),
        "tmin_std": float(np.nanstd(feat["tmin"][mask])),
        "precip_mean": float(np.nanmean(feat["precip"][mask])),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "Aloft AI Weather Backend"}


@app.get("/api/climate-normal")
def climate_normal(lat: float = Query(...), lon: float = Query(...)):
    hist = fetch_history(lat, lon)
    feat = build_feature_frame(hist)
    today_doy = datetime.now().timetuple().tm_yday
    normal = climate_normal_for_doy(feat, today_doy)
    return {"location": {"lat": lat, "lon": lon}, "day_of_year": today_doy, "climate_normal": normal}


@app.get("/api/predict")
def predict(lat: float = Query(...), lon: float = Query(...)):
    """Trains a fresh RandomForest on real historical data for this exact
    location and predicts tomorrow's max/min temperature, alongside the
    official Open-Meteo forecast for comparison.

    Methodology notes:
    - Accuracy is evaluated with TimeSeriesSplit (5 folds), never a random
      shuffle-split — for a time series, a random split would let the
      model "see the future" and overstate accuracy.
    - Reports MAE, RMSE and R² from those out-of-fold predictions, plus a
      predicted-vs-actual series so you can see model performance over time,
      not just a single number.
    - The final tomorrow-prediction confidence interval comes from the
      actual spread of predictions across the RandomForest's individual
      trees (not an assumed distribution).
    - Results are cached per-location for 45 minutes so repeat requests
      don't retrain from scratch every time.
    """
    cache_key = (round(lat, 2), round(lon, 2))
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cache_hit": True}

    hist = fetch_history(lat, lon)
    feat = build_feature_frame(hist)

    X = make_X(feat)
    y_max = feat["tmax"]
    y_min = feat["tmin"]

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_max) & ~np.isnan(y_min)
    X_train, y_max_train, y_min_train = X[valid], y_max[valid], y_min[valid]

    if len(X_train) < 100:
        raise HTTPException(422, "Not enough historical data for this location to train a model")

    # ---- TimeSeriesSplit cross-validation: honest, leakage-free evaluation ----
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.full(len(y_max_train), np.nan)
    oof_dates_idx = np.arange(len(y_max_train))

    for train_idx, test_idx in tscv.split(X_train):
        fold_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42)
        fold_model.fit(X_train[train_idx], y_max_train[train_idx])
        oof_pred[test_idx] = fold_model.predict(X_train[test_idx])

    oof_mask = ~np.isnan(oof_pred)
    mae = float(mean_absolute_error(y_max_train[oof_mask], oof_pred[oof_mask]))
    rmse = float(np.sqrt(mean_squared_error(y_max_train[oof_mask], oof_pred[oof_mask])))
    r2 = float(r2_score(y_max_train[oof_mask], oof_pred[oof_mask]))

    # predicted-vs-actual series for charting: most recent ~200 out-of-fold points
    valid_dates = np.array(feat["dates"])[valid]
    oof_idx_valid = oof_dates_idx[oof_mask]
    chart_n = min(200, len(oof_idx_valid))
    chart_idx = oof_idx_valid[-chart_n:]
    predicted_vs_actual = [
        {"date": str(valid_dates[i]), "predicted": round(float(oof_pred[i]), 1), "actual": round(float(y_max_train[i]), 1)}
        for i in chart_idx
    ]

    # ---- final models trained on all available data ----
    model_max = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42)
    model_min = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42)
    model_max.fit(X_train, y_max_train)
    model_min.fit(X_train, y_min_train)

    # recent actual conditions (not lagged archive) to build tomorrow's feature vector
    recent = fetch_recent(lat, lon, past_days=8)
    rd = recent["daily"]
    r_tmax = np.array(rd["temperature_2m_max"], dtype=float)
    r_tmin = np.array(rd["temperature_2m_min"], dtype=float)
    r_humid = np.array(rd["relative_humidity_2m_mean"], dtype=float)
    r_pressure = np.array(rd["surface_pressure_mean"], dtype=float)
    r_wind = np.array(rd["wind_speed_10m_max"], dtype=float)
    r_radiation = np.array(rd["shortwave_radiation_sum"], dtype=float)

    tomorrow_doy = (datetime.now() + timedelta(days=1)).timetuple().tm_yday
    doy_sin = np.sin(2 * np.pi * tomorrow_doy / 365.25)
    doy_cos = np.cos(2 * np.pi * tomorrow_doy / 365.25)

    tmax_roll3 = float(np.nanmean(r_tmax[-3:]))
    tmin_roll3 = float(np.nanmean(r_tmin[-3:]))
    humid_roll3 = float(np.nanmean(r_humid[-3:]))
    pressure_roll3 = float(np.nanmean(r_pressure[-3:]))
    wind_roll3 = float(np.nanmean(r_wind[-3:]))
    radiation_roll3 = float(np.nanmean(r_radiation[-3:]))
    tmax_roll7 = float(np.nanmean(r_tmax[-7:])) if len(r_tmax) >= 7 else tmax_roll3
    pressure_roll7 = float(np.nanmean(r_pressure[-7:])) if len(r_pressure) >= 7 else pressure_roll3
    pressure_trend = float(r_pressure[-2] - r_pressure[-4]) if len(r_pressure) >= 4 else 0.0
    tmax_year_ago = float(feat["tmax"][-365]) if len(feat["tmax"]) >= 365 else float(np.nanmean(feat["tmax"]))

    x_tomorrow = np.array([[doy_sin, doy_cos, tmax_roll3, tmin_roll3, humid_roll3,
                             pressure_roll3, wind_roll3, radiation_roll3,
                             tmax_roll7, pressure_roll7, pressure_trend, tmax_year_ago]])

    pred_max = float(model_max.predict(x_tomorrow)[0])
    pred_min = float(model_min.predict(x_tomorrow)[0])

    # ---- confidence interval from the real spread across the forest's trees ----
    tree_preds_max = np.array([t.predict(x_tomorrow)[0] for t in model_max.estimators_])
    tree_std_max = float(np.std(tree_preds_max))
    ci_low, ci_high = float(np.percentile(tree_preds_max, 10)), float(np.percentile(tree_preds_max, 90))

    tree_preds_min = np.array([t.predict(x_tomorrow)[0] for t in model_min.estimators_])
    tree_std_min = float(np.std(tree_preds_min))

    importances = sorted(
        zip(FEATURE_NAMES, model_max.feature_importances_.round(3).tolist()),
        key=lambda t: -t[1]
    )

    # official forecast for the same day, for direct comparison
    fc = fetch_forecast(lat, lon)
    official_max = fc["daily"]["temperature_2m_max"][1]
    official_min = fc["daily"]["temperature_2m_min"][1]

    payload = {
        "location": {"lat": lat, "lon": lon},
        "target_date": (datetime.now() + timedelta(days=1)).date().isoformat(),
        "ai_model_prediction": {
            "tmax_c": round(pred_max, 1),
            "tmin_c": round(pred_min, 1),
            "tmax_confidence_interval": {"low": round(ci_low, 1), "high": round(ci_high, 1), "std": round(tree_std_max, 2)},
            "tmin_confidence_std": round(tree_std_min, 2),
            "holdout_mae_c": round(mae, 2),
            "holdout_rmse_c": round(rmse, 2),
            "holdout_r2": round(r2, 3),
            "trained_on_days": int(len(X_train)),
            "cv_method": f"TimeSeriesSplit ({n_splits} folds, no shuffle — no future leakage)",
            "feature_importances": importances,
        },
        "predicted_vs_actual": predicted_vs_actual,
        "official_forecast": {
            "tmax_c": official_max,
            "tmin_c": official_min,
        },
        "agreement_delta_c": round(abs(pred_max - official_max), 1),
        "cache_hit": False,
        "note": "AI model is trained fresh per location on real historical archive data across "
                "12 features. Accuracy (MAE/RMSE/R²) comes from TimeSeriesSplit cross-validation "
                "on unseen days — an honest estimate, not a marketing number. The confidence "
                "interval reflects real disagreement across the forest's individual trees. It "
                "still won't beat Open-Meteo's physics-based forecast, which models incoming "
                "weather systems this approach can't see."
    }

    _cache_set(cache_key, payload)
    return payload


@app.get("/api/anomaly")
def anomaly(lat: float = Query(...), lon: float = Query(...)):
    """Z-score anomaly detection: is today's real observed weather unusual
    for this place and time of year?"""
    hist = fetch_history(lat, lon)
    feat = build_feature_frame(hist)
    fc = fetch_forecast(lat, lon)

    today_doy = datetime.now().timetuple().tm_yday
    normal = climate_normal_for_doy(feat, today_doy)

    current_temp = fc["current"]["temperature_2m"]
    z_max = (current_temp - normal["tmax_mean"]) / (normal["tmax_std"] or 1)

    if z_max >= 2:
        label, severity = "Heatwave conditions", "high"
    elif z_max >= 1.2:
        label, severity = "Warmer than usual", "moderate"
    elif z_max <= -2:
        label, severity = "Cold snap conditions", "high"
    elif z_max <= -1.2:
        label, severity = "Colder than usual", "moderate"
    else:
        label, severity = "Within normal range", "none"

    return {
        "location": {"lat": lat, "lon": lon},
        "current_temp_c": current_temp,
        "climate_normal_max_c": round(normal["tmax_mean"], 1),
        "climate_normal_std_c": round(normal["tmax_std"], 1),
        "z_score": round(float(z_max), 2),
        "label": label,
        "severity": severity,
        "sample_years": HISTORY_YEARS,
    }


@app.get("/api/advisory")
def advisory(
    temp: float = Query(...),
    humidity: float = Query(...),
    wind: float = Query(...),
    precipitation_prob: float = Query(0),
    weather_code: int = Query(0),
):
    """Rule-based expert system using real heat-index style thresholds.
    Not ML, but genuinely computed from the inputs — not hardcoded text."""

    advisories = []

    # Heat index approximation (simplified NOAA formula, valid warm range)
    if temp >= 27:
        hi = (
            -8.784695 + 1.61139411 * temp + 2.338549 * humidity
            - 0.14611605 * temp * humidity - 0.012308094 * temp ** 2
            - 0.016424828 * humidity ** 2 + 0.002211732 * temp ** 2 * humidity
            + 0.00072546 * temp * humidity ** 2 - 0.000003582 * temp ** 2 * humidity ** 2
        )
    else:
        hi = temp

    if hi >= 41:
        advisories.append({"type": "health", "level": "danger",
                            "text": f"Heat index ~{hi:.0f}°C — heat stroke risk. Avoid strenuous outdoor activity, hydrate frequently."})
    elif hi >= 32:
        advisories.append({"type": "health", "level": "caution",
                            "text": f"Heat index ~{hi:.0f}°C — fatigue possible with prolonged exposure. Take shade breaks, drink water."})

    if temp <= 5:
        advisories.append({"type": "health", "level": "caution",
                            "text": "Cold conditions — wear layered clothing, cover extremities."})

    # clothing
    if temp >= 30:
        advisories.append({"type": "clothing", "level": "info", "text": "Light, breathable fabrics; sun protection recommended."})
    elif 18 <= temp < 30:
        advisories.append({"type": "clothing", "level": "info", "text": "Comfortable in light layers."})
    elif 8 <= temp < 18:
        advisories.append({"type": "clothing", "level": "info", "text": "A jacket or sweater is advisable."})
    else:
        advisories.append({"type": "clothing", "level": "info", "text": "Heavy coat, gloves and layering recommended."})

    # wind
    if wind >= 40:
        advisories.append({"type": "wind", "level": "caution", "text": f"Strong winds ({wind:.0f} km/h) — secure loose objects, caution for two-wheelers."})

    # rain / irrigation & travel
    if precipitation_prob >= 60:
        advisories.append({"type": "activity", "level": "info", "text": f"{precipitation_prob:.0f}% chance of rain — carry protection, and hold off on irrigation today."})
    elif precipitation_prob < 20 and temp >= 25:
        advisories.append({"type": "agriculture", "level": "info", "text": "Low rain chance and warm conditions — soil may dry faster; consider watering plants/crops."})

    if weather_code >= 95:
        advisories.append({"type": "safety", "level": "danger", "text": "Thunderstorms expected — avoid open areas and tall isolated trees."})

    if not advisories:
        advisories.append({"type": "general", "level": "info", "text": "Conditions are unremarkable — good day for most outdoor plans."})

    return {"heat_index_c": round(hi, 1), "advisories": advisories}


@app.get("/api/air-quality")
def air_quality(lat: float = Query(...), lon: float = Query(...)):
    """Real air quality (US AQI, PM2.5, PM10) from Open-Meteo's Air Quality
    API, plus UV index from the standard forecast API. Both free, no key."""
    aqi_params = {
        "latitude": lat, "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",
        "timezone": "auto",
    }
    aqi_r = _get_with_retry(AIR_QUALITY_URL, aqi_params)
    if aqi_r.status_code != 200:
        _raise_for_bad_status(aqi_r, "Air quality service")
    aqi = aqi_r.json().get("current", {})

    uv_params = {
        "latitude": lat, "longitude": lon,
        "daily": "uv_index_max", "timezone": "auto", "forecast_days": 1,
    }
    uv_r = _get_with_retry(FORECAST_URL, uv_params)
    uv_max = uv_r.json()["daily"]["uv_index_max"][0] if uv_r.status_code == 200 else None

    us_aqi = aqi.get("us_aqi")
    if us_aqi is None:
        aqi_label, aqi_level = "Unavailable", "none"
    elif us_aqi <= 50:
        aqi_label, aqi_level = "Good", "none"
    elif us_aqi <= 100:
        aqi_label, aqi_level = "Moderate", "info"
    elif us_aqi <= 150:
        aqi_label, aqi_level = "Unhealthy for sensitive groups", "moderate"
    elif us_aqi <= 200:
        aqi_label, aqi_level = "Unhealthy", "high"
    else:
        aqi_label, aqi_level = "Very unhealthy", "high"

    if uv_max is None:
        uv_label = "Unavailable"
    elif uv_max < 3:
        uv_label = "Low"
    elif uv_max < 6:
        uv_label = "Moderate"
    elif uv_max < 8:
        uv_label = "High"
    elif uv_max < 11:
        uv_label = "Very high"
    else:
        uv_label = "Extreme"

    return {
        "us_aqi": us_aqi, "aqi_label": aqi_label, "aqi_level": aqi_level,
        "pm2_5": aqi.get("pm2_5"), "pm10": aqi.get("pm10"),
        "uv_index_max": uv_max, "uv_label": uv_label,
    }


@app.get("/api/alerts")
def alerts(lat: float = Query(...), lon: float = Query(...)):
    """Data-driven alerts synthesized from the real anomaly detector, heat
    index, and forecast codes. NOTE: these are derived from open weather
    data, not official government warnings — labeled as such so they're
    never confused with a real meteorological agency alert."""
    hist = fetch_history(lat, lon)
    feat = build_feature_frame(hist)
    fc = fetch_forecast(lat, lon)
    cur = fc["current"]

    today_doy = datetime.now().timetuple().tm_yday
    normal = climate_normal_for_doy(feat, today_doy)
    z_max = (cur["temperature_2m"] - normal["tmax_mean"]) / (normal["tmax_std"] or 1)

    alert_list = []
    if z_max >= 2:
        alert_list.append({"severity": "high", "title": "Heatwave signal",
                            "text": f"Current temperature is {z_max:.1f} standard deviations above the {HISTORY_YEARS}-year normal for this date."})
    elif z_max <= -2:
        alert_list.append({"severity": "high", "title": "Cold snap signal",
                            "text": f"Current temperature is {abs(z_max):.1f} standard deviations below the {HISTORY_YEARS}-year normal for this date."})

    if cur["wind_speed_10m"] >= 40:
        alert_list.append({"severity": "moderate", "title": "Strong wind",
                            "text": f"Sustained wind near {cur['wind_speed_10m']:.0f} km/h."})

    if cur["weather_code"] >= 95:
        alert_list.append({"severity": "high", "title": "Thunderstorm activity",
                            "text": "Thunderstorms reported in current conditions."})

    if fc["daily"]["precipitation_probability_max"][0] >= 80:
        alert_list.append({"severity": "moderate", "title": "Heavy rain likely",
                            "text": f"{fc['daily']['precipitation_probability_max'][0]}% chance of precipitation today."})

    return {"alerts": alert_list, "source": "Derived from Open-Meteo data + local anomaly detection — not an official government alert."}