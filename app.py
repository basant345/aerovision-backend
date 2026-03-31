from flask import Flask, request, jsonify
import os
import numpy as np
import requests
from datetime import datetime, timedelta
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd
from zoneinfo import ZoneInfo
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time

IST = ZoneInfo("Asia/Kolkata")

# ── EnvAlert cache & helpers (permanent fix) ─────────────────────────────────
_envalert_cache = {"data": None, "ts": 0}
_CACHE_TTL = 300  # 5 minutes — serve cached data if EnvAlert blocks

ENVALERT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://erc.mp.gov.in/EnvAlert/",
    "Origin": "https://erc.mp.gov.in",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}

def fetch_envalert_all_with_cache():
    """Fetch ALL stations with Indian headers, 3 retries, 5-min cache."""
    now = time.time()
    if _envalert_cache["data"] and (now - _envalert_cache["ts"]) < _CACHE_TTL:
        print("[EnvAlert] Serving from cache", flush=True)
        return _envalert_cache["data"]
    url = "https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id=ALL"
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=ENVALERT_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                _envalert_cache["data"] = data
                _envalert_cache["ts"] = now
                print(f"[EnvAlert] Fetched {len(data)} stations (attempt {attempt+1})", flush=True)
                return data
        except Exception as e:
            last_err = e
            print(f"[EnvAlert] Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    if _envalert_cache["data"]:
        print("[EnvAlert] All retries failed — serving stale cache", flush=True)
        return _envalert_cache["data"]
    print(f"[EnvAlert] All retries failed, no cache: {last_err}", flush=True)
    return None


def fetch_envalert_station_with_retry(station_id):
    """Fetch single station — first from ALL cache, then individual with retry."""
    # Try cache first — avoids individual station blocks entirely
    cached = fetch_envalert_all_with_cache()
    if cached:
        for st in cached:
            if str(st.get("station_id")) == str(station_id):
                return st
    # Fallback: individual fetch with Indian headers
    url = f"https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id={station_id}"
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=ENVALERT_HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return data
        except Exception as e:
            print(f"[EnvAlert] Station {station_id} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(1)
    return None


# Hardcoded MP city coordinates — instant, no API needed
MP_CITY_COORDS = {
    "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Jabalpur": (23.1815, 79.9864), "Gwalior": (26.2183, 78.1828),
    "Ujjain": (23.1765, 75.7885), "Sagar": (23.8388, 78.7378),
    "Dewas": (22.9623, 76.0552), "Satna": (24.5694, 80.8322),
    "Ratlam": (23.3315, 75.0367), "Rewa": (24.5362, 81.2956),
    "Katni": (23.8333, 80.4000), "Singrauli": (24.1997, 82.6739),
    "Khandwa": (21.8245, 76.3490), "Khargone": (21.8234, 75.6127),
    "Damoh": (23.8333, 79.4333), "Neemuch": (24.4760, 74.8693),
    "Panna": (24.7167, 80.1833), "Pithampur": (22.6167, 75.6833),
    "Narsinghpur": (22.9497, 79.1942), "Maihar": (24.2667, 80.7667),
    "Mandideep": (23.1000, 77.5333), "Betul": (21.9000, 77.9000),
    "Anuppur": (23.1028, 81.6850), "Chhindwara": (22.0574, 78.9382),
    "Bhind": (26.5613, 78.7876), "Morena": (26.4944, 77.9983),
    "Shivpuri": (25.4231, 77.6578), "Chhatarpur": (24.9167, 79.5833),
    "Seoni": (22.0856, 79.5414), "Balaghat": (21.8133, 80.1860),
    "Raisen": (23.3314, 77.7887), "Rajgarh": (24.0167, 76.7333),
    "Shajapur": (23.4268, 76.2774), "Dhar": (22.5985, 75.2985),
    "Barwani": (22.0333, 74.9000), "Sidhi": (24.4167, 81.8833),
    "Umaria": (23.5245, 80.8380), "Dindori": (22.9437, 81.0790),
    "Ashoknagar": (24.5750, 77.7283), "Guna": (24.6481, 77.3152),
    "Nagda": (23.4500, 75.4167), "Itarsi": (22.6167, 77.7667),
    "Shahdol": (23.2833, 81.3500), "Mandsaur": (24.0765, 75.0711),
    "Narmadapuram": (22.7533, 77.7125), "Vidisha": (23.5251, 77.8082),
    "Sehore": (23.2006, 77.0845), "CTSDF": (23.2599, 77.4126),
}
# ─────────────────────────────────────────────────────────────────────────────


# Disable GPU for CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
# Allow all origins for React Native app compatibility
# React Native doesn't send traditional browser origins
CORS(app, 
     resources={r"/*": {
         "origins": ["https://airqualitycities.iiti.ac.in", "http://localhost:8080"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "Accept"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False
     }}
)

api_key = "701cf10ad3df9b6f5f58f40bfba7e837"

# Add after_request handler to ensure CORS headers

TARGET_POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

POLLUTANT_API_MAP = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "o3": "ozone",
    "co": "carbon_monoxide"
}

# City to Station ID mapping
CITY_STATIONS = {
    "Anuppur": [18],
    "Betul": [22],
    "Bhopal": [27, 34, 10],
    "CTSDF": [44],
    "Damoh": [7],
    "Dewas": [23, 3],
    "Gwalior": [16, 29, 30, 15],
    "Indore": [31, 36, 35, 37, 40, 38, 33, 13],
    "Jabalpur": [41, 12, 42, 43],
    "Katni": [11, 19],
    "Khandwa": [32],
    "Khargone": [25],
    "Maihar": [8],
    "Mandideep": [5],
    "Narsinghpur": [26],
    "Neemuch": [17],
    "Panna": [39],
    "Pithampur": [1],
    "Ratlam": [9],
    "Rewa": [20, 21],
    "Sagar": [28, 14],
    "Satna": [6],
    "Singrauli": [4, 24],
    "Ujjain": [2]
}

# EnvAlert API pollutant mapping to API response keys
ENVALERT_POLLUTANT_MAP = {
    "pm2_5": ("pm25", "pm25_subindex"),
    "pm10": ("pm10", "pm10_subindex"),
    "no2": ("nox", "nox_subindex"),  # NOx is used as NO2 proxy
    "so2": ("so2", "so2_subindex"),
    "o3": ("ozone", "ozone_subindex"),
    "co": ("co", "co_subindex")
}

WEATHER_COLS = [
    'temperature_2m', 'dew_point_2m', 'precipitation', 'wind_speed_10m',
    'cloud_cover', 'surface_pressure', 'vapour_pressure_deficit',
    'boundary_layer_height', 'sunshine_duration'
]

AQI_BREAKPOINTS = {
    'pm2_5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, np.inf, 401, 500)],
    'pm10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, np.inf, 401, 500)],
    'no2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, np.inf, 401, 500)],
    'o3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, np.inf, 401, 500)],
    'co': [(0, 1000, 0, 50), (1001, 2000, 51, 100), (2001, 10000, 101, 200), (10001, 17000, 201, 300), (17001, 34000, 301, 400), (34001, np.inf, 401, 500)],
    'so2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, np.inf, 401, 500)]
}

AQI_CATEGORIES = {
    (0, 50): 'Good',
    (51, 100): 'Satisfactory',
    (101, 200): 'Moderately Polluted',
    (201, 300): 'Poor',
    (301, 400): 'Very Poor',
    (401, 500): 'Severe'
}


# new added
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km



# new added
@lru_cache(maxsize=100)
def get_city_latlon(city):
    return get_city_coordinates(city)

def find_nearest_city(city_name):
    base_lat, base_lon = get_city_coordinates(city_name)
    if not base_lat:
        return None

    nearest_city = None
    min_dist = float("inf")

    for city in CITY_STATIONS.keys():
        if city.lower() == city_name.lower():
            continue

        lat, lon = get_city_latlon(city)
        if not lat:
            continue

        dist = haversine(base_lat, base_lon, lat, lon)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city

    return nearest_city

# new added
def get_fallback_data_from_nearest_city(city_name):
    nearest_city = find_nearest_city(city_name)

    if not nearest_city:
        print("❌ No nearest city found", flush=True)
        return None

    print(f"⚠ Using fallback city: {nearest_city}", flush=True)

    station_ids = CITY_STATIONS[nearest_city][:2]  # only 2 stations

    pollutant_values = {p: [] for p in TARGET_POLLUTANTS}
    pollutant_aqis = {p: [] for p in TARGET_POLLUTANTS}

    for station_id in station_ids:
        data = fetch_envalert_current_aqi(station_id)
        if not data:
            continue

        for pollutant in TARGET_POLLUTANTS:
            value_key, aqi_key = ENVALERT_POLLUTANT_MAP[pollutant]

            try:
                val = float(data.get(value_key))
                pollutant_values[pollutant].append(val)
            except:
                pass

            try:
                aqi = float(data.get(aqi_key))
                pollutant_aqis[pollutant].append(aqi)
            except:
                pass

    result = {}
    for p in TARGET_POLLUTANTS:
        if pollutant_values[p]:
            result[p] = {
                "value": round(sum(pollutant_values[p]) / len(pollutant_values[p]), 2),
                "aqi": round(sum(pollutant_aqis[p]) / len(pollutant_aqis[p]), 0)
            }

    return result if result else None


def get_aqi_sub_index(C, pollutant):
    if pd.isna(C): return np.nan
    breakpoints = AQI_BREAKPOINTS.get(pollutant)
    for B_low, B_high, I_low, I_high in breakpoints:
        if B_low <= C <= B_high:
            sub_index = ((I_high - I_low) / (B_high - B_low)) * (C - B_low) + I_low
            return min(round(sub_index), 500)
    return np.nan

def get_category_info(aqi):
    for (low, high), cat in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            color_map = {
                'Good': 'green',
                'Satisfactory': 'yellow',
                'Moderately Polluted': 'orange',
                'Poor': 'red',
                'Very Poor': 'purple',
                'Severe': 'maroon'
            }
            return cat, f"{cat} air quality.", color_map.get(cat, "gray")
    return "Out of Range", "AQI beyond measurable limits.", "gray"

# Lazy load models on demand
models = {}
models_loaded = {}

def get_model(pollutant):
    """Lazy load model when needed"""
    if pollutant not in models_loaded:
        try:
            path = os.path.join(os.path.dirname(__file__), f"best_cnn_{pollutant}.keras")
            models[pollutant] = load_model(path)
            models_loaded[pollutant] = True
            print(f"✅ Loaded model for {pollutant}", flush=True)
        except Exception as e:
            print(f"Model load error for {pollutant}: {e}", flush=True)
            models[pollutant] = None
            models_loaded[pollutant] = False
    return models.get(pollutant)

@lru_cache(maxsize=200)
def get_city_coordinates(city_name):
    # Instant lookup from hardcoded MP coords — no API needed
    for key, coords in MP_CITY_COORDS.items():
        if key.lower() == city_name.lower():
            return coords
    # Fallback to OpenWeatherMap only for unknown cities
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},Madhya Pradesh,IN&limit=1&appid={api_key}"
        res = requests.get(url, timeout=8)
        data = res.json()
        if data and isinstance(data, list):
            lat = data[0].get("lat")
            lon = data[0].get("lon")
            if lat is not None and lon is not None:
                return lat, lon
    except Exception as e:
        print(f"[get_city_coordinates] fallback failed for {city_name}: {e}", flush=True)
    return None, None

def fetch_envalert_current_aqi(station_id):
    """Fetch current AQI data for a specific station — with retry and Indian headers"""
    return fetch_envalert_station_with_retry(station_id)

def get_today_data_from_envalert(city_name):
    """
    Fetch today's air quality data from EnvAlert API for the given city.
    Returns average values and AQIs if stations found, or None if no data available.
    """
    try:
        # Normalize city name for matching (case-insensitive)
        city_key = None
        for key in CITY_STATIONS.keys():
            if key.lower() == city_name.lower():
                city_key = key
                break
        
        if not city_key:
            print(f"City '{city_name}' not found in CITY_STATIONS mapping", flush=True)
            return None
        
        station_ids = CITY_STATIONS[city_key]
        print(f"Found {len(station_ids)} stations for {city_key}: {station_ids}", flush=True)
        
        # Fetch data for each station in parallel
        all_pollutant_values = {p: [] for p in TARGET_POLLUTANTS}
        all_pollutant_aqis = {p: [] for p in TARGET_POLLUTANTS}
        
        # Get station data from ALL-stations cache (avoids individual blocks)
        all_cached = fetch_envalert_all_with_cache()
        station_data_list = []
        if all_cached:
            cached_map = {str(st.get("station_id")): st for st in all_cached}
            station_data_list = [cached_map.get(str(sid)) for sid in station_ids]
        # Fallback: individual fetch if cache empty
        if not any(station_data_list):
            with ThreadPoolExecutor(max_workers=min(len(station_ids), 5)) as executor:
                station_data_list = list(executor.map(fetch_envalert_current_aqi, station_ids))
        
        for station_data in station_data_list:
            if not station_data:
                continue
            
            print(f"Station data: {station_data.get('station_name', 'Unknown')}", flush=True)
            
            # Extract pollutant values and their AQIs
            for pollutant in TARGET_POLLUTANTS:
                value_key, aqi_key = ENVALERT_POLLUTANT_MAP.get(pollutant)
                
                # Get concentration value
                value = station_data.get(value_key)
                if value is not None and value != '' and value != 'null':
                    try:
                        all_pollutant_values[pollutant].append(float(value))
                    except (ValueError, TypeError):
                        pass
                
                # Get AQI sub-index
                aqi_value = station_data.get(aqi_key)
                if aqi_value is not None and aqi_value != '' and aqi_value != 'null':
                    try:
                        all_pollutant_aqis[pollutant].append(float(aqi_value))
                    except (ValueError, TypeError):
                        pass
        
        # Calculate averages
        result = {}
        for pollutant in TARGET_POLLUTANTS:
            values = all_pollutant_values[pollutant]
            aqis = all_pollutant_aqis[pollutant]
            
            if values and aqis:
                avg_value = sum(values) / len(values)
                avg_aqi = sum(aqis) / len(aqis)
                result[pollutant] = {
                    'value': avg_value,
                    'aqi': round(avg_aqi)
                }
                print(f"EnvAlert average {pollutant}: value={avg_value:.2f}, aqi={avg_aqi:.0f} (from {len(values)} stations)", flush=True)
        
        # If we got at least some data, return it
        if result:
            return result
        else:
            print(f"No valid pollutant data found for {city_name}", flush=True)
            return None
            
    except Exception as e:
        print(f"Error in get_today_data_from_envalert: {e}", flush=True)
        return None

def fetch_pollutant_series(lat, lon, pollutant):
    try:
        end_datetime_ist = datetime.now(IST).replace(minute=0, second=0, microsecond=0)
        start_datetime = end_datetime_ist - timedelta(hours=71)

        start_date = start_datetime.date().strftime("%Y-%m-%d")
        end_date = end_datetime_ist.date().strftime("%Y-%m-%d")

        api_field = POLLUTANT_API_MAP[pollutant]

        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly={api_field}&timezone=Asia%2FKolkata"
        )
        data = None
        for _attempt in range(3):
            try:
                response = requests.get(url, timeout=20)
                data = response.json()
                break
            except Exception as _e:
                print(f"[{pollutant}] pollutant fetch attempt {_attempt+1} failed: {_e}", flush=True)
                time.sleep(1)
        if not data:
            return [], []
        values = data["hourly"].get(api_field, [])
        timestamps = data["hourly"].get("time", [])

        # Align last 72 hours ending at current hour
        current_hour = datetime.now(IST).replace(minute=0, second=0, microsecond=0)
        current_index = None
        for i, ts in enumerate(timestamps):
            ts_dt = datetime.fromisoformat(ts).replace(tzinfo=IST)
            if ts_dt >= current_hour:
                current_index = i
                break
        if current_index is None:
            current_index = len(timestamps) - 1

        start_index = max(0, current_index - 71)
        series = values[start_index:current_index+1]
        ts_series = timestamps[start_index:current_index+1]

        return series, ts_series
    except Exception as e:
        print(f"[{pollutant.upper()}] Pollutant fetch error:", e, flush=True)
        return [], []

def fetch_weather_series(lat, lon):
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=4)
    weather_params = ",".join(WEATHER_COLS)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={weather_params}"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=20)
            data = response.json()
            hourly = data["hourly"]
            result = [[hourly[col][i] for col in WEATHER_COLS] for i in range(len(hourly['time']))]
            if result:
                return result
        except Exception as e:
            print(f"[fetch_weather_series] Attempt {attempt+1} failed: {e}", flush=True)
            import time as t; t.sleep(1)
    return []

def calculate_errors(envalert_today_data, model_predictions_for_error):
    """
    Calculate errors: avg of all stations - predicted by model
    Returns errors for pm2.5 conc, pm2.5 aqi, pm10 conc, pm10 aqi, and overall aqi
    """
    errors = {}
    
    try:
        # PM2.5 errors
        if envalert_today_data and "pm2_5" in envalert_today_data and "pm2_5" in model_predictions_for_error:
            api_pm25_value = envalert_today_data["pm2_5"]["value"]
            api_pm25_aqi = envalert_today_data["pm2_5"]["aqi"]
            model_pm25_value = model_predictions_for_error["pm2_5"]["value"]
            model_pm25_aqi = model_predictions_for_error["pm2_5"]["aqi"]
            
            errors["pm2_5_concentration"] = round(api_pm25_value - model_pm25_value, 2)
            errors["pm2_5_aqi"] = round(api_pm25_aqi - model_pm25_aqi, 2)
            
            print(f"PM2.5 - API: {api_pm25_value}, Model: {model_pm25_value}, Error: {errors['pm2_5_concentration']}", flush=True)
            print(f"PM2.5 AQI - API: {api_pm25_aqi}, Model: {model_pm25_aqi}, Error: {errors['pm2_5_aqi']}", flush=True)
        
        # PM10 errors
        if envalert_today_data and "pm10" in envalert_today_data and "pm10" in model_predictions_for_error:
            api_pm10_value = envalert_today_data["pm10"]["value"]
            api_pm10_aqi = envalert_today_data["pm10"]["aqi"]
            model_pm10_value = model_predictions_for_error["pm10"]["value"]
            model_pm10_aqi = model_predictions_for_error["pm10"]["aqi"]
            
            errors["pm10_concentration"] = round(api_pm10_value - model_pm10_value, 2)
            errors["pm10_aqi"] = round(api_pm10_aqi - model_pm10_aqi, 2)
            
            print(f"PM10 - API: {api_pm10_value}, Model: {model_pm10_value}, Error: {errors['pm10_concentration']}", flush=True)
            print(f"PM10 AQI - API: {api_pm10_aqi}, Model: {model_pm10_aqi}, Error: {errors['pm10_aqi']}", flush=True)
        
        # Overall AQI error - calculate from all available pollutants
        if envalert_today_data and model_predictions_for_error:
            # Get all AQI values from EnvAlert (excluding o3)
            envalert_aqis = []
            for pollutant in TARGET_POLLUTANTS:
                if pollutant != "o3" and pollutant in envalert_today_data:
                    envalert_aqis.append(envalert_today_data[pollutant]["aqi"])
            
            # Get all AQI values from model predictions (excluding o3)
            model_aqis = []
            for pollutant in TARGET_POLLUTANTS:
                if pollutant != "o3" and pollutant in model_predictions_for_error:
                    model_aqis.append(model_predictions_for_error[pollutant]["aqi"])
            
            if envalert_aqis and model_aqis:
                api_overall_aqi = max(envalert_aqis)
                model_overall_aqi = max(model_aqis)
                errors["overall_aqi"] = round(api_overall_aqi - model_overall_aqi, 2)
                
                print(f"Overall AQI - API: {api_overall_aqi}, Model: {model_overall_aqi}, Error: {errors['overall_aqi']}", flush=True)
        
        print(f"Calculated errors: {errors}", flush=True)
        
    except Exception as e:
        print(f"Error calculating errors: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    return errors

def predict_pollutant(pollutant, data, weather_data, timestamps, start_day=1):
    """
    Predict pollutant values starting from a given day.
    start_day=0 for today, start_day=1 for tomorrow onwards
    """
    try:
        model = get_model(pollutant)  # Use lazy loading
        if not model or len(data) < 72:
            return []

        # Latest weather features
        weather_features = weather_data[-1][:9] if weather_data else [0] * 9

        # Prepare initial sequence for model
        seq = [0.0] + data[-72:] + weather_features
        sequence = np.array(seq).reshape((1, 82, 1))

        results = []

        # Determine previous day date
        today_date = datetime.now(IST).date()
        prev_date = today_date - timedelta(days=1)

        # Get indices of previous day
        prev_day_indices = [i for i, ts in enumerate(timestamps) if datetime.fromisoformat(ts).date() == prev_date]

        if not prev_day_indices:
            print(f"No previous day data found for {prev_date}")
            return []

        for i in range(start_day, 7):
            pred_val = float(abs(model.predict(sequence, verbose=0)[0, 0]))

            # Find current hour of previous day
            hour_now = datetime.now(IST).hour
            prev_hour_index = next((idx for idx in prev_day_indices if datetime.fromisoformat(timestamps[idx]).hour == hour_now), None)

            if prev_hour_index is None:
                prev_hour_index = prev_day_indices[-1]

            # Take previous 23 hours from previous day
            start_index = max(prev_hour_index - 23, 0)
            last_23_hours = [data[j] for j in range(start_index, prev_hour_index)]

            # Combine with predicted value
            values_avg = last_23_hours + [pred_val]
            C_avg = sum(values_avg) / len(values_avg)

            aqi = get_aqi_sub_index(C_avg, pollutant)
            category, warning, color = get_category_info(aqi)

            date = (datetime.utcnow() + timedelta(days=i)).strftime("%Y-%m-%d")
            day = "Today" if i == 0 else "Tomorrow" if i == 1 else (datetime.utcnow() + timedelta(days=i)).strftime("%d %b")

            results.append({
                "day": day,
                "date": date,
                "value": round(pred_val, 2),
                "aqi": int(aqi) if not pd.isna(aqi) else 0,
                "category": category,
                "warning": warning,
                "color": color
            })

            # Update sequence for next prediction
            sequence[0, -1, 0] = pred_val
            sequence = np.roll(sequence, -1, axis=1)

        return results

    except Exception as e:
        print(f"Prediction error for {pollutant}: {e}", flush=True)
        return []


def getAvgOfAllStationsValues():
    try:
        stations = fetch_envalert_all_with_cache()
        if not stations:
            return None

        if not isinstance(stations, list):
            raise ValueError("Unexpected API response format")

        pm25_values = []
        pm10_values = []

        for station in stations:
            # PM2.5
            pm25 = station.get("pm25")
            if pm25 not in (None, "", "ID"):
                try:
                    pm25_values.append(float(pm25))
                except ValueError:
                    pass

            # PM10
            pm10 = station.get("pm10")
            if pm10 not in (None, "", "ID"):
                try:
                    pm10_values.append(float(pm10))
                except ValueError:
                    pass

        return {
            "pm25_avg": round(sum(pm25_values) / len(pm25_values), 2) if pm25_values else None,
            "pm10_avg": round(sum(pm10_values) / len(pm10_values), 2) if pm10_values else None,
            "pm25_stations": len(pm25_values),
            "pm10_stations": len(pm10_values),
            "total_stations": len(stations)
        }

    except Exception as e:
        print(f"Error fetching all stations current AQI: {e}", flush=True)
        return None


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        city_name = request.json.get("city")
        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return jsonify({"error": "Invalid city"}), 400

        # 🌦 Weather data
        weather_data = fetch_weather_series(lat, lon)
        if not weather_data:
            print(f"[predict] Weather fetch failed for {city_name}, using empty fallback", flush=True)
            weather_data = []  # Continue without weather — model will use defaults

        # ✅ EnvAlert (PRIMARY → city stations)
        envalert_today_data = get_today_data_from_envalert(city_name)
        env_source = "city"

        # 🔁 FALLBACK → nearest city (2 stations)
        if not envalert_today_data:
            envalert_today_data = get_fallback_data_from_nearest_city(city_name)
            env_source = "nearest_city_fallback"

        result = {}
        model_predictions_for_error = {}
        today_pollutants = []

        # ⚡ Fetch pollutant series in parallel
        def fetch_pollutant_data(pollutant):
            return pollutant, fetch_pollutant_series(lat, lon, pollutant)

        with ThreadPoolExecutor(max_workers=6) as executor:
            pollutant_results = dict(executor.map(fetch_pollutant_data, TARGET_POLLUTANTS))

        # 🔮 MODEL predictions (TODAY INCLUDED)
        for pollutant in TARGET_POLLUTANTS:
            pol_data, ts_series = pollutant_results.get(pollutant, ([], []))

            prediction = predict_pollutant(
                pollutant,
                pol_data,
                weather_data,
                ts_series,
                start_day=0
            )

            result[pollutant] = prediction

            # Store TODAY model prediction for error calc
            if prediction:
                model_predictions_for_error[pollutant] = prediction[0]

        # 🧮 Error calculation (EnvAlert vs Model)
        errors = calculate_errors(envalert_today_data, model_predictions_for_error)

        # ➕ Apply error correction (PM2.5 & PM10) → TODAY + FUTURE
        # Today (i=0): capped at 90% of station value so prediction stays below station
        # Future days (i>0): only apply reduced bias (30%) to avoid flat same-value forecast
        BIAS_FACTOR_TODAY  = 0.85
        BIAS_FACTOR_FUTURE = 0.70
        station_pm25 = envalert_today_data.get("pm2_5", {}).get("value") if envalert_today_data else None
        station_pm10_val = envalert_today_data.get("pm10", {}).get("value") if envalert_today_data else None
        station_caps = {"pm2_5": station_pm25, "pm10": station_pm10_val}

        for pollutant in ["pm2_5", "pm10"]:
            error_key = f"{pollutant}_concentration"
            if error_key in errors and pollutant in result:
                for i in range(len(result[pollutant])):
                    bias = BIAS_FACTOR_TODAY if i == 0 else BIAS_FACTOR_FUTURE
                    corrected = result[pollutant][i]["value"] + (errors[error_key] * bias)
                    # Cap only today at 90% of station value
                    if i == 0:
                        cap = station_caps.get(pollutant)
                        if cap and corrected > cap * 0.90:
                            corrected = cap * 0.90
                    result[pollutant][i]["value"] = round(corrected, 2)

                    new_aqi = get_aqi_sub_index(result[pollutant][i]["value"], pollutant)
                    result[pollutant][i]["aqi"] = int(new_aqi) if not pd.isna(new_aqi) else 0

                    category, warning, color = get_category_info(result[pollutant][i]["aqi"])
                    result[pollutant][i]["category"] = category
                    result[pollutant][i]["warning"] = warning
                    result[pollutant][i]["color"] = color

        # ➕ PM10 = PM10 + PM2.5 (MODEL BASED)
        pm10_preds = result.get("pm10", [])
        pm25_preds = result.get("pm2_5", [])

        if pm10_preds and pm25_preds:
            for i in range(min(len(pm10_preds), len(pm25_preds))):
                combined_value = pm10_preds[i]["value"] + pm25_preds[i]["value"]
                # Cap only today's PM10 at 90% of station value
                if i == 0 and station_pm10_val and combined_value > station_pm10_val * 0.90:
                    combined_value = station_pm10_val * 0.90
                pm10_preds[i]["value"] = round(combined_value, 2)

                new_aqi = get_aqi_sub_index(combined_value, "pm10")
                pm10_preds[i]["aqi"] = int(new_aqi) if not pd.isna(new_aqi) else 0

                category, warning, color = get_category_info(pm10_preds[i]["aqi"])
                pm10_preds[i]["category"] = category
                pm10_preds[i]["warning"] = warning
                pm10_preds[i]["color"] = color

        # 📅 Today's pollutants
        for pollutant in TARGET_POLLUTANTS:
            if result.get(pollutant):
                today_data = result[pollutant][0].copy()
                today_data["pollutant"] = pollutant
                today_pollutants.append(today_data)

        # 🌍 Overall AQI (excluding O3)
        overall_daily_aqi = []
        for i in range(7):
            daily_values = []
            for p in TARGET_POLLUTANTS:
                if p != "o3" and len(result.get(p, [])) > i:
                    daily_values.append({
                        "pollutant": p,
                        "aqi": result[p][i]["aqi"],
                        "value": result[p][i]["value"],
                        "category": result[p][i]["category"],
                        "warning": result[p][i]["warning"],
                        "color": result[p][i]["color"]
                    })

            if daily_values:
                highest = max(daily_values, key=lambda x: x["aqi"])
                overall_daily_aqi.append({
                    "day": result[TARGET_POLLUTANTS[0]][i]["day"],
                    "date": result[TARGET_POLLUTANTS[0]][i]["date"],
                    "main_pollutant": highest["pollutant"],
                    "value": highest["value"],
                    "aqi": highest["aqi"],
                    "category": highest["category"],
                    "warning": highest["warning"],
                    "color": highest["color"]
                })

        return jsonify({
            "city": city_name,
            "predictions": result,
            "today_pollutants": today_pollutants,
            "overall_daily_aqi": overall_daily_aqi,
            "errors": errors,
            "env_source": env_source,   # 🔥 city / nearest_city_fallback
            "lat": lat,
            "lon": lon
        })

    except Exception as e:
        print(f"Error in /predict: {e}", flush=True)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/weather', methods=['POST', 'OPTIONS'])
def weather_forecast():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        city_name = request.json.get("city")
        if not city_name:
            return jsonify({"error": "City name required"}), 400

        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return jsonify({"error": "City not found"}), 404

        today = datetime.now(IST).date()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=3)).strftime("%Y-%m-%d")

        # Call 1: daily forecast (no current — avoids Open-Meteo conflict)
        daily_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&timezone=Asia/Kolkata&start_date={start_date}&end_date={end_date}"
        )
        # Call 2: current weather only (no date range)
        current_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature,precipitation,weathercode"
            f"&timezone=Asia/Kolkata"
        )

        daily_data = None
        current_data = None
        for _attempt in range(3):
            try:
                if daily_data is None:
                    r1 = requests.get(daily_url, timeout=15)
                    daily_data = r1.json()
                if current_data is None:
                    r2 = requests.get(current_url, timeout=15)
                    current_data = r2.json()
                break
            except Exception as _e:
                print(f"[weather] attempt {_attempt+1} failed: {_e}", flush=True)
                time.sleep(1)

        if not daily_data:
            return jsonify({"error": "Weather service unavailable"}), 503

        daily = daily_data.get("daily", {})
        current = current_data.get("current", {}) if current_data else {}

        forecast = []
        for i in range(len(daily.get("time", []))):
            date_str = daily["time"][i]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            day = "Today" if i == 0 else "Tomorrow" if i == 1 else date_obj.strftime("%A")
            forecast.append({
                "date": date_str,
                "day": day,
                "max_temp": daily["temperature_2m_max"][i],
                "min_temp": daily["temperature_2m_min"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "max_wind_speed_kmh": daily["windspeed_10m_max"][i]
            })

        return jsonify({
            "city": city_name,
            "forecast": forecast,
            "current": {
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "precipitation": current.get("precipitation"),
                "weathercode": current.get("weathercode"),
            }
        })

    except Exception as e:
        print(f"Error in /weather: {e}", flush=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/station/<int:station_id>', methods=['GET'])
def proxy_station_aqi(station_id):
    try:
        # First try to get from ALL stations cache — faster and avoids individual blocks
        all_stations = fetch_envalert_all_with_cache()
        if all_stations:
            for st in all_stations:
                if str(st.get("station_id")) == str(station_id):
                    return jsonify([st])
        # Fallback to individual fetch
        data = fetch_envalert_station_with_retry(station_id)
        if data is None:
            return jsonify([]), 200
        return jsonify(data if isinstance(data, list) else [data])
    except Exception as e:
        print(f"Error proxying station {station_id}: {e}", flush=True)
        return jsonify([]), 200

@app.route('/api/get_average', methods=['GET'])
def get_average():
    data = getAvgOfAllStationsValues()

    if data is None:
        return jsonify({"error": "Failed to fetch average AQI data"}), 500

    return jsonify(data)



@app.route('/predict_grid', methods=['POST', 'OPTIONS'])
def predict_grid():
    """
    Accepts a city name + grid_size, generates a grid of lat/lon points
    around the city, runs CNN models for each point, returns predicted
    AQI per point. Used by frontend to build a model-driven IDW heatmap.
    """
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        city_name = request.json.get("city")
        grid_size = int(request.json.get("grid_size", 3))   # 3x3 = 9 points (fast)
        radius_deg = float(request.json.get("radius_deg", 0.3))  # ~33km radius

        if not city_name:
            return jsonify({"error": "city is required"}), 400

        # Get city center coordinates
        center_lat, center_lon = get_city_coordinates(city_name)
        if not center_lat or not center_lon:
            return jsonify({"error": f"Could not find coordinates for {city_name}"}), 400

        print(f"🗺️ predict_grid: {city_name} ({center_lat},{center_lon}) grid={grid_size}x{grid_size}", flush=True)

        # Generate evenly spaced grid of points around city center
        points = []
        step = (radius_deg * 2) / (grid_size - 1) if grid_size > 1 else 0
        for i in range(grid_size):
            for j in range(grid_size):
                pt_lat = round(center_lat - radius_deg + i * step, 4)
                pt_lon = round(center_lon - radius_deg + j * step, 4)
                points.append({"lat": pt_lat, "lon": pt_lon})

        print(f"📍 Generated {len(points)} grid points", flush=True)

        def predict_aqi_for_point(point):
            pt_lat = point["lat"]
            pt_lon = point["lon"]
            try:
                # Fetch weather data for this point
                weather_data = fetch_weather_series(pt_lat, pt_lon)
                if not weather_data:
                    return {"lat": pt_lat, "lon": pt_lon, "aqi": None, "error": "no_weather"}

                # Fetch all 6 pollutant series in parallel
                def fetch_pol(pollutant):
                    return pollutant, fetch_pollutant_series(pt_lat, pt_lon, pollutant)

                with ThreadPoolExecutor(max_workers=6) as ex:
                    pol_results = dict(ex.map(fetch_pol, TARGET_POLLUTANTS))

                # Run CNN model for each pollutant, collect today's AQI
                daily_aqis = []
                pollutant_details = {}

                for pollutant in TARGET_POLLUTANTS:
                    if pollutant == "o3":
                        continue  # exclude o3 from overall AQI (matches /predict logic)

                    pol_data, ts_series = pol_results.get(pollutant, ([], []))
                    prediction = predict_pollutant(
                        pollutant, pol_data, weather_data, ts_series, start_day=0
                    )

                    if prediction and len(prediction) > 0:
                        today_aqi = prediction[0]["aqi"]
                        daily_aqis.append(today_aqi)
                        pollutant_details[pollutant] = {
                            "aqi": today_aqi,
                            "value": prediction[0]["value"],
                            "category": prediction[0]["category"]
                        }

                if not daily_aqis:
                    return {"lat": pt_lat, "lon": pt_lon, "aqi": None, "error": "no_predictions"}

                # Overall AQI = max across pollutants (same as /predict endpoint)
                overall_aqi = max(daily_aqis)
                print(f"  ✅ ({pt_lat},{pt_lon}) → AQI={overall_aqi}", flush=True)

                return {
                    "lat": pt_lat,
                    "lon": pt_lon,
                    "aqi": overall_aqi,
                    "pollutants": pollutant_details
                }

            except Exception as e:
                print(f"  ❌ Error for ({pt_lat},{pt_lon}): {e}", flush=True)
                return {"lat": pt_lat, "lon": pt_lon, "aqi": None, "error": str(e)}

        # Process all grid points in parallel (max 5 at once to avoid overload)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(predict_aqi_for_point, points))

        valid_results = [r for r in results if r.get("aqi") is not None]
        failed_count = len(results) - len(valid_results)

        print(f"✅ predict_grid done: {len(valid_results)} valid, {failed_count} failed", flush=True)

        return jsonify({
            "city": city_name,
            "center": {"lat": center_lat, "lon": center_lon},
            "grid_size": grid_size,
            "total_points": len(points),
            "valid_points": len(valid_results),
            "grid": valid_results
        })

    except Exception as e:
        print(f"Error in /predict_grid: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/debug_stations', methods=['GET'])
def debug_stations():
    """Debug endpoint — shows raw EnvAlert API response for first 3 stations"""
    try:
        data = fetch_envalert_all_with_cache()
        if isinstance(data, list) and len(data) > 0:
            return jsonify({
                "total": len(data),
                "sample_keys": list(data[0].keys()),
                "first_3": data[:3]
            })
        return jsonify({"raw": str(data)[:500]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/all_stations_aqi", methods=["GET", "OPTIONS"])
def all_stations_aqi():
    """Returns individual AQI for every station: {station_id: aqi}"""
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200
    try:
        stations = fetch_envalert_all_with_cache()
        if not stations or not isinstance(stations, list):
            return jsonify({"error": "EnvAlert unavailable"}), 503
        result = {}
        for station in stations:
            sid = station.get("station_id")
            aqi_raw = station.get("aqi") or station.get("AQI")
            if sid is None or aqi_raw in (None, "", "null", "NULL", "ID"):
                continue
            try:
                aqi_val = float(str(aqi_raw).strip())
                if 0 < aqi_val <= 500:
                    result[int(sid)] = round(aqi_val)
            except (ValueError, TypeError):
                continue
        return jsonify(result)
    except Exception as e:
        print(f"[all_stations_aqi] Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route('/mp_ranking', methods=['POST', 'OPTIONS'])
def mp_ranking():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    try:
        city_name = (request.json or {}).get("city", "").strip()

        # ── 1. Fetch ALL station data ──────────────────────────────
        stations = fetch_envalert_all_with_cache()
        if not stations or not isinstance(stations, list) or len(stations) == 0:
            return jsonify({"error": "No station data available"}), 503

        print(f"[mp_ranking] Got {len(stations)} stations. Sample keys: {list(stations[0].keys())}", flush=True)

        # ── 2. Reverse map: station_id (int) → city name ───────────
        station_to_city = {}
        for cname, ids in CITY_STATIONS.items():
            for sid in ids:
                station_to_city[int(sid)] = cname

        # ── 3. Parse each station → get AQI ────────────────────────
        city_aqi_map = {}  # city → list of aqi values

        for station in stations:
            # Get station_id — could be int or string
            sid_raw = station.get("station_id")
            try:
                sid = int(str(sid_raw).strip())
            except (ValueError, TypeError):
                continue

            # Get AQI — try multiple field names, API returns string "177"
            aqi_raw = None
            for field in ["aqi", "AQI", "overall_aqi", "pm25_subindex", "pm10_subindex"]:
                val = station.get(field)
                if val not in (None, "", "null", "NULL", "ID", "N/A"):
                    aqi_raw = val
                    break

            # If no aqi field, calculate from pm25 using our breakpoints
            if aqi_raw is None:
                pm25_raw = station.get("pm25") or station.get("PM25") or station.get("pm2_5")
                if pm25_raw not in (None, "", "null", "ID"):
                    try:
                        pm25_val = float(str(pm25_raw).strip())
                        aqi_raw = get_aqi_sub_index(pm25_val, "pm2_5")
                    except:
                        pass

            if aqi_raw is None:
                continue

            try:
                aqi_val = float(str(aqi_raw).strip())
                if aqi_val <= 0 or aqi_val > 500:
                    continue
            except (ValueError, TypeError):
                continue

            # Match station to city
            city = station_to_city.get(sid)

            # Fallback: match by station_name string
            if not city:
                sname = str(station.get("station_name", "") or station.get("name", "")).lower()
                for cname in CITY_STATIONS:
                    if cname.lower() in sname:
                        city = cname
                        break

            if city:
                city_aqi_map.setdefault(city, []).append(aqi_val)
                print(f"[mp_ranking] Station {sid} → {city}: AQI={aqi_val}", flush=True)

        print(f"[mp_ranking] Cities mapped: {list(city_aqi_map.keys())}", flush=True)

        if not city_aqi_map:
            return jsonify({"error": "Could not map any stations to cities. Check station IDs."}), 500

        # ── 4. Average AQI per city, sort highest → lowest ─────────
        city_rankings = []
        for cname, aqis in city_aqi_map.items():
            avg_aqi = round(sum(aqis) / len(aqis))
            category, _, color = get_category_info(avg_aqi)
            city_rankings.append({
                "city": cname,
                "aqi": avg_aqi,
                "category": category,
                "color": color,
                "station_count": len(aqis),
            })

        city_rankings.sort(key=lambda x: x["aqi"], reverse=False)
        for i, entry in enumerate(city_rankings):
            entry["rank"] = i + 1

        # ── 5. Find rank of requested city ─────────────────────────
        target_entry = next(
            (e for e in city_rankings if e["city"].lower() == city_name.lower()),
            None
        )

        return jsonify({
            "city": city_name,
            "rank": target_entry["rank"] if target_entry else None,
            "total_cities": len(city_rankings),
            "aqi": target_entry["aqi"] if target_entry else None,
            "category": target_entry["category"] if target_entry else None,
            "color": target_entry["color"] if target_entry else None,
            "all_rankings": city_rankings,
        })

    except Exception as e:
        print(f"[mp_ranking] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/monthly_average', methods=['POST', 'OPTIONS'])
def monthly_average():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200
    try:
        data = request.get_json()
        city_name = data.get("city", "").strip()

        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return jsonify({"error": f"City '{city_name}' not found"}), 404

        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=29)

        results = {}
        for pollutant, api_field in POLLUTANT_API_MAP.items():
            url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly={api_field}&timezone=Asia%2FKolkata"
            )
            try:
                resp = None
                for _ma_attempt in range(3):
                    try:
                        resp = requests.get(url, timeout=20)
                        break
                    except Exception as _mae:
                        print(f"[monthly_average] attempt {_ma_attempt+1} failed: {_mae}", flush=True)
                        time.sleep(1)
                if resp is None:
                    continue
                d = resp.json()
                hourly_values = d["hourly"].get(api_field, [])
                hourly_times  = d["hourly"].get("time", [])

                # Group by date and compute daily average
                daily = {}
                for ts, val in zip(hourly_times, hourly_values):
                    if val is None:
                        continue
                    date_str = ts[:10]
                    daily.setdefault(date_str, []).append(val)

                results[pollutant] = [
                    {"date": date, "avg": round(sum(vals) / len(vals), 2)}
                    for date, vals in sorted(daily.items())
                ]
            except Exception as e:
                print(f"[monthly_average] {pollutant} error: {e}", flush=True)
                results[pollutant] = []

        # Compute daily overall AQI from pm2_5 daily averages
        import math
        aqi_series_raw = []
        for entry in results.get("pm2_5", []):
            try:
                avg_val = entry.get("avg")
                if avg_val is None:
                    aqi_series_raw.append({"date": entry["date"], "avg": None})
                else:
                    aqi_val = get_aqi_sub_index(float(avg_val), "pm2_5")
                    safe = round(aqi_val) if (aqi_val and not math.isnan(float(aqi_val))) else None
                    aqi_series_raw.append({"date": entry["date"], "avg": safe})
            except Exception as ex:
                print(f"[monthly_average] AQI calc error: {ex}", flush=True)
                aqi_series_raw.append({"date": entry["date"], "avg": None})

        # Apply correction factor: align Open-Meteo AQI to EnvAlert real sensor AQI
        correction = 0
        try:
            envalert_today = get_today_data_from_envalert(city_name)
            if envalert_today and "pm2_5" in envalert_today:
                envalert_aqi_today = max([envalert_today[p]["aqi"] for p in TARGET_POLLUTANTS if p != "o3" and p in envalert_today])
                # Find today's Open-Meteo AQI
                today_str = datetime.now(IST).date().strftime("%Y-%m-%d")
                openmeteo_today = next((e["avg"] for e in aqi_series_raw if e["date"] == today_str and e["avg"] is not None), None)
                if openmeteo_today:
                    correction = round(envalert_aqi_today - openmeteo_today)
                    print(f"[monthly_average] AQI correction for {city_name}: EnvAlert={envalert_aqi_today}, OpenMeteo={openmeteo_today}, offset={correction}", flush=True)
        except Exception as ce:
            print(f"[monthly_average] correction calc error: {ce}", flush=True)

        # Apply correction clamped to ±80 to avoid wild shifts
        correction = max(-80, min(80, correction))
        aqi_series = []
        for entry in aqi_series_raw:
            if entry["avg"] is not None:
                corrected = max(0, min(500, entry["avg"] + correction))
                aqi_series.append({"date": entry["date"], "avg": corrected})
            else:
                aqi_series.append(entry)

        # Apply per-pollutant correction to align Open-Meteo values with EnvAlert sensors
        try:
            if envalert_today:
                today_str = datetime.now(IST).date().strftime("%Y-%m-%d")
                for pollutant in TARGET_POLLUTANTS:
                    if pollutant not in envalert_today or pollutant not in results:
                        continue
                    live_val = envalert_today[pollutant]["value"]
                    # Find today's Open-Meteo value for this pollutant
                    today_entry = next((e for e in results[pollutant] if e["date"] == today_str), None)
                    if not today_entry or today_entry["avg"] is None:
                        continue
                    pol_correction = round(live_val - today_entry["avg"], 2)
                    # Clamp correction to ±100
                    pol_correction = max(-100, min(100, pol_correction))
                    print(f"[monthly_average] {pollutant} correction: live={live_val}, openmeteo={today_entry['avg']}, offset={pol_correction}", flush=True)
                    # Apply to all days
                    results[pollutant] = [
                        {"date": e["date"], "avg": round(max(0, e["avg"] + pol_correction), 2)}
                        if e["avg"] is not None else e
                        for e in results[pollutant]
                    ]
        except Exception as pe:
            print(f"[monthly_average] pollutant correction error: {pe}", flush=True)

        return jsonify({
            "city": city_name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "aqi": aqi_series,
            "pollutants": results
        })

    except Exception as e:
        print(f"[monthly_average] Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


# ── LLM Chat Route (Gemini) ───────────────────────────────────────────────
# Requires: pip install google-generativeai python-dotenv

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_message = data.get("message", "").strip()
        current_city = data.get("city", "")

        if not user_message:
            return jsonify({"error": "message is required"}), 400

        # ── Detect if user is asking about a specific MP city ──
        asked_city = None
        for city_key in CITY_STATIONS.keys():
            if city_key.lower() in user_message.lower():
                asked_city = city_key
                break

        # Use asked city if found, otherwise fall back to selected city
        target_city = asked_city or current_city

        # Fetch live EnvAlert data for that city
        live_aqi = None
        if target_city:
            live_aqi = get_today_data_from_envalert(target_city)

        # Build context string
        context = ""
        if target_city and live_aqi:
            context = f"City: {target_city}\nLive AQI data: {live_aqi}\n\n"
        elif target_city:
            context = f"City: {target_city}\n(No live data available, use general knowledge)\n\n"

        # 🔥 System Prompt
        system_prompt = (
            "You are AeroBot, an air quality assistant for Madhya Pradesh, India. "
            "You have knowledge about air quality, AQI levels, pollutants, and health impacts "
            "for all major cities in Madhya Pradesh including Indore, Bhopal, Jabalpur, Gwalior, "
            "Ujjain, Sagar, Dewas, Satna, Ratlam, Rewa, Katni, Singrauli, Khandwa, Khargone, "
            "Pithampur, Mandideep, Narsinghpur, Neemuch, Maihar, Betul, Anuppur, and others. "
            "When a user asks about any MP city, answer based on the live data provided or general knowledge. "
            "If the user asks about a city not in Madhya Pradesh, politely tell them this assistant "
            "only covers Madhya Pradesh cities. "
            "IMPORTANT: Always respond in plain simple paragraphs only. "
            "Do NOT use bullet points, asterisks (*), bold (**), headers (#), "
            "or any markdown formatting whatsoever. "
            "Write everything as natural flowing sentences in 2-3 short paragraphs. "
            "Be concise, friendly, and use simple language. "
            "Respond in the same language the user writes in (Hindi or English)."
        )

        # 🔥 Combine system + context + user prompt
        full_prompt = (
            f"{system_prompt}\n\n"
            f"{context}"
            f"User question: {user_message}"
        )

        # Call Gemini
        # Call Gemini — try multiple models if quota exceeded
        GEMINI_MODELS = [
             "gemini-2.5-flash",
             "gemini-2.5-flash-lite",
             "gemini-2.0-flash-001",
             "gemini-2.0-flash-lite",
        ]

        reply = None
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(full_prompt)
                reply = response.text
                print(f"[/api/chat] Used model: {model_name}", flush=True)
                break
            except Exception as model_err:
                print(f"[/api/chat] Model {model_name} failed: {model_err}", flush=True)
                continue

        if not reply:
                return jsonify({"error": "All Gemini models quota exceeded. Try again tomorrow."}), 429

        # Clean any remaining markdown just in case
        import re
        reply = re.sub(r'\*\*(.*?)\*\*', r'\1', reply)
        reply = re.sub(r'\*(.*?)\*', r'\1', reply)
        reply = re.sub(r'^\*\s+', '', reply, flags=re.MULTILINE)
        reply = re.sub(r'^\-\s+', '', reply, flags=re.MULTILINE)
        reply = re.sub(r'#{1,6}\s', '', reply)

        print(f"[/api/chat] asked_city={asked_city}, target={target_city} → reply length={len(reply)}", flush=True)
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[/api/chat] Error: {e}", flush=True)
        return jsonify({"error": "LLM service unavailable"}), 500
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/debug_models', methods=['GET'])
def debug_models():
    try:
        available = [m.name for m in genai.list_models() 
                     if 'generateContent' in m.supported_generation_methods]
        return jsonify({"models": available})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    print("🚀 Flask server is starting...", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
