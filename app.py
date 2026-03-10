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

# Disable GPU for CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
# Allow all origins for React Native app compatibility
# React Native doesn't send traditional browser origins
CORS(app, 
     resources={r"/*": {
         "origins": "*",
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

# Cache for geocoding results (city -> coordinates)
@lru_cache(maxsize=100)
def get_city_coordinates(city_name):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        res = requests.get(url, timeout=5)
        data = res.json()
        if data and isinstance(data, list):
            item = data[0]
            lat = item.get('lat')
            lon = item.get('lon')
            if lat is not None and lon is not None:
                return lat, lon
    except Exception as e:
        print("Error in get_city_coordinates:", e, flush=True)
    return None, None

def fetch_envalert_current_aqi(station_id):
    """Fetch current AQI data for a specific station"""
    try:
        url = f"https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id={station_id}"
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # API returns a list with one station object
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return data
        else:
            print(f"EnvAlert AQI API failed for station {station_id} with status {response.status_code}", flush=True)
            return None
    except Exception as e:
        print(f"Error fetching EnvAlert AQI for station {station_id}: {e}", flush=True)
        return None

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
        
        # Use ThreadPoolExecutor to fetch station data concurrently
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
        response = requests.get(url, timeout=10)
        data = response.json()
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
    try:
        end_date = datetime.utcnow().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=4)
        weather_params = ",".join(WEATHER_COLS)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={weather_params}"
        response = requests.get(url, timeout=10)
        data = response.json()
        hourly = data["hourly"]
        return [[hourly[col][i] for col in WEATHER_COLS] for i in range(len(hourly['time']))]
    except:
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
    url = "https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id=ALL"
    try:
        resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        stations = resp.json()

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
        if not lat or not lon:
            return jsonify({"error": "Invalid city"}), 400

        # 🌦 Weather data
        weather_data = fetch_weather_series(lat, lon)
        if not weather_data:
            return jsonify({"error": "Weather fetch failed"}), 400

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
        for pollutant in ["pm2_5", "pm10"]:
            error_key = f"{pollutant}_concentration"
            if error_key in errors and pollutant in result:
                for i in range(len(result[pollutant])):
                    result[pollutant][i]["value"] = round(
                        result[pollutant][i]["value"] + errors[error_key], 2
                    )

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
        if not lat or not lon:
            return jsonify({"error": "City not found"}), 404

        today = datetime.utcnow().date()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=3)).strftime("%Y-%m-%d")

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&timezone=auto&start_date={start_date}&end_date={end_date}"
        )

        response = requests.get(url, timeout=10)
        data = response.json()
        daily = data.get("daily", {})

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
            "forecast": forecast
        })

    except Exception as e:
        print(f"Error in /weather: {e}", flush=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/station/<int:station_id>', methods=['GET'])
def proxy_station_aqi(station_id):
    try:
        url = f"https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id={station_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://erc.mp.gov.in/",
            "Accept": "application/json"
        }
        resp = requests.post(url, headers=headers, timeout=15)
        data = resp.json()
        return jsonify(data)
    except Exception as e:
        print(f"Error proxying station {station_id}: {e}", flush=True)
        return jsonify({"error": "Failed to fetch station data"}), 500

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
        url = "https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id=ALL"
        resp = requests.post(url, timeout=30)
        data = resp.json()
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
        url = "https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id=ALL"
        resp = requests.post(url, timeout=30)
        resp.raise_for_status()
        stations = resp.json()
        if not isinstance(stations, list):
            return jsonify({"error": "Unexpected response"}), 500
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
        url = "https://erc.mp.gov.in/EnvAlert/Wa-CityAQI?id=ALL"
        resp = requests.post(url, timeout=30)
        resp.raise_for_status()
        stations = resp.json()

        if not isinstance(stations, list) or len(stations) == 0:
            return jsonify({"error": "No station data available"}), 500

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

        city_rankings.sort(key=lambda x: x["aqi"], reverse=True)
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


if __name__ == "__main__":
    print("🚀 Flask server is starting...", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)