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
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

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

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200
    
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        city_name = request.json.get("city")
        lat, lon = get_city_coordinates(city_name)
        if not lat or not lon:
            return jsonify({"error": "Invalid city"}), 400

        weather_data = fetch_weather_series(lat, lon)
        if not weather_data:
            return jsonify({"error": "Weather fetch failed"}), 400

        # Try to get today's data from EnvAlert API
        envalert_today_data = get_today_data_from_envalert(city_name)
        
        result = {}
        model_predictions_for_error = {}  # Store model predictions for error calculation
        today_pollutants = []
        # Only use API data for PM2.5 and PM10
        api_pollutants = ["pm2_5", "pm10"]
        use_api_data = envalert_today_data is not None

        print(f"\n{'='*60}")
        print(f"Using EnvAlert API for PM2.5 and PM10 (today)")
        print(f"Using Model Predictions for CO, NO2, O3, SO2 (all days)")
        print(f"{'='*60}\n")

        # Fetch pollutant data for all pollutants in parallel
        def fetch_pollutant_data(pollutant):
            return pollutant, fetch_pollutant_series(lat, lon, pollutant)
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            pollutant_results = dict(executor.map(lambda p: fetch_pollutant_data(p), TARGET_POLLUTANTS))

        for pollutant in TARGET_POLLUTANTS:
            pol_data, ts_series = pollutant_results.get(pollutant, ([], []))
            
            # Only use API data for PM2.5 and PM10 for today
            if use_api_data and pollutant in api_pollutants and pollutant in envalert_today_data:
                # Get model prediction for today for error calculation
                model_pred_today = predict_pollutant(pollutant, pol_data, weather_data, ts_series, start_day=0)
                if model_pred_today:
                    model_predictions_for_error[pollutant] = model_pred_today[0]
                
                # Use EnvAlert API data for today (PM2.5 and PM10 only)
                api_data = envalert_today_data[pollutant]
                api_value = api_data['value']
                api_aqi = api_data['aqi']
                
                category, warning, color = get_category_info(api_aqi)
                
                today_data = {
                    "day": "Today",
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "value": round(api_value, 2),
                    "aqi": int(api_aqi),
                    "category": category,
                    "warning": warning,
                    "color": color
                }
                
                # Get predictions for tomorrow onwards (start_day=1)
                future_predictions = predict_pollutant(pollutant, pol_data, weather_data, ts_series, start_day=1)
                result[pollutant] = [today_data] + future_predictions
            else:
                # Use model predictions for all days including today (CO, NO2, O3, SO2)
                prediction = predict_pollutant(pollutant, pol_data, weather_data, ts_series, start_day=0)
                result[pollutant] = prediction

        # Calculate errors (avg of all stations - predicted by model)
        errors = calculate_errors(envalert_today_data, model_predictions_for_error)

        # Adjust pm10 values by adding pm2_5 values (only for model predictions, not API data)
        pm10_preds = result.get("pm10", [])
        pm25_preds = result.get("pm2_5", [])
        if pm10_preds and pm25_preds:
            for i in range(min(len(pm10_preds), len(pm25_preds))):
                # For EnvAlert API data (day 0 only), PM10 already includes PM2.5
                # Only add PM2.5 to PM10 if using model predictions
                if not (use_api_data and i == 0 and "pm10" in envalert_today_data and "pm2_5" in envalert_today_data):
                    combined_value = pm10_preds[i]["value"] + pm25_preds[i]["value"]
                    pm10_preds[i]["value"] = round(combined_value, 2)
                    new_aqi = get_aqi_sub_index(combined_value, "pm10")
                    pm10_preds[i]["aqi"] = int(new_aqi) if not pd.isna(new_aqi) else 0
                    category, warning, color = get_category_info(pm10_preds[i]["aqi"])
                    pm10_preds[i]["category"] = category
                    pm10_preds[i]["warning"] = warning
                    pm10_preds[i]["color"] = color

        # Add errors to future predictions (days 1-6) for PM2.5 and PM10
        if errors:
            # Add PM2.5 concentration error to future predictions
            if "pm2_5_concentration" in errors and "pm2_5" in result:
                for i in range(1, len(result["pm2_5"])):
                    result["pm2_5"][i]["value"] = round(result["pm2_5"][i]["value"] + errors["pm2_5_concentration"], 2)
                    # Recalculate AQI based on adjusted concentration
                    new_aqi = get_aqi_sub_index(result["pm2_5"][i]["value"], "pm2_5")
                    result["pm2_5"][i]["aqi"] = int(new_aqi) if not pd.isna(new_aqi) else 0
                    category, warning, color = get_category_info(result["pm2_5"][i]["aqi"])
                    result["pm2_5"][i]["category"] = category
                    result["pm2_5"][i]["warning"] = warning
                    result["pm2_5"][i]["color"] = color
            
            # Add PM10 concentration error to future predictions
            if "pm10_concentration" in errors and "pm10" in result:
                for i in range(1, len(result["pm10"])):
                    result["pm10"][i]["value"] = round(result["pm10"][i]["value"] + errors["pm10_concentration"], 2)
                    # Recalculate AQI based on adjusted concentration
                    new_aqi = get_aqi_sub_index(result["pm10"][i]["value"], "pm10")
                    result["pm10"][i]["aqi"] = int(new_aqi) if not pd.isna(new_aqi) else 0
                    category, warning, color = get_category_info(result["pm10"][i]["aqi"])
                    result["pm10"][i]["category"] = category
                    result["pm10"][i]["warning"] = warning
                    result["pm10"][i]["color"] = color


        # Prepare today's pollutants list
        for pollutant in TARGET_POLLUTANTS:
            prediction = result.get(pollutant, [])
            if prediction:
                today_data = prediction[0].copy()
                today_data["pollutant"] = pollutant
                today_pollutants.append(today_data)

        # Calculate overall_daily_aqi ignoring ozone ('o3')
        overall_daily_aqi = []
        for i in range(7):
            daily_values = []
            for p in TARGET_POLLUTANTS:
                pollutant_data = result.get(p, [])
                if len(pollutant_data) > i:
                    daily_values.append({
                        "pollutant": p,
                        "aqi": pollutant_data[i]["aqi"],
                        "value": pollutant_data[i]["value"],
                        "category": pollutant_data[i]["category"],
                        "warning": pollutant_data[i]["warning"],
                        "color": pollutant_data[i]["color"]
                    })
            if daily_values:
                daily_values_sorted = sorted(daily_values, key=lambda x: x["aqi"], reverse=True)
                if daily_values_sorted[0]["pollutant"] == "o3" and len(daily_values_sorted) > 1:
                    highest = daily_values_sorted[1]
                else:
                    highest = daily_values_sorted[0]
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

        response_data = {
            "city": city_name,
            "predictions": result,
            "today_pollutants": today_pollutants,
            "overall_daily_aqi": overall_daily_aqi,
            "errors": errors,
            "lat": lat,
            "lon": lon,
            "data_source": {
                "pm2_5": "EnvAlert API (today)" if (use_api_data and "pm2_5" in envalert_today_data) else "Model Predictions",
                "pm10": "EnvAlert API (today)" if (use_api_data and "pm10" in envalert_today_data) else "Model Predictions",
                "co": "Model Predictions",
                "no2": "Model Predictions",
                "o3": "Model Predictions",
                "so2": "Model Predictions"
            }
        }
        
        return jsonify(response_data)

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
        resp = requests.post(url, timeout=10)
        data = resp.json()
        return jsonify(data)
    except Exception as e:
        print(f"Error proxying station {station_id}: {e}", flush=True)
        return jsonify({"error": "Failed to fetch station data"}), 500

if __name__ == "__main__":
    print("🚀 Flask server is starting...", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)