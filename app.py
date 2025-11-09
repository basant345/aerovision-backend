from flask import Flask, request, jsonify
import os
import numpy as np
import requests
from datetime import datetime, timedelta
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Disable GPU for CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": [
            "http://localhost:8080",
            "https://airqualitycities.iiti.ac.in"
        ]
    }
})

api_key = "701cf10ad3df9b6f5f58f40bfba7e837"

TARGET_POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

POLLUTANT_API_MAP = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "o3": "ozone",
    "co": "carbon_monoxide"
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

models = {}
for pollutant in TARGET_POLLUTANTS:
    try:
        path = os.path.join(os.path.dirname(__file__), f"best_cnn_{pollutant}.keras")
        models[pollutant] = load_model(path)
    except Exception as e:
        print(f"Model load error for {pollutant}: {e}", flush=True)
        models[pollutant] = None

def get_city_coordinates(city_name):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        res = requests.get(url)
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

        print(f"\n🌍 Fetching pollutant [{pollutant}] from URL:")
        print(url)

        response = requests.get(url)
        data = response.json()
        values = data["hourly"].get(api_field, [])
        series = values[-72:]

        print(f"✅ [{pollutant}] Values received: {len(series)} hours")
        return series
    except Exception as e:
        print(f"[{pollutant.upper()}] Pollutant fetch error:", e, flush=True)
        return []

def fetch_weather_series(lat, lon):
    try:
        end_date = datetime.utcnow().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=4)
        weather_params = ",".join(WEATHER_COLS)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={weather_params}"
        response = requests.get(url)
        data = response.json()
        hourly = data["hourly"]
        return [[hourly[col][i] for col in WEATHER_COLS] for i in range(len(hourly['time']))]
    except:
        return []

def predict_pollutant(pollutant, data, weather_data):
    try:
        print(f"\n----------------------------")
        print(f"🔮 Starting Prediction for: {pollutant.upper()}")
        print(f"Data Length Received: {len(data)}")

        model = models.get(pollutant)
        if not model or len(data) < 72:
            print("❗ Model missing or insufficient data.")
            return []

        while len(data) < 72:
            data.insert(0, data[0])

        weather_features = weather_data[-1][:9] if weather_data else [0] * 9
        seq = [0.0] + data[-72:] + weather_features
        sequence = np.array(seq).reshape((1, 82, 1))

        print(f"📏 Model Sequence Shape: {sequence.shape}")

        results = []

        for i in range(7):
            if i == 0:
                # ✅ Use actual pollutant value for Day 0
                pred_val = float(data[-1]) if data else 0.0
                print(f"   Day 0 (Actual) Value → {pred_val}")
            else:
                # 🔮 Model prediction for Day 1–6
                pred_val = float(abs(model.predict(sequence, verbose=0)[0, 0]))
                print(f"   Day {i} (Predicted) Value → {pred_val}")
                sequence = np.roll(sequence, -1, axis=1)
                sequence[0, -1, 0] = pred_val

            aqi = get_aqi_sub_index(pred_val, pollutant)
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

        print(f"✅ Completed Prediction for {pollutant.upper()}")
        print("----------------------------\n")
        return results

    except Exception as e:
        print(f"Prediction error for {pollutant}: {e}", flush=True)
        return []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city_name = request.json.get("city")
        lat, lon = get_city_coordinates(city_name)
        if not lat or not lon:
            return jsonify({"error": "Invalid city"}), 400

        weather_data = fetch_weather_series(lat, lon)
        if not weather_data:
            return jsonify({"error": "Weather fetch failed"}), 400

        result = {}
        today_pollutants = []

        for pollutant in TARGET_POLLUTANTS:
            pol_data = fetch_pollutant_series(lat, lon, pollutant)
            prediction = predict_pollutant(pollutant, pol_data, weather_data)
            result[pollutant] = prediction

        # Adjust pm10 values
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

        for pollutant in TARGET_POLLUTANTS:
            prediction = result.get(pollutant, [])
            if prediction:
                today_data = prediction[0]
                today_data["pollutant"] = pollutant
                today_pollutants.append(today_data)

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

        return jsonify({
            "city": city_name,
            "predictions": result,
            "today_pollutants": today_pollutants,
            "overall_daily_aqi": overall_daily_aqi,
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

        response = requests.get(url)
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

if __name__ == "__main__":
    print("🚀 Flask server is starting...", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


