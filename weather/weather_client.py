import requests
from config import WEATHER_API_KEY, WEATHER_CITY

WEATHER_SEVERITY = {
    "clear sky":          0.0,
    "few clouds":         0.1,
    "scattered clouds":   0.2,
    "broken clouds":      0.3,
    "overcast clouds":    0.4,
    "light rain":         0.6,
    "moderate rain":      0.7,
    "heavy intensity rain": 0.9,
    "thunderstorm":       1.0,
    "fog":                0.8,
    "mist":               0.5,
    "haze":               0.4,
}

_cached_weather = {"condition": "clear sky", "score": 0.0}

def get_weather() -> dict:
    """
    Fetch current weather from OpenWeatherMap.
    Returns condition string and severity score 0-1.
    Cached — call every 10 minutes, not every frame.
    """
    global _cached_weather
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?q={WEATHER_CITY}&appid={WEATHER_API_KEY}")
        resp = requests.get(url, timeout=5)
        data = resp.json()

        condition = data["weather"][0]["description"].lower()
        score     = WEATHER_SEVERITY.get(condition, 0.3)

        _cached_weather = {"condition": condition, "score": score}
        print(f"[Weather] {condition} → severity {score:.1f}")

    except Exception as e:
        print(f"[Weather] API failed ({e}), using cached: {_cached_weather}")

    return _cached_weather