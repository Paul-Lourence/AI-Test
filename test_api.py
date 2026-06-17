import requests

data = {
    "ph": 7.2,
    "temperature": 26.5,
    "turbidity": 2.1,
    "tds": 180
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)

print(response.json())