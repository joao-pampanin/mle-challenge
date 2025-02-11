import requests

url = 'http://0.0.0.0:8000/predict'
api_key = 'ABCD-1234-EFGH-5678'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "type": "casa",
    "sector": "vitacura",
    "net_usable_area": 152.0,
    "net_area": 257.0,
    "n_rooms": 3.0,
    "n_bathroom": 3.0,
    "latitude": -33.3794,
    "longitude": -70.5447
}

response = requests.post(url, params={'api_key': api_key}, headers=headers, json=data)

print(response.status_code)
print(response.json())