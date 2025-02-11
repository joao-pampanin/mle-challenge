#!/bin/bash

curl -X 'POST' \
  'http://localhost:8000/predict?api_key=ABCD-1234-EFGH-5678' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "type": "casa",
  "sector": "vitacura",
  "net_usable_area": 152.0,
  "net_area": 257.0,
  "n_rooms": 3.0,
  "n_bathroom": 3.0,
  "latitude": -33.3794,
  "longitude": -70.5447
}'