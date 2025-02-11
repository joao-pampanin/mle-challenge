import logging
import os
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.config import Config

app = FastAPI()

log_dir = "data/00_logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"{log_dir}/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

model = joblib.load(Config.PATHS.MODEL_PATH)


class PropertyData(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: int
    n_bathroom: int
    latitude: float
    longitude: float


def verify_api_key(api_key: str = Query(...)) -> None:
    if api_key != Config.API.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.get("/", response_model=Dict[str, str])
def read_root() -> Dict[str, str]:
    """
    Root endpoint that returns a welcome message.
    """
    return {
        "message": "Welcome to the API. Visit /docs for the interactive documentation."
    }


@app.post(
    "/predict", dependencies=[Depends(verify_api_key)], response_model=Dict[str, Any]
)
def predict(data: PropertyData) -> Dict[str, Any]:
    """
    Endpoint to make predictions based on the input data.

    Parameters:
    data (PropertyData): The input data for the prediction.

    Returns:
    Dict[str, Any]: The prediction result.
    """
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    logger.info(f"Prediction made for input: {data.dict()} - Result: {prediction[0]}")
    return {"prediction": prediction[0]}
