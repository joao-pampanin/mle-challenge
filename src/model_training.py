from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from config import Config


def train_model(
    train: pd.DataFrame,
    train_cols: List[str],
    target: str,
    preprocessor_path: str,
    model_path: str,
) -> Pipeline:
    """
    Trains the model with the GradientBoostingRegressor regression estimator.

    Parameters:
    train (pd.DataFrame): The training data.
    train_cols (List[str]): The list of training columns.
    target (str): The target column name.
    preprocessor_path (str): The file path to the saved preprocessor.
    model_path (str): The file path to save the trained model.

    Returns:
    Pipeline: The trained model pipeline.
    """
    preprocessor = joblib.load(preprocessor_path)

    steps = [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                learning_rate=Config.MODEL_PARAMS.LEARNING_RATE,
                n_estimators=Config.MODEL_PARAMS.N_ESTIMATORS,
                max_depth=Config.MODEL_PARAMS.MAX_DEPTH,
                loss=Config.MODEL_PARAMS.LOSS,
            ),
        ),
    ]

    pipeline = Pipeline(steps)
    pipeline.fit(train[train_cols], train[target])

    joblib.dump(pipeline, model_path)

    return pipeline
