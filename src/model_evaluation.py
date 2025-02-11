import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_model(
    test: pd.DataFrame, train_cols: List[str], target: str, model_path: str
) -> None:
    """
    Evaluates the model on the test data and prints metrics.

    Parameters:
    test (pd.DataFrame): The test data.
    train_cols (List[str]): The list of training columns.
    target (str): The target column name.
    model_path (str): The file path to the saved model.
    """
    pipeline = joblib.load(model_path)
    test_predictions = pipeline.predict(test[train_cols])
    test_target = test[target].values

    print_metrics(test_predictions, test_target)
    logging.info(test_predictions)


def print_metrics(predictions: np.ndarray, target: np.ndarray) -> None:
    """
    Prints evaluation metrics.

    Parameters:
    predictions (np.ndarray): The predicted values.
    target (np.ndarray): The true target values.
    """
    logging.info("RMSE: %s", np.sqrt(mean_squared_error(predictions, target)))
    logging.info("MAPE: %s", mean_absolute_percentage_error(predictions, target))
    logging.info("MAE : %s", mean_absolute_error(predictions, target))
