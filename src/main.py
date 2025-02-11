from typing import Tuple

import pandas as pd

from config import Config
from data_preprocessing import load_data, preprocess_data
from model_evaluation import evaluate_model
from model_training import train_model


def main() -> None:
    """
    Main function that executes the training pipeline. It loads and preprocess data, then trains evaluates the model.
    """
    # Load
    train, test = load_data(Config.PATHS.RAW_TRAIN_PATH, Config.PATHS.RAW_TEST_PATH)

    # Preprocess
    categorical_cols = ["type", "sector"]
    target = "price"
    preprocessor, train_cols = preprocess_data(
        train, categorical_cols, target, Config.PATHS.PREPROCESSOR_PATH
    )

    # Train
    pipeline = train_model(
        train,
        train_cols,
        target,
        Config.PATHS.PREPROCESSOR_PATH,
        Config.PATHS.MODEL_PATH,
    )

    # Evaluate
    evaluate_model(test, train_cols, target, Config.PATHS.MODEL_PATH)


if __name__ == "__main__":
    main()
