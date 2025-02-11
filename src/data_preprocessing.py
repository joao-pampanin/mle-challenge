from typing import Tuple

import joblib
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test CSV files into pandas DataFrames.

    Parameters:
    train_path (str): The file path to the training data CSV file.
    test_path (str): The file path to the test data CSV file.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test data as pandas DataFrames.
    """
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")
    except pd.errors.EmptyDataError:
        raise ValueError("No data: The file is empty")
    except pd.errors.ParserError:
        raise ValueError("Parsing error: The file could not be parsed")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    return train, test


def preprocess_data(
    train: pd.DataFrame,
    categorical_cols: list[str],
    target: str,
    preprocessor_path: str,
) -> Tuple[ColumnTransformer, list[str]]:
    """
    Preprocesses the data using Target Encoding.

    Parameters:
    train (pd.DataFrame): The training data.
    categorical_cols (list[str]): The list of categorical columns to be encoded.
    target (str): The model target column name.
    preprocessor_path (str): The file path to save the preprocessor.

    Returns:
    Tuple[ColumnTransformer, list[str]]: The fitted preprocessor and the list of training columns.
    """
    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_transformer, categorical_cols)]
    )

    train_cols = [col for col in train.columns if col not in ["id", target]]

    preprocessor.fit(train[train_cols], train[target])

    joblib.dump(preprocessor, preprocessor_path)

    return preprocessor, train_cols
