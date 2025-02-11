class PathConfig:
    RAW_TRAIN_PATH = 'data/01_raw/train.csv'
    RAW_TEST_PATH = 'data/01_raw/test.csv'
    PROCESSED_TRAIN_PATH = 'data/02_processed/train_preprocessed.csv'
    PROCESSED_TEST_PATH = 'data/02_processed/test_preprocessed.csv'
    PREPROCESSOR_PATH = 'data/02_processed/preprocessor.joblib'
    MODEL_PATH = 'data/03_model/model.joblib'

class ModelParams:
    LEARNING_RATE = 0.01
    N_ESTIMATORS = 300
    MAX_DEPTH = 5
    LOSS = "absolute_error"

class Config:
    PATHS = PathConfig
    MODEL_PARAMS = ModelParams
