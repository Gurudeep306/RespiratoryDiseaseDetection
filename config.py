# Configuration File for Respiratory Disease Detection

## Data Configuration
DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "results/models"
PLOTS_DIR = "results/plots"

## Audio Processing
SAMPLE_RATE = 22050
MIN_SILENCE_LEN_MS = 500
SILENCE_THRESH_DB = -40
NUM_CHUNKS = 50

## Feature Extraction
N_MFCC = 13
WAVELET_TYPE = "db4"
WAVELET_LEVEL = 3

## Model Training
TEST_SIZE = 0.3
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 15
SVM_KERNEL = "rbf"
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_DROPOUT = 0.2
XGBOOST_N_TRIALS = 50

## Ensemble Configuration
ENSEMBLE_WEIGHTS = {
    "rf": 0.33,
    "svm": 0.33,
    "xgb": 0.34
}

## Data Balancing
USE_SMOTE = True
SMOTE_RANDOM_STATE = 42

## Feature Processing
REMOVE_LOW_VARIANCE = True
LOW_VARIANCE_THRESHOLD = 0.01

## Evaluation
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "g_mean",
    "auc"
]

## Logging
VERBOSE = True
SAVE_PLOTS = True
SAVE_MODELS = True

## Class Labels
CLASS_LABELS = {
    0: "HC (Healthy Controls)",
    1: "PD (Parkinson's Disease)"
}

## Random Seeds (for reproducibility)
NUMPY_SEED = 42
SKLEARN_SEED = 42
TENSORFLOW_SEED = 42
