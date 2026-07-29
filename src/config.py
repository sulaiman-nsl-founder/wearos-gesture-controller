import os

# Project root is the parent directory of 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Model Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.tflite")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.npy")

# Machine Learning Parameters
RECORD_DURATION = 2.0  # Seconds of data to record per gesture
NUM_TIMESTEPS = 100    # Target sequence length after interpolation
DOUBLE_TAP_TIMEOUT = 1.0 # Max seconds between taps for double-tap trigger

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
