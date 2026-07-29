import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import glob
import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

from . import config

def resample_data(df, target_steps):
    """Resamples the sequence using linear interpolation to target_steps length."""
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    
    if len(timestamps) < 2 or timestamps[-1] == timestamps[0]:
        return np.zeros((target_steps, features.shape[1]))
        
    norm_time = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    target_time = np.linspace(0, 1, target_steps)
    
    resampled_features = []
    for i in range(features.shape[1]):
        interpolator = interp1d(norm_time, features[:, i], kind='linear', fill_value="extrapolate")
        resampled_features.append(interpolator(target_time))
        
    return np.column_stack(resampled_features)

def run():
    print("=========================================")
    print("      Data Preprocessor Engine           ")
    print("=========================================")
    all_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {config.RAW_DATA_DIR}")
        return
        
    X_list = []
    y_list = []
    
    for file in all_files:
        basename = os.path.basename(file)
        label = basename.split('_')[0]
        
        df = pd.read_csv(file)
        if len(df) < 2:
            print(f"Skipping {file}, not enough data points.")
            continue
            
        resampled = resample_data(df, config.NUM_TIMESTEPS)
        X_list.append(resampled)
        y_list.append(label)
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Loaded data shape: X={X.shape}, y={y.shape}")
    
    num_samples, num_steps, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    X_scaled = X_flat_scaled.reshape(num_samples, num_steps, num_features)
    
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Saved StandardScaler to {config.SCALER_PATH}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    np.save(config.CLASSES_PATH, label_encoder.classes_)
    print(f"Classes discovered: {label_encoder.classes_}")
    
    y_categorical = to_categorical(y_encoded)
    
    indices = np.arange(X_scaled.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    X_scaled = X_scaled[indices]
    y_categorical = y_categorical[indices]
    
    x_save_path = os.path.join(config.PROCESSED_DATA_DIR, "X_data.npy")
    y_save_path = os.path.join(config.PROCESSED_DATA_DIR, "y_data.npy")
    
    np.save(x_save_path, X_scaled)
    np.save(y_save_path, y_categorical)
    print(f"Saved {x_save_path} and {y_save_path} successfully.")

if __name__ == "__main__":
    run()
