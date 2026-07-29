import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import glob
import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

DATASET_DIR = "dataset"
NUM_TIMESTEPS = 100 # Standardize every sample to 100 rows

def resample_data(df, target_steps):
    """Resamples the sequence using linear interpolation to target_steps length."""
    # First column is timestamp, and next are accel x,y,z
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    
    # Normalize timestamps to 0...1 to ensure interpolation domain matches
    if len(timestamps) < 2 or timestamps[-1] == timestamps[0]:
        return np.zeros((target_steps, features.shape[1]))
        
    norm_time = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    target_time = np.linspace(0, 1, target_steps)
    
    # Interpolate each feature column
    resampled_features = []
    for i in range(features.shape[1]):
        interpolator = interp1d(norm_time, features[:, i], kind='linear', fill_value="extrapolate")
        resampled_features.append(interpolator(target_time))
        
    return np.column_stack(resampled_features)

def preprocess_data():
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not all_files:
        print("No CSV files found in dataset/")
        return
        
    X_list = []
    y_list = []
    
    for file in all_files:
        # Extract label from filename (e.g. fist_1785...csv -> fist)
        basename = os.path.basename(file)
        label = basename.split('_')[0]
        
        df = pd.read_csv(file)
        if len(df) < 2:
            print(f"Skipping {file}, not enough data points.")
            continue
            
        resampled = resample_data(df, NUM_TIMESTEPS)
        X_list.append(resampled)
        y_list.append(label)
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Loaded data shape: X={X.shape}, y={y.shape}")
    
    # Normalize features
    # Since X is 3D (samples, timesteps, features), we must reshape to 2D for scaler
    num_samples, num_steps, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    # Reshape back to 3D
    X_scaled = X_flat_scaled.reshape(num_samples, num_steps, num_features)
    
    # Save the scaler for live inference
    joblib.dump(scaler, "scaler.pkl")
    print("Saved StandardScaler to scaler.pkl")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder classes for inference mapping
    np.save("classes.npy", label_encoder.classes_)
    print(f"Classes found: {label_encoder.classes_}")
    
    y_categorical = to_categorical(y_encoded)
    
    # Shuffle dataset
    indices = np.arange(X_scaled.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    X_scaled = X_scaled[indices]
    y_categorical = y_categorical[indices]
    
    # Save preprocessed data
    np.save("X_data.npy", X_scaled)
    np.save("y_data.npy", y_categorical)
    print("Saved X_data.npy and y_data.npy successfully.")

if __name__ == "__main__":
    preprocess_data()
