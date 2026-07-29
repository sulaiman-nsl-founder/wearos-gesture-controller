import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from . import config

def run():
    print("=========================================")
    print("      Model Training Engine              ")
    print("=========================================")
    print("Loading preprocessed data...")
    try:
        x_path = os.path.join(config.PROCESSED_DATA_DIR, "X_data.npy")
        y_path = os.path.join(config.PROCESSED_DATA_DIR, "y_data.npy")
        
        X = np.load(x_path)
        y = np.load(y_path)
        classes = np.load(config.CLASSES_PATH)
    except FileNotFoundError as e:
        print(f"Data not found: {e}. Run preprocessing first.")
        return
        
    print(f"Data shape: X={X.shape}, y={y.shape}")
    num_classes = len(classes)
    num_timesteps = X.shape[1]
    num_features = X.shape[2]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_timesteps, num_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("\nStarting training...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    
    print("\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(config.MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Model successfully trained and saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    run()
