import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

def train_model():
    # Load preprocessed data
    print("Loading preprocessed data...")
    try:
        X = np.load("X_data.npy")
        y = np.load("y_data.npy")
        classes = np.load("classes.npy")
    except FileNotFoundError:
        print("Preprocessed data not found. Run preprocess.py first.")
        return
        
    print(f"Data shape: X={X.shape}, y={y.shape}")
    num_classes = len(classes)
    num_timesteps = X.shape[1]
    num_features = X.shape[2]
    
    # Simple validation split since dataset might be small
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Build the model
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
    model.summary()
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    
    # Export to TFLite
    print("\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optional: Quantization for smaller/faster model
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()
    
    with open('gesture_model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("✅ Model successfully trained and saved as 'gesture_model.tflite'")

if __name__ == "__main__":
    train_model()
