import time
import numpy as np
import threading
import joblib
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from touch_sdk import Watch
from scipy.interpolate import interp1d

# Config
RECORD_DURATION = 2.0
NUM_TIMESTEPS = 100

class LiveClassifierWatch(Watch):
    def __init__(self, model_path, scaler_path, classes_path):
        super().__init__()
        print("Loading Model...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Loading Scaler & Classes...")
        self.scaler = joblib.load(scaler_path)
        self.classes = np.load(classes_path)
        
        self.is_recording = False
        self.recording_start_time = 0
        self.current_buffer = []

    def on_tap(self):
        if not self.is_recording:
            print(f"🟢 TAP DETECTED! Analyzing next {RECORD_DURATION} seconds...")
            self.is_recording = True
            self.recording_start_time = time.time()
            self.current_buffer = []

    def on_sensors(self, sensors):
        if self.is_recording:
            current_time = time.time()
            accel = sensors.acceleration
            
            if accel:
                self.current_buffer.append([current_time, accel[0], accel[1], accel[2]])
            
            if current_time - self.recording_start_time >= RECORD_DURATION:
                self.is_recording = False
                self.predict()

    def predict(self):
        if len(self.current_buffer) < 2:
            print("Not enough data collected.")
            return

        print("Processing data...")
        # Convert buffer to numpy
        data = np.array(self.current_buffer)
        timestamps = data[:, 0]
        features = data[:, 1:]
        
        if timestamps[-1] == timestamps[0]:
            print("Invalid timestamps, skipping prediction.")
            return
            
        norm_time = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        target_time = np.linspace(0, 1, NUM_TIMESTEPS)
        
        resampled_features = []
        for i in range(features.shape[1]):
            interpolator = interp1d(norm_time, features[:, i], kind='linear', fill_value="extrapolate")
            resampled_features.append(interpolator(target_time))
            
        X = np.column_stack(resampled_features) # shape: (100, 3)
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # Reshape for TFLite model (1, 100, 3)
        X_input = X_scaled.reshape(1, NUM_TIMESTEPS, 3).astype(np.float32)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], X_input)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Decode
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100
        predicted_label = self.classes[class_idx]
        
        print(f"\n====================================")
        print(f"🎯 PREDICTION: {predicted_label.upper()} ({confidence:.2f}%)")
        print(f"====================================\n")
        print("Ready for next tap!")

def main():
    print("=========================================")
    print("   Live Gesture Classifier (TFLite)      ")
    print("=========================================")
    
    try:
        watch = LiveClassifierWatch(
            model_path="gesture_model.tflite",
            scaler_path="scaler.pkl",
            classes_path="classes.npy"
        )
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        print("Make sure you have run preprocess.py and train_wearos_model.py first!")
        return
        
    print("Connecting to Watch...")
    watch_thread = threading.Thread(target=watch.start, daemon=True)
    watch_thread.start()
    
    print("\nReady! Tap your watch, then perform a gesture.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
