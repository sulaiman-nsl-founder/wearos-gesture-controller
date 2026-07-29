import time
import numpy as np
import threading
import joblib
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from touch_sdk import Watch
from scipy.interpolate import interp1d

from . import config
from . import actions

class DesktopControllerWatch(Watch):
    def __init__(self):
        super().__init__()
        print("Loading Model...")
        self.interpreter = tf.lite.Interpreter(model_path=config.MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Loading Scaler & Classes...")
        self.scaler = joblib.load(config.SCALER_PATH)
        self.classes = np.load(config.CLASSES_PATH)
        
        self.is_recording = False
        self.recording_start_time = 0
        self.current_buffer = []
        self.last_tap_time = 0

    def on_tap(self):
        current_time = time.time()
        time_since_last = current_time - self.last_tap_time
        self.last_tap_time = current_time
        
        if time_since_last < config.DOUBLE_TAP_TIMEOUT and not self.is_recording:
            print(f"🟢 DOUBLE TAP DETECTED! Analyzing next {config.RECORD_DURATION} seconds...")
            self.is_recording = True
            self.recording_start_time = time.time()
            self.current_buffer = []
        elif not self.is_recording:
            print(f"Single tap detected. Tap again within {config.DOUBLE_TAP_TIMEOUT}s to trigger gesture recording.")

    def on_sensors(self, sensors):
        if self.is_recording:
            current_time = time.time()
            accel = sensors.acceleration
            
            if accel:
                self.current_buffer.append([current_time, accel[0], accel[1], accel[2]])
            
            if current_time - self.recording_start_time >= config.RECORD_DURATION:
                self.is_recording = False
                self.predict()

    def predict(self):
        if len(self.current_buffer) < 2:
            print("Not enough data collected.")
            return

        print("Processing data...")
        data = np.array(self.current_buffer)
        timestamps = data[:, 0]
        features = data[:, 1:]
        
        if timestamps[-1] == timestamps[0]:
            return
            
        norm_time = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        target_time = np.linspace(0, 1, config.NUM_TIMESTEPS)
        
        resampled_features = []
        for i in range(features.shape[1]):
            interpolator = interp1d(norm_time, features[:, i], kind='linear', fill_value="extrapolate")
            resampled_features.append(interpolator(target_time))
            
        X = np.column_stack(resampled_features)
        X_scaled = self.scaler.transform(X)
        X_input = X_scaled.reshape(1, config.NUM_TIMESTEPS, 3).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], X_input)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100
        predicted_label = self.classes[class_idx]
        
        print(f"\n🎯 PREDICTION: {predicted_label.upper()} ({confidence:.2f}%)")
        
        # Execute the mapped action
        action_func = actions.GESTURE_ACTIONS.get(predicted_label.lower())
        if action_func:
            action_func()
        else:
            actions.on_unknown()
            
        print("Ready for next double-tap!\n")


def run():
    print("=========================================")
    print("      Gesture OS Controller              ")
    print("=========================================")
    
    try:
        watch = DesktopControllerWatch()
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return
        
    print("Connecting to Watch...")
    watch_thread = threading.Thread(target=watch.start, daemon=True)
    watch_thread.start()
    
    print("\nReady! DOUBLE-TAP your watch, then perform a gesture.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    run()
