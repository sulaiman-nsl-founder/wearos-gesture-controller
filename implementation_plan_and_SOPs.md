# Detailed Project Blueprint & AI Handoff Document
**Wear OS Machine Learning Pipeline via `touch-sdk`**

---

## 1. Project Status & Handoff Summary

This section details exactly what has been accomplished so far and what remains to be built.

### ✅ What is Done (Current State)
1. **SDK Integration:** The `touch-sdk` library is successfully installed and communicating with the Wear OS watch.
2. **Sensor Telemetry:** `test_watch.py` confirms that we can successfully intercept live `sensors.acceleration` data (X, Y, Z coordinates).
3. **Event Detection:** The `on_tap` event listener in the `Watch` class is actively working and can successfully register when the user taps the watch face.
4. **Graphing/Visualization:** A live Matplotlib thread was successfully implemented in `test_watch.py` to plot the incoming data in real-time.

### ⏳ What Needs to be Done (Next Immediate Tasks)
1. **[Task 1] Build the Data Collector (`collect_watch_data.py`):** We need to bridge the `on_tap` event to a recording mechanism that saves exactly $N$ seconds of data to a CSV file.
2. **[Task 2] Collect the Dataset:** The human user needs to run the Data Collector and physically perform the gestures to build a robust dataset.
3. **[Task 3] Build the Preprocessor (`preprocess.py`):** We need to write code to clean, normalize, and format the raw CSV files into NumPy arrays suitable for Deep Learning.
4. **[Task 4] Train the Model (`train_wearos_model.py`):** We need to define, train, and evaluate a 1D Convolutional Neural Network (Conv1D) or LSTM using TensorFlow, then export it as a `.tflite` model.
5. **[Task 5] Build Live Classifier (`live_watch_classifier.py`):** Finally, we need a script that runs the `touch-sdk` stream through the `.tflite` model in real-time to predict gestures as they happen.

---

## 2. Environment & Dependency Setup

Before proceeding, verify that the environment is fully equipped for ML training and data ingestion.

1. **Python Virtual Environment:**
   Ensure you are operating inside an isolated environment (e.g., `gesture_env`).
2. **Required Packages:**
   ```bash
   pip install touch-sdk tensorflow pandas numpy matplotlib scikit-learn
   ```
3. **Folder Structure Requirement:**
   Create a directory named `dataset/` in your root folder. This is where all CSVs will be stored.

---

## 3. SOP 1: Detailed Data Acquisition via `touch-sdk`

### Objective
Create `collect_watch_data.py` to record labeled training data.

### Technical Implementation Details
- **Imports:** `import time, csv, os, threading` from standard library, plus `from touch_sdk import Watch`.
- **State Management:** The `Watch` class needs a boolean flag (e.g., `self.is_recording = False`) and a timer `self.recording_start_time`.
- **The Prompt Loop:** 
  The main thread should run a `while True:` loop asking the user: `"Enter gesture name (or 'quit'): "`. Once entered, it prints `"Awaiting tap on watch..."`
- **The `on_tap` Logic:**
  When `on_tap` fires, if a gesture name is queued and `self.is_recording` is False, set `self.is_recording = True`, mark `self.recording_start_time = time.time()`, and clear the temporary buffer.
- **The `on_sensors` Logic:**
  If `self.is_recording` is True:
  1. Append the current `(time.time(), accel_x, accel_y, accel_z)` to `self.current_buffer`.
  2. Check if `time.time() - self.recording_start_time >= RECORD_DURATION` (e.g., 2.0 seconds).
  3. If duration is met, set `self.is_recording = False` and trigger the save function.
- **File Saving:** Write the buffer to `dataset/{gesture_name}_{timestamp}.csv`. The CSV should have headers: `timestamp, accel_x, accel_y, accel_z`.

---

## 4. SOP 2: Detailed Data Preprocessing

### Objective
Create `preprocess.py` to convert raw CSVs into ML-ready Tensors.

### Technical Implementation Details
- **Data Loading:** Use `os.listdir('dataset')` and `pandas.read_csv()` to iterate through all files. Extract the label from the filename (e.g., `swipe_1623.csv` -> label `swipe`).
- **Resampling/Interpolation:** Watch sensors don't always fire at perfect intervals. You must resample the timeline so every 2.0 second window has exactly the same number of data points (e.g., 100 rows). You can use `scipy.interpolate` or Pandas resampling.
- **Normalization:** Initialize a `scikit-learn` `StandardScaler()`. Fit it to the accelerometer data so the mean becomes 0 and standard deviation becomes 1. *Crucial: Save this scaler object (e.g., using `joblib` or `pickle`) because the live classifier will need it.*
- **Label Encoding:** Use `sklearn.preprocessing.LabelEncoder` to convert text labels ('swipe', 'punch') into integers (0, 1), and then `tf.keras.utils.to_categorical` to convert those to one-hot vectors.
- **Output:** Save the processed features as `X_train.npy`, `X_test.npy` and labels as `y_train.npy`, `y_test.npy`.

---

## 5. SOP 3: Detailed Model Training & Evaluation

### Objective
Create `train_wearos_model.py` to build the neural network and export a `.tflite` file.

### Technical Implementation Details
- **Architecture:** 
  Build a `tf.keras.Sequential` model:
  1. `Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_time_steps, 3))`
  2. `MaxPooling1D(pool_size=2)`
  3. `LSTM(64, return_sequences=False)` or another `Conv1D` + `Flatten()`.
  4. `Dense(32, activation='relu')`
  5. `Dropout(0.5)`
  6. `Dense(num_classes, activation='softmax')`
- **Compilation:** Use `optimizer='adam'` and `loss='categorical_crossentropy'`, monitoring `metrics=['accuracy']`.
- **Training:** Call `model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])`.
- **Conversion to TFLite:**
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open('gesture_model.tflite', 'wb') as f:
      f.write(tflite_model)
  ```

---

## 6. SOP 4: Detailed Real-Time Inference Loop

### Objective
Create `live_watch_classifier.py` to execute predictions live.

### Technical Implementation Details
- **Initialization:** Load `gesture_model.tflite` into a `tf.lite.Interpreter`. Load the `StandardScaler` saved during preprocessing.
- **The Live Buffer:** Implement a `collections.deque` with a `maxlen` equal to the required `num_time_steps` from training.
- **The Inference Trigger:** 
  When `on_tap` occurs, you can either:
  1. Record for the next 2.0 seconds, then predict (High accuracy, but delayed).
  2. Immediately take the *past* 2.0 seconds from the rolling buffer and predict (Instant, but requires the gesture to be finished *before* the tap).
  *Recommendation: Start with Option 1 for easier syncing.*
- **Executing Prediction:**
  Convert the buffer to a NumPy array, apply the `StandardScaler.transform()`, reshape to `(1, num_time_steps, 3)`, and pass it to the TFLite interpreter. Use `np.argmax()` on the output tensor to find the predicted class index, and map it back to the string label.
