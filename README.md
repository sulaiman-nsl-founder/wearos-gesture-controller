# WearOS Gesture Controller ⌚🤖

An end-to-end Machine Learning pipeline that bridges Wear OS smartwatch kinematics to Windows desktop automation. This project leverages the `touch-sdk` to stream live accelerometer data, processes it into machine-learning-ready tensors, and uses a lightweight TensorFlow Lite (TFLite) 1D Convolutional Neural Network to classify wrist gestures in real-time.

## Features
- **Data Acquisition Engine:** Record labeled accelerometer telemetry securely over Wi-Fi/Bluetooth to build your own dataset.
- **Automated ML Pipeline:** Includes scripts for resampling, normalizing (Scikit-Learn), and training a time-series Deep Learning model (TFLite).
- **Live OS Control:** Map gestures (e.g., `flip`, `fist`) to automated OS-level keyboard macros using `pyautogui`.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sulaiman-nsl-founder/wearos-gesture-controller.git
   cd wearos-gesture-controller
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv gesture_env
   .\gesture_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install touch-sdk tensorflow pandas numpy matplotlib scikit-learn pyautogui scipy
   ```
   *(Note: This project sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` internally to prevent protobuf runtime mismatch errors common in TF environments.)*

---

## 🚀 How to Use the Pipeline

### 1. Collect Data
Run the data collector to gather training data. It will prompt you for a gesture name. Double-tap your watch to trigger a 2-second recording window.
```bash
python collect_watch_data.py
```

### 2. Preprocess Data
Once you have collected several CSV files in the `dataset/` directory, process them into normalized NumPy arrays (`X_data.npy`, `y_data.npy`).
```bash
python preprocess.py
```

### 3. Train the Model
Train the Conv1D model on your dataset. This will output a highly optimized `gesture_model.tflite` model.
```bash
python train_wearos_model.py
```

### 4. Run Live OS Controller
Run the live inference engine. Double-tap the watch to arm the sensor, perform your gesture, and watch it control your Windows desktop!
- `Flip`: Toggles Virtual Desktops.
- `Fist`: Minimizes all windows (Shows Desktop).
```bash
python desktop_controller.py
```

---

## Architecture details
- `test_watch.py`: A visual graphing utility using `matplotlib` to verify your SDK connection.
- `implementation_plan_and_SOPs.md`: Detailed architectural design and Standard Operating Procedures.
