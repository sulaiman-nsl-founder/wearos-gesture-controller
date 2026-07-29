# WearOS Gesture Controller ⌚🤖

### Live Demonstrations

| Fist Gesture | Flip Gesture |
| :---: | :---: |
| ![Fist Gesture](demonstration/fist.gif) | ![Flip Gesture](demonstration/flip.gif) |

An end-to-end Machine Learning pipeline that bridges Wear OS smartwatch kinematics to Windows desktop automation. This project leverages the `touch-sdk` to stream live accelerometer data, processes it into machine-learning-ready tensors, and uses a lightweight TensorFlow Lite (TFLite) 1D Convolutional Neural Network to classify wrist gestures in real-time.

## Features
- **Data Acquisition Engine:** Record labeled accelerometer telemetry securely to build your own dataset.
- **Automated ML Pipeline:** Includes modular scripts for resampling, normalizing (Scikit-Learn), and training a time-series Deep Learning model (TFLite).
- **Live OS Control:** Map gestures (e.g., `flip`, `fist`) to automated OS-level keyboard macros using `pyautogui`.

---

## ⌚ Watch Setup & Bluetooth Connection

Before running the Python pipeline on your PC, you must configure your Wear OS smartwatch.

1. **Install the APK:** 
   Locate the installation file inside the `apk_for_watch` folder in this repository. You will need to install this APK onto your Wear OS watch.
   *(Insert link on how to sideload/install an APK to a Wear OS watch here)*
   
2. **Enable the SDK:** 
   Once the app is installed, open it on your watch and **turn on the SDK toggle**.
   
3. **Connect via Bluetooth:** 
   Ensure your PC's Bluetooth is turned on, and pair your Windows PC directly with your Wear OS watch via Bluetooth settings.

---

## 🛠️ PC Installation & Setup

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

---

## 🚀 How to Use the Pipeline

This project is built on a modular architecture. You control everything through the central `main.py` entry point.

### 1. Collect Data
Run the data collector to gather training data. It will prompt you for a gesture name. Double-tap your watch to trigger a 2-second recording window.
```bash
python main.py collect
```

### 2. Preprocess Data
Once you have collected data in the `data/raw/` directory, process them into normalized NumPy arrays.
```bash
python main.py process
```

### 3. Train the Model
Train the Conv1D model on your dataset. This will output a highly optimized `gesture_model.tflite` model in the `models/` directory.
```bash
python main.py train
```

### 4. Run Live OS Controller
Run the live inference engine. Double-tap the watch to arm the sensor, perform your gesture, and watch it control your Windows desktop!
- `Flip`: Toggles Virtual Desktops.
- `Fist`: Minimizes all windows (Shows Desktop).
```bash
python main.py run
```

*(To map custom gestures to your PC, edit the `GESTURE_ACTIONS` dictionary inside `src/actions.py`)*
