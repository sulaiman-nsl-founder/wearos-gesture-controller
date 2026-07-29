import os
import time
import csv
import threading
from touch_sdk import Watch

# Configuration
RECORD_DURATION = 2.0  # seconds to record after tap
DATASET_DIR = "dataset"

class DataCollectorWatch(Watch):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.recording_start_time = 0
        self.current_buffer = []
        self.current_gesture = None
        
        # Ensure dataset directory exists
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)

    def set_gesture(self, gesture_name):
        self.current_gesture = gesture_name
        print(f"\n[READY] Gesture set to '{gesture_name}'. Awaiting tap on watch to start recording...")

    def on_tap(self):
        if self.current_gesture and not self.is_recording:
            print(f"🟢 TAP DETECTED! Recording '{self.current_gesture}' for {RECORD_DURATION} seconds...")
            self.is_recording = True
            self.recording_start_time = time.time()
            self.current_buffer = []
        elif not self.current_gesture:
            print("Tap detected, but no gesture name is set. Please enter a gesture name in the console.")

    def on_sensors(self, sensors):
        if self.is_recording:
            current_time = time.time()
            accel = sensors.acceleration
            
            if accel:
                # Append data row: timestamp, accel_x, accel_y, accel_z
                self.current_buffer.append([current_time, accel[0], accel[1], accel[2]])
            
            # Check if we have recorded for the desired duration
            if current_time - self.recording_start_time >= RECORD_DURATION:
                self.is_recording = False
                self.save_buffer()

    def save_buffer(self):
        if not self.current_buffer:
            print("⚠️ No data collected during the recording window.")
            return

        timestamp_str = str(int(time.time() * 1000))
        filename = f"{self.current_gesture}_{timestamp_str}.csv"
        filepath = os.path.join(DATASET_DIR, filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'accel_x', 'accel_y', 'accel_z'])
            writer.writerows(self.current_buffer)
        
        print(f"✅ Successfully saved {len(self.current_buffer)} data points to {filepath}")
        
        # Reset gesture to force user to confirm readiness again
        self.current_gesture = None
        print("\n---")
        print(f"Ready for next sample. Press Enter in console to record '{filepath.split('_')[0].split(os.sep)[-1]}' again, or type a new one.")


def main():
    watch = DataCollectorWatch()
    
    # Run the watch listener in a background thread
    watch_thread = threading.Thread(target=watch.start, daemon=True)
    watch_thread.start()

    print("=========================================")
    print("      Touch SDK ML Data Collector        ")
    print("=========================================")
    print("Connecting to Watch...")
    time.sleep(2) # Give it a moment to connect

    last_gesture = ""

    while True:
        try:
            prompt_text = f"Enter gesture name (or press Enter to use '{last_gesture}', or 'quit'): " if last_gesture else "Enter gesture name (or 'quit'): "
            user_input = input(prompt_text).strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if user_input:
                last_gesture = user_input
            elif not user_input and last_gesture:
                pass # Use last_gesture
            else:
                print("Please enter a valid gesture name first.")
                continue
                
            watch.set_gesture(last_gesture)
            
            # Wait until recording is finished before prompting again
            while watch.current_gesture is not None:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
