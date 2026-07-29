import pyautogui

# State variables (e.g. tracking virtual desktops)
state = {
    "on_desktop_2": False
}

def on_flip():
    """Action triggered when 'flip' gesture is detected."""
    print("👉 Executing action: Switching Desktop")
    if state["on_desktop_2"]:
        # Switch to Desktop 1
        pyautogui.hotkey('ctrl', 'win', 'left')
        state["on_desktop_2"] = False
    else:
        # Switch to Desktop 2
        pyautogui.hotkey('ctrl', 'win', 'right')
        state["on_desktop_2"] = True

def on_fist():
    """Action triggered when 'fist' gesture is detected."""
    print("👉 Executing action: Minimizing all windows")
    pyautogui.hotkey('win', 'd')

def on_unknown():
    """Action triggered if a gesture is detected but not mapped below."""
    print("👉 Unknown or unmapped gesture detected.")


# ==========================================
# 🚀 THE ACTION MAP
# ==========================================
# When the model predicts a string label (e.g. 'flip'), it looks here
# to figure out which Python function to run.
GESTURE_ACTIONS = {
    "flip": on_flip,
    "fist": on_fist
}
