import cv2
import tensorflow as tf
import numpy as np
import pyautogui
import time

# --- Configuration ---
MODEL_PATH = 'gesture_recognition_model.h5'
IMG_HEIGHT, IMG_WIDTH = 128, 128 # Must match training input size

# Update this list based on the order your classes were trained in!
# Check the output of train_model.py for 'Class labels:'
# E.g., if train_model.py says {'play_pause': 0, 'next_track': 1, ...}
GESTURE_LABELS = ['next_track', 'play_pause', 'previous_track', 'stop', 'volume_down', 'volume_up']

# Define the key mapping for each gesture
# These are standard media keys (e.g., on a laptop keyboard)
# Adjust as needed for your system or specific media player
KEY_MAPPING = {
    'play_pause': 'a',  # Or 'space' depending on your media player
    'next_track': 'b',
    'previous_track': 'c',
    'volume_up': 'd',
    'volume_down': 'e',
    'stop': 'f',
}

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'gesture_recognition_model.h5' exists after running train_model.py")
    exit()

# --- Open Webcam ---
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- Gesture Media Control ---")
print("Show your gestures to control media.")
print("Press 'q' to quit.")

# Variables for gesture smoothing and cooldown
last_gesture = None
gesture_cooldown_start_time = 0
COOLDOWN_DURATION = 2 # seconds, to prevent rapid multiple commands

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1) # Flip frame horizontally (common for webcams)

    # Pre-process the frame for model input
    img_array = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_gesture = GESTURE_LABELS[predicted_class_index]
    confidence = predictions[predicted_class_index] * 100

    # Display prediction on screen
    text = f"Gesture: {predicted_gesture} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Gesture Control', frame)

    # --- Media Control Logic ---
    current_time = time.time()
    if current_time - gesture_cooldown_start_time > COOLDOWN_DURATION:
        if confidence > 85: # Only act if confidence is high enough (adjust this threshold)
            if predicted_gesture != last_gesture: # Only act if gesture changes
                if predicted_gesture in KEY_MAPPING:
                    key_to_press = KEY_MAPPING[predicted_gesture]
                    print(f"Detected: {predicted_gesture}. Pressing '{key_to_press}'")
                    pyautogui.press(key_to_press)
                    last_gesture = predicted_gesture
                    gesture_cooldown_start_time = current_time # Reset cooldown
        else:
            last_gesture = None # Reset last gesture if confidence drops

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()