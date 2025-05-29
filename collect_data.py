import cv2
import os
import time

# --- Configuration ---
DATA_DIR = 'data' # This folder will store your images
GESTURES = ['play_pause', 'next_track', 'previous_track', 'volume_up', 'volume_down', 'stop'] # Your chosen gestures
NUM_SAMPLES_PER_GESTURE = 2000 # Aim for at least 200, more is better (e.g., 500-1000)
IMG_SIZE = (128, 128) # Standard size for model input

# Create directories if they don't exist
for gesture in GESTURES:
    path = os.path.join(DATA_DIR, 'train', gesture)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(DATA_DIR, 'test', gesture)
    os.makedirs(path, exist_ok=True)

print(f"Folders created in '{DATA_DIR}' for gestures: {GESTURES}")
print("Press 'q' to quit during collection.")

# Open webcam
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for gesture_idx, gesture_name in enumerate(GESTURES):
    print(f"\n--- Collecting data for: '{gesture_name}' ---")
    print("Prepare to show your hand gesture in 3 seconds...")
    time.sleep(3) # Give yourself time to get ready

    sample_count = 0
    while sample_count < NUM_SAMPLES_PER_GESTURE:
        ret, frame = cap.read() # Read a frame from the webcam
        if not ret:
            print("Failed to grab frame.")
            break

        # You might want to flip the frame if your camera mirrors it
        frame = cv2.flip(frame, 1)

        # Display the current frame (optional, helps you see what's being captured)
        cv2.imshow('Collecting Gestures', frame)

        # Wait for 's' key to save a sample, or 'q' to quit
        key = cv2.waitKey(1) & 0xFF # waitKey(1) means wait 1ms

        if key == ord('s'): # Press 's' to save a sample
            # Resize and save the frame
            # You might want to crop around the hand later, but for now, save the whole frame
            # For simplicity, we will assume the hand is roughly centered.
            img_path = os.path.join(DATA_DIR, 'train', gesture_name, f'{gesture_name}_{sample_count}.jpg')
            # Save resized image (adjust to IMG_SIZE)
            # For now, let's just save the full frame. We'll resize later when loading.
            cv2.imwrite(img_path, frame)
            print(f"Saved {sample_count + 1}/{NUM_SAMPLES_PER_GESTURE} for {gesture_name}")
            sample_count += 1
            time.sleep(0.1) # Small delay to avoid saving too many images from same pose

        elif key == ord('q'): # Press 'q' to quit
            break # Exit the loop and the program

    if key == ord('q'): # If 'q' was pressed, break outer loop too
        break

# Release webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")