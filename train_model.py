import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATA_DIR = 'data'
IMG_HEIGHT, IMG_WIDTH = 128, 128 # Must match what you collected or specify target
BATCH_SIZE = 128
EPOCHS = 50 # You might need more, e.g., 20-50, based on performance
NUM_CLASSES = 6 # Update this based on your number of gestures
# Ensure this matches your GESTURES list from collect_data.py
# GESTURES = ['play_pause', 'next_track', 'previous_track', 'volume_up', 'volume_down', 'stop']


# --- Data Loading and Augmentation ---
# ImageDataGenerator for training data (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to 0-1
    rotation_range=20, # Randomly rotate images
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2, # Shear transformations
    zoom_range=0.2, # Randomly zoom images
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest' # Fill new pixels with nearest value
)

# ImageDataGenerator for validation/test data (only rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # For multiple classes
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Store class labels
# This maps folder names to numerical indices (e.g., 'play_pause': 0, 'next_track': 1)
class_labels = {v: k for k, v in train_generator.class_indices.items()}
print("Class labels:", class_labels)


# --- Build the CNN Model (Simple architecture, you can use Transfer Learning later) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Helps prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Output layer, 'softmax' for multi-class classification
])

# --- Compile the Model ---
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print a summary of the model's layers

# --- Train the Model ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- Save the Trained Model ---
model.save('gesture_recognition_model.h5')
print("Model saved as 'gesture_recognition_model.h5'")

# --- Plotting Training History (Optional but good for understanding) ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()