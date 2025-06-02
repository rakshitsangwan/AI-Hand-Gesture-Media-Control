# AI Hand Gesture Media Control

## Project Overview

This project implements a real-time hand gesture recognition system that allows users to control media playback on their computer using specific hand movements. Leveraging computer vision and deep learning, the system captures live video from a webcam, detects hand landmarks using MediaPipe, and then classifies custom hand gestures using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Detected gestures are translated into media control commands (e.g., play/pause, volume up/down, next/previous track) via `pyautogui`.

This project demonstrates an end-to-end machine learning pipeline, from custom data collection and model training to real-time inference and system integration.

## Key Features

* **Real-time Gesture Recognition:** Detects and classifies hand gestures from a live webcam feed.
* **Media Control Integration:** Controls media playback (play/pause, volume, track navigation) using `pyautogui`.
* **Customizable Gestures:** Easily extendable to include more custom gestures by collecting new data.
* **Full ML Pipeline:** Covers data collection, model training, and deployment for inference.
* **User Feedback:** Displays recognized gesture and confidence level on the webcam feed.

## Technology Stack

* **Python 3.8+**
* **TensorFlow / Keras:** For building and training the deep learning model.
* **OpenCV (`opencv-python`):** For webcam access and image processing.
* **MediaPipe:** For efficient and accurate hand landmark detection.
* **Numpy:** For numerical operations.
* **`pyautogui`:** For simulating keyboard presses to control media.
* **`scikit-learn`:** For data splitting.
* **Git / GitHub:** For version control and project hosting.

## Project Structure
gesture_control_project/
├── data/
│   ├── train/
│   │   ├── play_pause/
│   │   ├── volume_up/
│   │   ├── volume_down/
│   │   ├── next_track/
│   │   ├── previous_track/
│   │   └── stop/
│   └── test/
│       ├── play_pause/
│       ├── volume_up/
│       ├── volume_down/
│       ├── next_track/
│       ├── previous_track/
│       └── stop/
├── venv/                   # Python Virtual Environment (ignored by Git)
├── collect_data.py         # Script for collecting gesture images
├── train_model.py          # Script for training the CNN model
├── media_control.py        # Script for real-time gesture recognition and media control
├── gesture_recognition_model.h5 # Trained model (ignored by Git, but will be generated)
├── .gitignore              # Specifies files/folders to ignore from Git
└── requirements.txt        # Lists all Python dependencies
└── README.md               # This file

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:


git clone [https://github.com/rakshitsangwan/AI-Hand-Gesture-Media-Control.git](https://github.com/rakshitsangwan/AI-Hand-Gesture-Media-Control.git)
cd AI-Hand-Gesture-Media-Control

(Replace rakshitsangwan with your actual GitHub username if you forked/renamed the repository.)
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
