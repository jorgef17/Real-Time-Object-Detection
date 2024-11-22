# Real-Time Object Detection with YOLOv8

This project demonstrates real-time object detection using the YOLOv8 model. The model is used to detect various objects in a video stream from a webcam. It is built using the `Ultralytics YOLO` library and OpenCV for live video processing.

## Features
- Real-time object detection using YOLOv8.
- Webcam video feed for object detection.
- Detection results saved to a `detections.txt` file.
- Screenshots can be captured by pressing the spacebar during video streaming.

## Requirements
- Python 3.7+
- Install required libraries using the following command:



## Usage

1. **Clone the repository:**

    ```
    git clone https://github.com/jorgef17/Real-Time-Object-Detection.git
    cd Real-Time-Object-Detection
    pip install -r requirements.txt
    ```

2. **Run the script to start object detection:**

    ```
    python main.py
    ```

3. **Press the space bar** to save a screenshot.

4. **Press "q"** to exit the program.


## Files

- `main.py`: Main Python script to run object detection.
- `detections.txt`: File where detected objects are logged.
- `screenshots/`: Folder where screenshots are saved.
