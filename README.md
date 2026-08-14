# Real-Time Intrusion Detection System

A real-time computer vision system that uses YOLOv8 object detection and tracking to monitor a user-defined restricted area. The system identifies tracked objects entering the zone, measures how long they remain inside, raises a loitering alert after a configurable time, and records intrusion events.

## Features

* YOLOv8-based object detection
* Persistent object tracking across video frames
* User-defined polygon-based restricted zones
* Zone entry and exit detection
* Dwell-time monitoring
* Configurable loitering threshold
* Object trajectory visualization
* Real-time analytics dashboard
* Event logging to CSV
* Separate zone-drawing utility

## Technologies

* Python
* OpenCV
* NumPy
* Ultralytics YOLOv8
* CSV logging

## How It Works

```text
Video Input
    ↓
YOLOv8 Detection
    ↓
Object Tracking
    ↓
Restricted-Zone Check
    ↓
Entry / Exit Detection
    ↓
Dwell-Time Monitoring
    ↓
Loitering Alert
    ↓
Event Logging
```

The system uses the bottom-center point of each detected object's bounding box to determine whether the object is inside the configured polygon.

## Project Structure

```text
real-time-intrusion-detection/
│
├── main.py
├── zone_drawer.py
├── zone.json
├── requirements.txt
├── README.md
└── .gitignore
```

Runtime files such as videos, model weights, and CSV logs are kept outside the repository.

## Requirements

Python 3.8 or newer.

Install the required packages:

```bash
pip install -r requirements.txt
```

The YOLOv8 model weights should be available locally as:

```text
yolov8n.pt
```

## Setup

### 1. Add the input video

Place the test video in the project directory and set its filename in `main.py`:

```python
VIDEO_PATH = "your_video.mp4"
```

### 2. Draw a restricted zone

Run:

```bash
python zone_drawer.py
```

Use the following controls:

* Left click to add polygon points
* Right click to remove the last point
* Press `s` to save the zone
* Press `q` or `Esc` to exit

The selected polygon is saved to:

```text
zone.json
```

### 3. Run the intrusion detection system

```bash
python main.py
```

The system displays:

* detected and tracked objects
* tracking IDs
* restricted zone
* current intrusion status
* dwell time
* loitering status
* system FPS
* active tracked objects

Press `q` or `Esc` to exit.

## Event Logging

The application records intrusion events locally in CSV format.

Events include:

```text
ENTERED_ZONE
SEVERE_LOITERING
EXITED_ZONE
```

Generated log files are ignored by Git and are not included in the repository.

## Configuration

The following settings can be adjusted in `main.py`:

```python
VIDEO_PATH = "your_video.mp4"
ZONE_FILE = "zone.json"
LOG_FILE = "advanced_intrusion_log.csv"
MODEL_NAME = "yolov8n.pt"
LOITERING_THRESHOLD = 3.0
```

`LOITERING_THRESHOLD` defines how long an object can remain inside the restricted zone before a loitering alert is generated.

## Limitations

* Detection and tracking performance depends on video quality, camera position, lighting, and occlusion.
* The restricted zone is configured for a specific camera view.
* The system is intended as a computer-vision prototype and has not been validated as a production security system.
* Video files and model weights are not included in the repository.

## Development Note

The project started from a reference implementation and was studied, modified, tested, and adapted for the current use case, including zone configuration, runtime behavior, display handling, and event monitoring.

## License

This project is intended for educational and portfolio use.
