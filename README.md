# Advanced Zone Intrusion Analytics (YOLOv8 & OpenCV)

This is a comprehensive, advanced computer vision project built for an internship showcase. It upgrades standard object tracking by turning it into a real-time **Analytics and Monitoring Dashboard**. 

## Advanced Features
- **Intelligent Loitering Detection:** Calculates precise "Dwell Time" for any object entering the zone. Basic intrusions trigger an orange warning, but objects loitering for longer than 3 seconds escalate to a **SEVERE** Red Alert.
- **Comet-Tail Trajectories:** Utilizing OpenCV polygons and tracking memory, the system draws beautiful fading tracking lines behind each object, visualizing their movement paths.
- **Real-Time Analytics Dashboard:** Features a sleek Translucent Heads-Up Display (HUD) on the sidebar showing system FPS, Event counting loops, Record Dwell Times, and a live-updating list containing the active trackers inside the zone.
- **Robust Event Logging:** Logs complex events (`ENTERED_ZONE`, `EXITED_ZONE`, `SEVERE_LOITERING`) to an `advanced_intrusion_log.csv` file.

## Prerequisites

Ensure you have Python 3.8+ installed.

1. Open your terminal in this directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### 1. (Optional) Draw a Custom Restricted Zone
By default, the application uses a predefined region at the bottom of the screen. If you'd like to draw your own custom zone, run the drawer tool:

```bash
python zone_drawer.py
```
- **Left Click:** Add points.
- **Right Click:** Remove the last point.
- **Press 's':** Save polygon and exit.

### 2. Run the Advanced Analytics System
```bash
python main.py
```

- Watch as YOLOv8 detects and tracks objects, building a comet tail of movement behind them.
- Notice the **Sidebar Dashboard** dynamically updating as objects enter the restricted zone.
- Watch the labels switch from `INTRUDER` to `LOITERING!` as the Timer crosses 3.0 seconds. 
- **Press 'q' or 'ESC'** to safely exit.

## Modifying the App
- To change how long someone must wait before triggering a loitering alert, open `main.py` and modify `LOITERING_THRESHOLD = 3.0`.
- To focus *only* on people, modify `results = model.track(..., classes=[0])`.

*Built using PyTorch, Ultralytics, and custom OpenCV HUD generation.*
