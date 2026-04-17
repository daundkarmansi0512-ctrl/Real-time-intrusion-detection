import cv2
import json
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime
from collections import defaultdict, deque

# --- Configuration ---
VIDEO_PATH = "test.mp4"
ZONE_FILE = "zone.json"
LOG_FILE = "advanced_intrusion_log.csv"
MODEL_NAME = "yolov8n.pt"
LOITERING_THRESHOLD = 3.0  # Seconds before an intrusion becomes a 'loitering' alert

def load_zone(frame_shape):
    """Load polygon zone from JSON or create a default one based on frame size."""
    if os.path.exists(ZONE_FILE):
        try:
            with open(ZONE_FILE, 'r') as f:
                points = json.load(f)
            print("Loaded custom zone from zone.json")
            return np.array(points, np.int32)
        except Exception as e:
            print(f"Error loading zone.json: {e}")
            
    # Default zone: a trapezium positioned in the lower-middle part of the screen
    h, w = frame_shape[:2]
    p1 = [int(w * 0.2), int(h * 0.5)]
    p2 = [int(w * 0.8), int(h * 0.5)]
    p3 = [int(w * 0.9), int(h * 0.9)]
    p4 = [int(w * 0.1), int(h * 0.9)]
    
    print("Using default fallback zone.")
    return np.array([p1, p2, p3, p4], np.int32)

def draw_transparent_polygon(image, polygon, color, alpha=0.3):
    """Draw a filled polygon with transparency."""
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_dashboard_sidebar(image, width=320, color=(15, 15, 15), alpha=0.85):
    """Draw a translucent black sidebar on the right side for analytics."""
    h, w = image.shape[:2]
    sidebar = image.copy()
    sidebar_x = w - width
    cv2.rectangle(sidebar, (sidebar_x, 0), (w, h), color, -1)
    cv2.addWeighted(sidebar, alpha, image, 1 - alpha, 0, image)
    # Add a thin stylish border line separating the video and sidebar
    cv2.line(image, (sidebar_x, 0), (sidebar_x, h), (100, 100, 100), 2)
    return sidebar_x

def log_event(object_id, class_name, event_type, csv_writer, file_handle):
    """Log tracking events to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if csv_writer and file_handle:
        csv_writer.writerow([timestamp, object_id, class_name, event_type])
        file_handle.flush()
    print(f"[{event_type}] Time: {timestamp} | ID: {object_id} | Class: {class_name}")

def main():
    # 1. Initialize YOLOv8 Model
    print(f"Loading YOLOv8 model ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)
    names = model.names

    # 2. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # 3. Setup Zone & Logging
    zone_polygon = load_zone(frame.shape)
    
    file_exists = os.path.exists(LOG_FILE)
    log_file_handle = open(LOG_FILE, mode='a', newline='')
    csv_writer = csv.writer(log_file_handle)
    if not file_exists:
        csv_writer.writerow(['Timestamp', 'Track_ID', 'Class', 'Event_Type'])

    # 4. State Variables for Advanced Features
    track_history = defaultdict(lambda: deque(maxlen=40)) # Stores recent coordinates for comet tail
    zone_entry_times = {}       # id -> float timestamp (entry time)
    object_classes = {}         # id -> str class name (for logging upon exit)
    logged_loiterers = set()    # Track who has triggered the severe alert
    
    total_events_today = 0
    max_dwell_time = 0.0

    print("\n--- Starting Advanced Intrusion Detection System ---")
    print("Press 'q' or 'ESC' to exit.")

    prev_frame_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time

        display_frame = frame.copy()

        # Run tracking (persist=True ensures IDs carry over between frames)
        results = model.track(frame, persist=True, verbose=False)
        
        currently_inside_ids = set()
        zone_breached_this_frame = False

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                class_name = names[cls_idx]
                object_classes[track_id] = class_name
                
                # Check point for intrusion (center-bottom)
                center_x = int((x1 + x2) / 2)
                bottom_y = y2
                
                # Add to trajectory history
                track_history[track_id].append((center_x, bottom_y))
                
                # Check intersection (>=0 means inside polygon)
                dist = cv2.pointPolygonTest(zone_polygon, (center_x, bottom_y), False)
                is_inside = dist >= 0

                bbox_color = (0, 200, 0) # Green default (safe)
                label = f"{class_name} #{track_id}"
                
                if is_inside:
                    zone_breached_this_frame = True
                    currently_inside_ids.add(track_id)
                    
                    # Logic: Newly entered
                    if track_id not in zone_entry_times:
                        zone_entry_times[track_id] = current_time
                        total_events_today += 1
                        log_event(track_id, class_name, "ENTERED_ZONE", csv_writer, log_file_handle)
                    
                    # Logic: Dwell Time
                    dwell_time = current_time - zone_entry_times[track_id]
                    max_dwell_time = max(max_dwell_time, dwell_time)
                    
                    # Logic: Escalation to Loitering
                    if dwell_time > LOITERING_THRESHOLD:
                         bbox_color = (0, 0, 255) # Red for Severe Alert
                         label = f"#{track_id} LOITERING! ({dwell_time:.1f}s)"
                         
                         if track_id not in logged_loiterers:
                             logged_loiterers.add(track_id)
                             log_event(track_id, class_name, "SEVERE_LOITERING", csv_writer, log_file_handle)
                    else:
                         bbox_color = (0, 140, 255) # Orange for basic intrusion
                         label = f"#{track_id} INTRUDER ({dwell_time:.1f}s)"

                # Draw Trajectory Tail (Comet tail effect)
                points = list(track_history[track_id])
                for i in range(1, len(points)):
                    # Fade color intensity over the tail (oldest points are dimmer)
                    thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 2.5)
                    cv2.line(display_frame, points[i - 1], points[i], bbox_color, thickness)

                # Draw Bounding Box and dynamic label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.circle(display_frame, (center_x, bottom_y), 5, bbox_color, -1)
                cv2.putText(display_frame, label, (x1, max(15, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

        # Cleanup objects that have LEFT the zone
        for tid in list(zone_entry_times.keys()):
            if tid not in currently_inside_ids:
                dwell = current_time - zone_entry_times[tid]
                c_name = object_classes.get(tid, "Unknown")
                log_event(tid, c_name, f"EXITED_ZONE (Dwell: {dwell:.1f}s)", csv_writer, log_file_handle)
                
                # Remove from tracking memory
                del zone_entry_times[tid]
                if tid in logged_loiterers:
                    logged_loiterers.remove(tid)

        # Draw the Restricted Zone Overlay
        zone_color = (0, 0, 255) if zone_breached_this_frame else (255, 100, 0) # Red vs Blueish
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts], isClosed=True, color=zone_color, thickness=2)
        draw_transparent_polygon(display_frame, zone_polygon, zone_color, alpha=0.3 if zone_breached_this_frame else 0.15)

        # ----------------------------------------------------
        # Draw Advanced Analytics Dashboard Sidebar
        # ----------------------------------------------------
        sb_width = 300
        # If the frame is too small, shrink the sidebar
        if frame.shape[1] < 800:
             sb_width = 200
             
        sb_x = draw_dashboard_sidebar(display_frame, width=sb_width)
        
        # Dashboard Title
        cv2.putText(display_frame, "ANALYTICS HUB", (sb_x + 15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(display_frame, (sb_x + 15, 55), (sb_x + sb_width - 15, 55), (200, 200, 200), 1)

        # System Metrics
        cv2.putText(display_frame, f"System FPS: {fps:.1f}", (sb_x + 15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Total Events Today: {total_events_today}", (sb_x + 15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Record Dwell: {max_dwell_time:.1f}s", (sb_x + 15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Current Status
        status_txt = "STATUS: SAFE"
        status_col = (0, 255, 0)
        if len(logged_loiterers) > 0:
            status_txt = "SEVERE LOITERING"
            status_col = (0, 0, 255)
        elif zone_breached_this_frame:
            status_txt = "AREA BREACHED"
            status_col = (0, 140, 255)
            
        cv2.putText(display_frame, status_txt, (sb_x + 15, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_col, 2)

        # Live Feed of Active Intruders
        cv2.putText(display_frame, "Active Live Trackers:", (sb_x + 15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset = 280
        if len(zone_entry_times) == 0:
             cv2.putText(display_frame, "-- Zone Empty --", (sb_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            for tid, etime in zone_entry_times.items():
                dwell = current_time - etime
                c_name = object_classes.get(tid, "Obj")
                
                # Red text if loitering, Orange if normal intruder
                row_col = (0, 0, 255) if dwell > LOITERING_THRESHOLD else (0, 165, 255)
                row_txt = f"{c_name.upper()} #{tid} ( {dwell:.1f}s )"
                cv2.putText(display_frame, row_txt, (sb_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, row_col, 2)
                y_offset += 30

        # Show Output
        cv2.imshow("Advanced Intrusion Analytics", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if log_file_handle:
        log_file_handle.close()
    print("System shutdown complete.")

if __name__ == "__main__":
    main()
