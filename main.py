import cv2
import json
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime
from collections import defaultdict, deque

# Configuration
VIDEO_PATH = "test3.mp4"
ZONE_FILE = "zone.json"
LOG_FILE = "advanced_intrusion_log.csv"
MODEL_NAME = "yolov8n.pt"
LOITERING_THRESHOLD = 3.0  # Seconds before an intrusion becomes a 'loitering' alert


def resize_for_display(frame, max_width=1280, max_height=720):
    """Resize frame for display only, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
            
    # Default zone
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
    # Initialize model
    print(f"Loading YOLOv8 model ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)
    names = model.names

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Load zone and log setup
    zone_polygon = load_zone(frame.shape)
    
    file_exists = os.path.exists(LOG_FILE)
    log_file_handle = open(LOG_FILE, mode='a', newline='')
    csv_writer = csv.writer(log_file_handle)
    if not file_exists:
        csv_writer.writerow(['Timestamp', 'Track_ID', 'Class', 'Event_Type'])

    # Tracking state
    track_history = defaultdict(lambda: deque(maxlen=40)) # Stores recent coordinates for comet tail
    zone_entry_times = {}       # id-float timestamp (entry time)
    object_classes = {}         # id-str class name (for logging upon exit)
    logged_loiterers = set()    # Track who has triggered the severe alert
    
    total_events_today = 0
    max_dwell_time = 0.0

    print("\n--- Starting Advanced Intrusion Detection System ---")
    print("Press 'q' or 'ESC' to exit.")

    cv2.namedWindow("Advanced Intrusion Analytics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Advanced Intrusion Analytics", 1280, 720)

    prev_frame_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed = current_time - prev_frame_time
        fps = 1 / elapsed if elapsed > 0 else 0
        prev_frame_time = current_time

        display_frame = frame.copy()

        # Detect and track objects
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
                
                # Use the bottom-center point of the box
                center_x = int((x1 + x2) / 2)
                bottom_y = y2
                
                # Store recent positions
                track_history[track_id].append((center_x, bottom_y))
                
                # Check whether the point is inside the zone
                dist = cv2.pointPolygonTest(zone_polygon, (center_x, bottom_y), False)
                is_inside = dist >= 0

                bbox_color = (0, 200, 0) # Green default (safe)
                label = f"{class_name} #{track_id}"
                
                if is_inside:
                    zone_breached_this_frame = True
                    currently_inside_ids.add(track_id)
                    
                    # New zone entry
                    if track_id not in zone_entry_times:
                        zone_entry_times[track_id] = current_time
                        total_events_today += 1
                        log_event(track_id, class_name, "ENTERED_ZONE", csv_writer, log_file_handle)
                    
                    # Dwell time
                    dwell_time = current_time - zone_entry_times[track_id]
                    max_dwell_time = max(max_dwell_time, dwell_time)
                    
                    # Loitering check
                    if dwell_time > LOITERING_THRESHOLD:
                         bbox_color = (0, 0, 255) # Red for Severe Alert
                         label = f"#{track_id} LOITERING! ({dwell_time:.1f}s)"
                         
                         if track_id not in logged_loiterers:
                             logged_loiterers.add(track_id)
                             log_event(track_id, class_name, "SEVERE_LOITERING", csv_writer, log_file_handle)
                    else:
                         bbox_color = (0, 140, 255) # Orange for basic intrusion
                         label = f"#{track_id} INTRUDER ({dwell_time:.1f}s)"

                # Draw trajectory
                points = list(track_history[track_id])
                for i in range(1, len(points)):
                    # Fade color intensity over the tail (oldest points are dimmer)
                    thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 2.5)
                    cv2.line(display_frame, points[i - 1], points[i], bbox_color, thickness)

                # Draw detection
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.circle(display_frame, (center_x, bottom_y), 5, bbox_color, -1)
                cv2.putText(display_frame, label, (x1, max(15, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

        # Handle objects leaving the zone
        for tid in list(zone_entry_times.keys()):
            if tid not in currently_inside_ids:
                dwell = current_time - zone_entry_times[tid]
                c_name = object_classes.get(tid, "Unknown")
                log_event(tid, c_name, f"EXITED_ZONE (Dwell: {dwell:.1f}s)", csv_writer, log_file_handle)
                
                # Remove from tracking memory
                del zone_entry_times[tid]
                if tid in logged_loiterers:
                    logged_loiterers.remove(tid)

        # Draw restricted zone
        zone_color = (0, 0, 255) if zone_breached_this_frame else (255, 100, 0) # Red vs Blueish
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts], isClosed=True, color=zone_color, thickness=2)
        draw_transparent_polygon(display_frame, zone_polygon, zone_color, alpha=0.3 if zone_breached_this_frame else 0.15)

        # Draw dashboard Sidebar
        sb_width = 300
        # Adjust sidebar width for small frames
        if frame.shape[1] < 800:
             sb_width = 200
             
        sb_x = draw_dashboard_sidebar(display_frame, width=sb_width)
        
        # Dashboard Title
        cv2.putText(display_frame, "ANALYTICS HUB", (sb_x + 15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(display_frame, (sb_x + 15, 55), (sb_x + sb_width - 15, 55), (200, 200, 200), 1)

        # System metrics
        cv2.putText(display_frame, f"System FPS: {fps:.1f}", (sb_x + 15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Total Events Today: {total_events_today}", (sb_x + 15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Record Dwell: {max_dwell_time:.1f}s", (sb_x + 15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Current status
        status_txt = "STATUS: SAFE"
        status_col = (0, 255, 0)
        if len(logged_loiterers) > 0:
            status_txt = "SEVERE LOITERING"
            status_col = (0, 0, 255)
        elif zone_breached_this_frame:
            status_txt = "AREA BREACHED"
            status_col = (0, 140, 255)
            
        cv2.putText(display_frame, status_txt, (sb_x + 15, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_col, 2)

        # Active zone entries
        cv2.putText(display_frame, "Active Live Trackers:", (sb_x + 15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset = 280
        if len(zone_entry_times) == 0:
             cv2.putText(display_frame, "-- Zone Empty --", (sb_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            for tid, etime in zone_entry_times.items():
                dwell = current_time - etime
                c_name = object_classes.get(tid, "Obj")
                
                # Highlight loitering entries
                row_col = (0, 0, 255) if dwell > LOITERING_THRESHOLD else (0, 165, 255)
                row_txt = f"{c_name.upper()} #{tid} ( {dwell:.1f}s )"
                cv2.putText(display_frame, row_txt, (sb_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, row_col, 2)
                y_offset += 30

        # Show output
        # Resize for display
        display_view = resize_for_display(display_frame, 1280, 720)
        cv2.imshow("Advanced Intrusion Analytics", display_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    if log_file_handle:
        log_file_handle.close()
    print("System shutdown complete.")

if __name__ == "__main__":
    main()