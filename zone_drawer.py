import cv2
import json
import numpy as np
import tkinter as tk

# Global variables
points = []
frame_copy = None
display_frame = None
display_scale = 1.0

# Get screen resolution
def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

# Resize frame to fit full screen
def get_display_frame(frame):
    screen_w, screen_h = get_screen_size()

    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)

    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return resized, scale

def to_display(pt):
    return (int(pt[0] * display_scale), int(pt[1] * display_scale))

def to_original(pt):
    return (int(pt[0] / display_scale), int(pt[1] / display_scale))

def click_event(event, x, y, flags, param):
    global points, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        orig_pt = to_original((x, y))
        points.append(orig_pt)

        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)

        if len(points) >= 2:
            prev_disp = to_display(points[-2])
            cv2.line(frame_copy, prev_disp, (x, y), (0, 255, 0), 2)

        cv2.imshow("Zone Drawer", frame_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            frame_copy = display_frame.copy()

            for i in range(len(points)):
                disp_pt = to_display(points[i])
                cv2.circle(frame_copy, disp_pt, 5, (0, 255, 0), -1)

                if i > 0:
                    prev_disp = to_display(points[i - 1])
                    cv2.line(frame_copy, prev_disp, disp_pt, (0, 255, 0), 2)

            cv2.imshow("Zone Drawer", frame_copy)

def main():
    global frame_copy, display_frame, display_scale

    video_path = "test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from video")
        return

    display_frame, display_scale = get_display_frame(frame)
    frame_copy = display_frame.copy()

    cv2.namedWindow("Zone Drawer", cv2.WINDOW_NORMAL)

    # Fullscreen mode
    cv2.setWindowProperty("Zone Drawer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Zone Drawer", frame_copy)
    cv2.setMouseCallback("Zone Drawer", click_event)

    print("\n--- Zone Drawer Instructions ---")
    print("1. Left-click to add points to your zone polygon.")
    print("2. Right-click to remove the last point.")
    print("3. Press 's' to save the zone and close.")
    print("4. Press 'q' or 'ESC' to quit without saving.")
    print("--------------------------------\n")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(points) >= 3:
                cv2.line(frame_copy, to_display(points[-1]), to_display(points[0]), (0, 255, 0), 2)
                cv2.imshow("Zone Drawer", frame_copy)
                cv2.waitKey(500)

                with open('zone.json', 'w') as f:
                    json.dump(points, f, indent=4)

                print(f"Zone saved successfully to zone.json! Coordinates: {points}")
                break
            else:
                print("A polygon needs at least 3 points! Add more points.")

        elif key == ord('q') or key == 27:
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
