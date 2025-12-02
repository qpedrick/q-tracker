import cv2
import time
import sqlite3
import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "queue_metrics.db"
IMG_FILE = "latest_debug.jpg"

# The Queue Zone (Polygon). 
# 0,0 is Top-Left. 1280,720 is Bottom-Right.
# Adjust these coordinates after looking at the Dashboard image!
QUEUE_REGION = np.array([
    [100, 100],  # Top Left
    [1180, 100], # Top Right
    [1180, 650], # Bottom Right
    [100, 650]   # Bottom Left
])

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots
                 (timestamp TEXT, count INTEGER)''')
    conn.commit()
    return conn

def main():
    conn = init_db()
    c = conn.cursor()

    print("🚀 Loading TensorRT Engine...")
    model = YOLO('yolov8n.engine', task='detect')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # --- SETUP ANNOTATORS ---
    tracker = sv.ByteTrack()
    zone = sv.PolygonZone(polygon=QUEUE_REGION)
    
    # Visualizers
    # Thickness and text_scale adjust visibility
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.RED,
        thickness=4,
        text_thickness=2,
        text_scale=2
    )
    # BoxAnnotator no longer takes text arguments in newer supervision versions
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )

    print("✅ Inference Running. View dashboard at http://<ip>:8501")
    
    last_log_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Frame read failed")
                break

            # 1. Detect
            results = model(frame, classes=0, verbose=False)[0]
            
            # 2. Track
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            # 3. Filter Zone
            mask = zone.trigger(detections=detections)
            queue_detections = detections[mask]
            queue_count = len(queue_detections)

            # 4. Log & Save Image (Every 1 second for smoother UI)
            current_time = time.time()
            if current_time - last_log_time > 1.0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Count: {queue_count}")
                
                # DB Write
                c.execute("INSERT INTO snapshots VALUES (?, ?)", (timestamp, queue_count))
                conn.commit()
                
                # Image Write (Draw boxes only on this frame)
                # Draw Zone
                frame = zone_annotator.annotate(scene=frame)
                # Draw Boxes
                frame = box_annotator.annotate(scene=frame, detections=queue_detections)
                
                # Save to disk for Streamlit to pick up
                cv2.imwrite(IMG_FILE, frame)
                
                last_log_time = current_time

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        conn.close()

if __name__ == "__main__":
    main()