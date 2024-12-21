from collections import defaultdict
import time
import cv2
import numpy as np
import json
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "/home/naren/MacV_task/macv-obj-tracking-video.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history and object tracking times
track_history = defaultdict(lambda: [])
object_times = defaultdict(float)
unique_ids = set()
object_classes = defaultdict(lambda: "Unknown")
last_positions = defaultdict(lambda: None)
class_confidences = defaultdict(lambda: 0.0)
visible_ids = defaultdict(int)

# Dynamic color map for IDs to prevent KeyError
np.random.seed(42)
colors = defaultdict(lambda: tuple(np.random.randint(0, 255, 3).tolist()))

# COCO class labels
coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Get video FPS for timing
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0
start_time = time.time()

# Movement threshold for static object detection
confidence_threshold = 0.45
persistence_threshold = 5  # Frames to persist IDs when temporarily lost

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1

        # Reduce image size for faster processing
        resized_frame = cv2.resize(frame, (640, 360))

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(resized_frame, persist=True)

        # Get the boxes, track IDs, class labels, and confidences
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_labels = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = resized_frame.copy()

        # Track unique IDs, time spent, and class labels
        current_ids = set()
        for track_id, class_label, confidence, box in zip(track_ids, class_labels, confidences, boxes):
            x, y, w, h = box
            # Apply confidence threshold
            if confidence < confidence_threshold:
                continue

            unique_ids.add(track_id)
            visible_ids[track_id] = persistence_threshold

            # Update class based on higher confidence
            if confidence > class_confidences[track_id]:
                object_classes[track_id] = coco_labels[class_label]
                class_confidences[track_id] = confidence
            current_ids.add(track_id)

            # Update time tracking only if object is visible
            object_times[track_id] += 1 / fps

            # Update tracking trails
            track = track_history[track_id]
            track.append((int(x), int(y)))
            if len(track) > 30:
                track.pop(0)

            # Draw diamond pointer with glow effect
            pointer = np.array([
                (int(x), int(y) - 10),  # Top
                (int(x) - 10, int(y)), # Left
                (int(x), int(y) + 10), # Bottom
                (int(x) + 10, int(y))  # Right
            ])
            cv2.polylines(annotated_frame, [pointer], isClosed=True, color=colors[track_id], thickness=2)
            cv2.fillPoly(annotated_frame, [pointer], colors[track_id])

            # Draw tracking trails with fading effect
            for i in range(1, len(track)):
                alpha = int(255 * (i / len(track)))
                cv2.line(annotated_frame, track[i - 1], track[i], colors[track_id], 2)

            # Draw bounding boxes with glow effect
            color = colors[track_id]
            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 3)

            # Label inside shadowed box
            text = f"ID: {track_id}, {object_classes[track_id]}, {object_times[track_id]:.2f}s"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2) - 25), (int(x - w / 2) + tw, int(y - h / 2)), color, -1)
            cv2.putText(annotated_frame, text, (int(x - w / 2), int(y - h / 2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Overlay stats bar with glass-like gradient
        overlay_height = 50
        stats_overlay = np.zeros((overlay_height, resized_frame.shape[1], 3), dtype=np.uint8)
        fps_display = frame_count / (time.time() - start_time)
        cv2.rectangle(stats_overlay, (0, 0), (resized_frame.shape[1], overlay_height), (40, 40, 40), -1)
        cv2.putText(stats_overlay, f"FPS: {fps_display:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(stats_overlay, f"IDs: {len(unique_ids)}", (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Combine video and stats overlay
        combined_frame = np.vstack((annotated_frame, stats_overlay))
        cv2.imshow("YOLO11 Tracking", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

output_data = {track_id: {"class": object_classes[track_id], "time": object_times[track_id]} for track_id in visible_ids}
with open("object_tracking_data.json", "w") as f:
    json.dump(output_data, f, indent=4)

cap.release()
cv2.destroyAllWindows()

