import sys
import cv2
import numpy as np
import time
import os
from collections import defaultdict
from ultralytics import YOLO

# Command-line arguments
input_path = sys.argv[1]  # Input video path
output_path = sys.argv[2] # Final output path
temp_frames_dir = "temp_frames"  # Temporary directory for saving frames
os.makedirs(temp_frames_dir, exist_ok=True)

# Load the YOLO model
model = YOLO("yolo11n.pt").to('cpu')  # Force CPU usage

# Open the input video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tracking variables
track_history = defaultdict(lambda: [])
object_times = defaultdict(float)
unique_ids = set()
colors = defaultdict(lambda: tuple(np.random.randint(0, 255, 3).tolist()))
confidence_threshold = 0.45

# COCO class labels
coco_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"]

# FPS tracking
frame_count = 0
start_time = time.time()

# Process video frame by frame
frame_index = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 360))

    # Run YOLO model inference with tracking
    results = model.track(resized_frame, persist=True)

    # Get detection results
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_labels = results[0].boxes.cls.int().cpu().tolist()
    confidences = results[0].boxes.conf.cpu().tolist()

    # Annotated frame for output
    annotated_frame = resized_frame.copy()

    # Process detections
    for track_id, class_label, confidence, box in zip(track_ids, class_labels, confidences, boxes):
        x, y, w, h = box

        # Skip low-confidence detections
        if confidence < confidence_threshold:
            continue

        # Track unique IDs
        unique_ids.add(track_id)
        object_times[track_id] += 1 / fps  # Track time visible

        # Assign color
        color = colors[track_id]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                      (int(x + w / 2), int(y + h / 2)), color, 3)

        # Draw label inside shadowed box
        if class_label < len(coco_labels):
            label = f"ID: {track_id}, {coco_labels[class_label]}, {confidence:.2f}"
        else:
            label = f"ID: {track_id}, Unknown, {confidence:.2f}"  # Fallback for unknown classes

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2) - 25),
                      (int(x - w / 2) + tw, int(y - h / 2)), color, -1)
        cv2.putText(annotated_frame, label, (int(x - w / 2), int(y - h / 2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw diamond pointer with glow effect
        pointer = np.array([
            (int(x), int(y) - 10),
            (int(x) - 10, int(y)),
            (int(x), int(y) + 10),
            (int(x) + 10, int(y))
        ])
        cv2.polylines(annotated_frame, [pointer], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(annotated_frame, [pointer], color)

        # Draw tracking trails with fading effect
        track = track_history[track_id]
        track.append((int(x), int(y)))
        if len(track) > 30:
            track.pop(0)
        for i in range(1, len(track)):
            cv2.line(annotated_frame, track[i - 1], track[i], color, 2)

    # Overlay FPS and object count
    overlay_height = 50
    stats_overlay = np.zeros((overlay_height, resized_frame.shape[1], 3), dtype=np.uint8)
    fps_display = frame_count / (time.time() - start_time)
    cv2.rectangle(stats_overlay, (0, 0), (resized_frame.shape[1], overlay_height), (40, 40, 40), -1)
    cv2.putText(stats_overlay, f"FPS: {fps_display:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(stats_overlay, f"IDs: {len(unique_ids)}", (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Combine video and overlay
    combined_frame = np.vstack((annotated_frame, stats_overlay))

    # Save frame as image for FFmpeg
    cv2.imwrite(f"{temp_frames_dir}/frame_{frame_index:04d}.jpg", combined_frame)
    frame_index += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

# Use FFmpeg to combine images into MP4
ffmpeg_command = f"ffmpeg -framerate {fps} -i {temp_frames_dir}/frame_%04d.jpg -vcodec libx264 -pix_fmt yuv420p {output_path}"
os.system(ffmpeg_command)

# Clean up temporary frames
for file in os.listdir(temp_frames_dir):
    os.remove(os.path.join(temp_frames_dir, file))
os.rmdir(temp_frames_dir)

print(f"Processing complete. Output saved to: {output_path}")

