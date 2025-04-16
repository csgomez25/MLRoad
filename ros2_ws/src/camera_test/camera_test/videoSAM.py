import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import torch

# Paths
video_path = "/home/ubuntu/Videos/Test_1.mp4"
checkpoint_path = "/home/ubuntu/Downloads/sam_vit_h_4b8939.pth"
output_dir = "car_masks_video_output"
os.makedirs(output_dir, exist_ok=True)

# Load SAM model
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video: {video_path}")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Rotate adjustment: dimensions switch
output_size = (height, width)
video_writer = cv2.VideoWriter(
    os.path.join(output_dir, "test_output.mp4"),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    output_size
)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    results = yolo(frame_rgb)[0]
    car_masks = []

    for i, det in enumerate(results.boxes):
        cls_id = int(det.cls)
        class_name = yolo.names[cls_id]

        if class_name.lower() != "car":
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=True)
        chosen_mask = max(masks, key=lambda m: m.sum())
        car_masks.append(chosen_mask)

    if car_masks:
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for mask in car_masks:
            combined_mask[mask.astype(bool)] = 1

        overlay = frame.copy()
        overlay[combined_mask == 1] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        video_writer.write(vis)

        # Live display
        cv2.imshow("Car Segmentation", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"Frame {frame_idx:05d} processed")
    else:
        print(f"No cars detected in frame {frame_idx:05d}")

    frame_idx += 1


cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing complete.")
