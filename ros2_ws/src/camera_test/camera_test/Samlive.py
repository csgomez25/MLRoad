import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import torch


checkpoint_path = "/home/ubuntu/Downloads/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error accessing the camera")
    exit()

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    results = yolo(frame_rgb)[0]
    car_masks = []

    for det in results.boxes:
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

        cv2.imshow("Live Car Segmentation", vis)
        print(f"Frame {frame_idx:05d} - cars segmented")
    else:
        cv2.imshow("Live Car Segmentation", frame)
        print(f"Frame {frame_idx:05d} - no cars")

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Live processing ended.")
