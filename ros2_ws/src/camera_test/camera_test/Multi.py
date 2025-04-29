import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

# Setup
camera_ids = [0, 1]  # Add more camera indices as needed
model_type = "vit_h"
checkpoint_path = "/home/ubuntu/Downloads/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)
yolo = YOLO("yolov8n.pt")

# Initialize video captures
caps = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Camera {camera_ids[idx]} failed to open.")

# Main loop
try:
    while True:
        frames_to_show = []

        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frames_to_show.append(np.zeros((480, 640, 3), dtype=np.uint8))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(frame_rgb)

            results = yolo(frame_rgb)[0]
            car_masks, person_masks = [], []

            for det in results.boxes:
                cls_id = int(det.cls)
                class_name = yolo.names[cls_id].lower()

                x1, y1, x2, y2 = map(int, det.xyxy[0])
                input_box = np.array([x1, y1, x2, y2])

                if class_name in ["car", "person"]:
                    masks_pred, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=True)
                    chosen_mask = max(masks_pred, key=lambda m: m.sum())
                    if class_name == "car":
                        car_masks.append(chosen_mask)
                    else:
                        person_masks.append(chosen_mask)

            car_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for mask in car_masks:
                car_mask[mask.astype(bool)] = 1
            for mask in person_masks:
                person_mask[mask.astype(bool)] = 1

            overlay = frame.copy()
            overlay[car_mask == 1] = (0, 255, 0)
            overlay[person_mask == 1] = (255, 0, 0)

            vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            frames_to_show.append(vis)

        # Resize and stack frames side by side
        resized = [cv2.resize(f, (640, 480)) for f in frames_to_show]
        combined = np.hstack(resized) if len(resized) > 1 else resized[0]
        cv2.imshow("Multi-Camera View - Car & Person Segmentation", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("Multi-camera processing ended.")
