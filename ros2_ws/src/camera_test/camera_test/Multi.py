import cv2
import numpy as np
import threading
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

# Camera and Model Setup 
camera_ids = [1, 2, 3]  
model_type = "vit_h"
checkpoint_path = "/home/ubuntu/Downloads/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)
yolo = YOLO("yolov8n.pt")

# Shared Storage for Frames and Masks
frames = [None] * len(camera_ids)
masks = [None] * len(camera_ids)
lock = threading.Lock()

# Processing Thread for Each Camera
def process_camera(idx, cam_id):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Camera {cam_id} failed to open.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)

        results = yolo(frame_rgb)[0]
        car_masks = []
        person_masks = []

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
                elif class_name == "person":
                    person_masks.append(chosen_mask)

        # Create binary masks
        car_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for mask in car_masks:
            car_mask[mask.astype(bool)] = 1
        for mask in person_masks:
            person_mask[mask.astype(bool)] = 1

        # Store data safely
        with lock:
            frames[idx] = frame
            masks[idx] = {"car": car_mask, "person": person_mask}

    cap.release()

# Start Camera Threads
threads = []
for idx, cam_id in enumerate(camera_ids):
    t = threading.Thread(target=process_camera, args=(idx, cam_id))
    t.start()
    threads.append(t)

# Display
try:
    while True:
        display = []
        with lock:
            for i in range(len(camera_ids)):
                if frames[i] is not None:
                    frame = frames[i].copy()
                    if masks[i] is not None:
                        overlay = frame.copy()
                        car_mask = masks[i]["car"]
                        person_mask = masks[i]["person"]

                        overlay[car_mask == 1] = (0, 255, 0)      
                        overlay[person_mask == 1] = (255, 0, 0)   

                        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                        display.append(vis)

        if display:
            combined = np.hstack(display)
            cv2.imshow("Multi-Camera Car & Person Segmentation", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

# Cleanup
cv2.destroyAllWindows()
for t in threads:
    t.join()
print("Multi-camera processing ended.")
