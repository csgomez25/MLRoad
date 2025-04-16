import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import torch

# Paths
image_dir = "/home/ubuntu/Documents/Data/archive/bdd100k/bdd100k/images/10k/test/Test1"
checkpoint_path = "/home/ubuntu/Downloads/sam_vit_h_4b8939.pth"
output_dir = "car_masks_output"
os.makedirs(output_dir, exist_ok=True)

# Load SAM
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# Load YOLOv8 model (pre-trained
yolo = YOLO("yolov8n.pt")

# Load image files
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for idx, file_name in enumerate(image_files):
    image_path = os.path.join(image_dir, file_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    results = yolo(image_rgb)[0] 
    car_masks = []

    for i, det in enumerate(results.boxes):
        cls_id = int(det.cls)
        class_name = yolo.names[cls_id]

        if class_name.lower() != "car":
            continue

        # Bounding box for SAM
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=True
        )

        # Choose the largest mask
        chosen_mask = max(masks, key=lambda m: m.sum())
        car_masks.append(chosen_mask)

        # Save individual mask
        individual_mask_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_car{i+1}_mask.png")
        cv2.imwrite(individual_mask_path, chosen_mask.astype(np.uint8) * 255)

    if car_masks:
        # Combine masks into one semantic-style mask
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for idx_mask, mask in enumerate(car_masks, start=1):
            # Assign a unique label to each car instance
            combined_mask[mask.astype(bool)] = 1  # "1" = car class label

        # Save final semantic mask
        mask_filename = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_semantic_mask.png")
        cv2.imwrite(mask_filename, combined_mask * 255) 
        print(f" Saved semantic mask: {mask_filename}")

        # Visualization
        overlay = image.copy()
        overlay[combined_mask == 1] = (0, 255, 0)  # Green overlay for car mask
        vis = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        vis_filename = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_vis.png")
        cv2.imwrite(vis_filename, vis)
        print(f" Saved visualization: {vis_filename}")
    else:
        print(f" No cars detected in {file_name}")
