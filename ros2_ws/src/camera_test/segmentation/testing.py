import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from image_only_dataset import ImageOnlyDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters & Paths
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 13
num_epochs = 10
num_workers = 2
image_height = 90   #720 oringal height
image_width = 160   #1280 orignial width
pin_memory = True
load_model = True

train_img_dir = "Data/raw image"
train_mask_dir = "Data/road_masks_output"
val_img_dir = "Data/test"
val_mask_dir = "predictions"
checkpoint_path = "road_segmentation.pth"
output_folder = "predictions"

# Training function
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

# Main training routine
def main():
    # Transforms
    train_transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, _ = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory,
    )

    # Use ImageOnlyDataset for predictions
    val_ds = ImageOnlyDataset(
        image_dir=val_img_dir,
        transform=val_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    if load_model and os.path.exists(checkpoint_path):
        load_checkpoint(torch.load(checkpoint_path, map_location=device), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Skip check_accuracy since masks are not available for test set
        # Save predictions only
        save_predictions_as_imgs(val_loader, model, folder=output_folder, device=device)

if __name__ == "__main__":
    main()
