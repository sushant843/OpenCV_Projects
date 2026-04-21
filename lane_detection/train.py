import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATASET_PATH = "lane_detection/data/tusimple_preprocessed"

# 1. Gets the exact folder where train.py lives (lane_detection)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 2. Joins that securely to the rest of your path
# DATASET_PATH = os.path.join(BASE_DIR, "data", "tusimple_preprocessed","training")
IMG_HEIGHT = 256
IMG_WIDTH = 512
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class LaneDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "frames")
        self.mask_dir = os.path.join(root_dir, "lane-masks")

        # --- DEBUGGING PRINTS ---
        print(f"Checking Path: {self.img_dir}")
        if not os.path.exists(self.img_dir):
            print(f"ERROR: The folder '{self.img_dir}' does not exist!")
            self.valid_imgs = []
            return

        all_files = os.listdir(self.img_dir)
        print(f"Found {len(all_files)} total files in frames folder.")
        
        self.valid_imgs = []
        for f in all_files:
            if f.lower().endswith(".jpg"):
                # Check for the mask. 
                # IMPORTANT: Ensure your masks are actually .png
                mask_name = f.replace(".jpg", ".png")
                mask_path = os.path.join(self.mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.valid_imgs.append(f)
        
        if len(self.valid_imgs) == 0:
            print("ERROR: Found 0 pairs. Check if masks have the same name as images!")
            # Print one sample to see what's happening
            if len(all_files) > 0:
                print(f"Sample frame name: {all_files[0]}")
                print(f"Expected mask path: {os.path.join(self.mask_dir, all_files[0].replace('.jpg', '.png'))}")
        else:
            print(f"Successfully matched {len(self.valid_imgs)} image-mask pairs.")

    def __len__(self):
        return len(self.valid_imgs)

    def __getitem__(self, idx):
        img_name = self.valid_imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize
        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

# =========================
# U-NET MODEL
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.middle(self.pool3(d3))

        u3 = self.up3(m)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(u1)

# =========================
# LOAD DATA
# =========================
dataset = LaneDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL SETUP
# =========================
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "lane_model.pth")
print("Model saved!")

# =========================
# TEST / VISUALIZATION
# =========================
model.eval()

img, mask = dataset[10]

with torch.no_grad():
    pred = model(img.unsqueeze(0).to(DEVICE))
    pred = torch.sigmoid(pred).cpu().squeeze().numpy()

img_np = img.permute(1,2,0).numpy()

cv2.imshow("Image", img_np)
cv2.imshow("Ground Truth", mask.squeeze().numpy())
cv2.imshow("Prediction", pred)

cv2.waitKey(0)
cv2.destroyAllWindows()