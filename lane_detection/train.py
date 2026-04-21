import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATASET_PATH = r"\lane_detection\data\tusimple_preprocessed"
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
        self.img_dir = os.path.join(root_dir, "training", "frames")
        self.mask_dir = os.path.join(root_dir, "training", "lane_masks")
        self.imgs = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]

        img_path = os.path.join(self.img_dir, img_name)

        # handle different extensions safely
        name = os.path.splitext(img_name)[0]
        possible_masks = [
            name + ".png",
            name + ".jpg",
            name + "_mask.png"
        ]

        mask = None
        for m in possible_masks:
            path = os.path.join(self.mask_dir, m)
            if os.path.exists(path):
                mask = cv2.imread(path, 0)
                break

        img = cv2.imread(img_path)

        if img is None or mask is None:
            raise FileNotFoundError(f"Missing file: {img_name}")

        # resize
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        # normalize
        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
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