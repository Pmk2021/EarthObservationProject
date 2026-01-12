import os

from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

from transformers import Dinov2Model, AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score

from dataset import Preprocessing_Transforms, DinoDataset
from EarthObservationProject.EarthObsoFinal.EarthObservationProject.Dino.models import extract_dino_features, backbone

### Configuration
DATA_DIR = Path("ipeo_hurricane_for_students/train")
VAL_DIR = Path("ipeo_hurricane_for_students/validation")

BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = Path("cached_features")
CACHE_DIR.mkdir(exist_ok=True)


### Initialize Datasets
train_img_ds = DinoDataset(DATA_DIR, Preprocessing_Transforms)
val_img_ds = datasets.ImageFolder(VAL_DIR, transform=Preprocessing_Transforms)

num_classes = len(train_img_ds.base.classes)
hidden_dim = backbone.config.hidden_size

X_train, y_train = extract_dino_features(
    train_img_ds, CACHE_DIR / "train.pt"
)

X_val, y_val = extract_dino_features(
    val_img_ds, CACHE_DIR / "val.pt"
)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)



### Training Loop
best_val = 0.0

for epoch in range(NUM_EPOCHS):
    classifier.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(classifier(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"\nEpoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")


    classifier.eval()
    preds, labels = [], []


    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(DEVICE)
            p = classifier(x).argmax(1).cpu()
            preds.extend(p.numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    print(f"Train Accuracy: {acc:.4f}")
    print(f"Train F1: {f1:.4f}")


    classifier.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            p = classifier(x).argmax(1).cpu()
            preds.extend(p.numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1: {f1:.4f}")

    if acc > best_val:
        best_val = acc
        torch.save(classifier.state_dict(), "dino_classifier.pth")
        print("✅ Model saved")

print("Best validation accuracy:", best_val)
