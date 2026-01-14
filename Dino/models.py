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

from Dino.dataset import Preprocessing_Transforms, DinoDataset

from torch.utils.data import DataLoader
from tqdm import tqdm

### Initialiate DINO backbone
MODEL_NAME = "rgydigital/dinov2-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

backbone = Dinov2Model.from_pretrained(MODEL_NAME).to(DEVICE)
backbone.eval()

hidden_dim = backbone.config.hidden_size


def extract_dino_features(dataset, cache_path, device = "cpu", batch_size=1):
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    feats, labels = [], []

    with torch.no_grad():
        for imgs, y in tqdm(loader, desc=f"Extracting {cache_path.name}"):
            imgs = imgs.to(DEVICE)
            out = backbone(imgs).last_hidden_state
            feat = out[:, 1:].mean(dim=1)

            feats.append(feat.cpu())
            labels.append(y)

    feats = torch.cat(feats)
    labels = torch.cat(labels)
    torch.save((feats, labels), cache_path)
    return feats, labels

### Classifier
classifier = nn.Sequential(
    nn.Linear(hidden_dim, 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)    
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    classifier.parameters(), lr=0.001, weight_decay=1e-5
)