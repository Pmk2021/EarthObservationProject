import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score
import csv
from pathlib import Path

from dataset import Hurricane, PoissonNoise



# -----Configuration-----
DATA_ROOT = "ipeo_hurricane_for_students"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# -----Transforms-----
# ImageNet mean and std
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    PoissonNoise(lam=30.0),
    transforms.Normalize(mean, std),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# -----Dataset / DataLoader-----
def build_dataloaders():
    train_dataset = Hurricane(DATA_ROOT, split="train", transforms=train_transforms)
    val_dataset   = Hurricane(DATA_ROOT, split="validation", transforms=val_transforms)
    test_dataset  = Hurricane(DATA_ROOT, split="test", transforms=val_transforms)

    train_labels = [label for _, label in train_dataset.data]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


# -----Model-----
def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


# -----Training / Evaluation-----
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    preds, targets = [], []
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds.append(outputs.argmax(1).cpu())
        targets.append(labels.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return (
        running_loss / len(loader.dataset),
        accuracy_score(targets, preds),
        f1_score(targets, preds),
    )


def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds.append(outputs.argmax(1).cpu())
            targets.append(labels.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return (
        running_loss / len(loader.dataset),
        accuracy_score(targets, preds),
        f1_score(targets, preds, average='weighted'),
    )


# -----Main-----
def main():
    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0

    # Store metrics for plots
    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [],   "val_acc": [],   "val_f1": [],
    }

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        # save metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_resnet18.pt")
            print("Best model saved", flush=True)

    # Save history to CSV for graphing
    with open("training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i], history["train_acc"][i], history["train_f1"][i],
                history["val_loss"][i],   history["val_acc"][i],   history["val_f1"][i],
            ])

    print("Saved metrics to training_history.csv", flush=True)

    # Test with best model
    model.load_state_dict(torch.load("best_resnet18.pt", map_location=DEVICE))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)

    print(f"TEST → loss={test_loss:.4f}, acc={test_acc:.3f}, f1={test_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()