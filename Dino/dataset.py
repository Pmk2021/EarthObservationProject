import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

from transformers import Dinov2Model, AutoImageProcessor

MODEL_NAME = "rgydigital/dinov2-small"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

Preprocessing_Transforms = v2.Compose([

    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=processor.image_mean, std=processor.image_std),
])


class DinoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, n_crops=1):
        self.base = datasets.ImageFolder(root)
        self.transform = transform
        self.n_crops = n_crops

    def __len__(self):
        return len(self.base) * self.n_crops

    def __getitem__(self, idx):
        img_idx = idx // self.n_crops
        img, label = self.base[img_idx]
        return self.transform(img), label