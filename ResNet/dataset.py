from torch.utils.data import Dataset
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # somehow without this the data will not open
import glob
import torch


class Hurricane(Dataset):

    # mapping between folder names and indices
    LABEL_CLASSES = {
      'no_damage': 0,
      'damage': 1
    }


    def __init__(self, root_dir, transforms=None, split='train'):
        
        """
        root_dir: path to the extracted zip folder
        split: 'train', 'validation', or 'test'
        """

        self.transforms = transforms
        split_dir = os.path.join(root_dir, split)

        # prepare data
        self.data = []  # list of tuples of (image path, label class)

        for label_name, label_idx in self.LABEL_CLASSES.items():
            folder = os.path.join(split_dir, label_name)
            
            images = glob.glob(os.path.join(folder, "*.jpeg"))

            for img_path in images:
                self.data.append((img_path, label_idx))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # Convert image to RGB, otherwise the pipeline may fail
        img = Image.open(img_path).convert("RGB")

        # apply transformation
        if self.transforms is not None:
            img = self.transforms(img)

        # return image and label
        return img, label

# Poisson Noise
class PoissonNoise(torch.nn.Module):
    def __init__(self, lam=30.0):
        """
        lam controls noise strength:
        larger = less noise, smaller = more noise
        """
        super().__init__()
        self.lam = lam

    def forward(self, x):
        # x is a tensor in [0, 1]
        # scale → sample → rescale
        noisy = torch.poisson(x * self.lam) / self.lam
        return torch.clamp(noisy, 0.0, 1.0)