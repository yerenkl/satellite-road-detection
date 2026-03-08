import kagglehub
import os
import shutil
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import v2
from torchvision import tv_tensors

CACHE_DIR = "/work3/s252653/.cache"
file_path = "/work3/s252653/satellite-road-detection-project/data"
os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR

def download_data(kaggle_dataset: str, file_path: str):
    path = kagglehub.dataset_download(kaggle_dataset)
    os.rename(path, file_path)
    shutil.rmtree(CACHE_DIR)
    return file_path

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    res = tensor * std + mean
    return res.clamp(0, 1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, split="train", image_column=None, mask_column=None, has_labels=True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.has_labels = has_labels
        
        rows = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))
        rows = rows[rows["split"] == split]
        
        self.image_paths = [os.path.join(dataset_path, p) for p in rows[image_column]]
        
        if has_labels:
            rows = rows.dropna(subset=[mask_column])
            self.image_paths = [os.path.join(dataset_path, p) for p in rows[image_column]]
            self.label_paths = [os.path.join(dataset_path, p) for p in rows[mask_column]]
        else:
            self.label_paths = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = tv_tensors.Image(img)
        
        if self.has_labels:
            mask = Image.open(self.label_paths[idx]).convert("L")
            mask = tv_tensors.Mask(mask)
            if self.transform:
                img, mask = self.transform(img, mask)
            
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            
            # Ensure mask is float and in [0, 1] range
            if mask.dtype == torch.uint8:
                mask = mask.float() / 255.0
            
            return img, mask
        else:
            # No labels for test set
            if self.transform:
                img = self.transform(img)
            
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            
            return img, None
