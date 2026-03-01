import os
from omegaconf import DictConfig
import kagglehub
import torch
import pandas as pd
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from torchvision import tv_tensors

CACHE_DIR = "/work3/s252653/.cache"
file_path = "/work3/s252653/satellite-road-detection-project/data"
os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    res = tensor * std + mean
    return res.clamp(0, 1)

def download_data(kaggle_dataset: str, file_path: str):
    path = kagglehub.dataset_download(kaggle_dataset)
    os.rename(path, file_path)
    shutil.rmtree(CACHE_DIR)
    return file_path

def create_data(kaggle_dataset: str, file_path: str, transforms: dict, image_column: str, mask_column: str):
    if not os.path.exists(file_path):
        print(f"Data not found at {file_path}. Downloading...")
        data_path = download_data(kaggle_dataset, file_path)
    else:
        data_path = file_path

    train_data = Dataset(data_path, transform=transforms['train'], split="train", image_column=image_column, mask_column=mask_column)
    val_data = Dataset(data_path, transform=transforms['val'], split="val", image_column=image_column, mask_column=mask_column)
    test_data = Dataset(data_path, transform=transforms['test'], split="test", image_column=image_column, mask_column=mask_column, has_labels=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i in range(5):
        X_sample, Y_sample = train_data[i]
        
        img_display = unnormalize(X_sample).permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")
        
        mask_display = Y_sample.cpu().numpy().squeeze()
        axes[1, i].imshow(mask_display, cmap="gray")
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis("off")

    # Save results
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/image_mask_comparison.png")
    plt.show()
    plt.close()
    
    return train_data, val_data, test_data

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