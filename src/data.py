import os
from omegaconf import DictConfig
import kagglehub
import torch
import pandas as pd
from PIL import Image
import shutil

CACHE_DIR = "/work3/s252653/.cache"
file_path = "/work3/s252653/satellite-road-detection-project/data"
os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR

def download_data(kaggle_dataset: str, file_path: str):
    path = kagglehub.dataset_download(kaggle_dataset)
    os.rename(path, file_path)
    shutil.rmtree(CACHE_DIR)
    return file_path

def create_data(kaggle_dataset: str, file_path: str, img_transform, mask_transform, split: str = "train"):
    if not os.path.exists(file_path):
        print(f"Data not found at {file_path}. Downloading from Kaggle dataset '{kaggle_dataset}'...")
        data_path = download_data(kaggle_dataset, file_path)
    else:
        data_path = file_path
    dataset = Dataset(data_path, img_transform=img_transform, mask_transform=mask_transform, split=split)
    # print first item
    X, Y = dataset[0]
    print(f"First image shape: {X.shape}, First mask shape: {Y.shape}")
    return data_path, dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, img_transform, mask_transform, split="train"):
        'Initialization'
        self.dataset_path = dataset_path
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.label_paths = []
        print(len(self.image_paths))
        print(len(self.label_paths))
        rows = pd.read_csv(dataset_path + "/metadata.csv")

        for i in range(len(rows)):
            if rows.iloc[i]["split"] == split:
                self.image_paths.append(dataset_path + "/" + rows.iloc[i]["tiff_image_path"])
                self.label_paths.append(dataset_path + "/" + rows.iloc[i]["tif_label_path"])

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.mask_transform(label)
        X = self.img_transform(image)
        return X, Y