import os
import matplotlib.pyplot as plt
from src.datasets.dataset_utils import unnormalize  
import torch
from torchvision import transforms as v2
import PIL.Image as Image
from torchvision import tv_tensors

def create_data(file_path: str, transforms: dict):
    """
    Create datasets for CHN6-CUG.
    """
    train_data = CHN6Dataset(file_path, transform=transforms['train'], split="train")
    val_data = CHN6Dataset(file_path, transform=transforms['val'], split="val")
    # test_data = CHN6Dataset(file_path, transform=transforms['test'], split="test")
    
    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}")
    
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
    
    return train_data, val_data

class CHN6Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, split="train"):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = os.listdir(dataset_path + '/' + split + '/images/')
        self.split = split
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.dataset_path, self.split, "images", img_name)
        mask_path = os.path.join(self.dataset_path, self.split, "gt", base_name[:-3] + "mask.png")

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        mask = tv_tensors.Mask(Image.open(mask_path).convert("L"))

        if self.transform:
            img, mask = self.transform(img, mask)

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if mask.dtype == torch.uint8:
            mask = mask.float() / 255.0

        return img, mask
        
if __name__ == "__main__":
    # Example usage
    transforms = {
        'train': None,  # Replace with actual transforms
        'val': None,
        'test': None
    }
    create_data("data/CHN6-CUG", transforms)

