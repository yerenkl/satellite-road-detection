import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Subset
from torch import Generator
from src.datasets.dataset_utils import download_data, unnormalize, Dataset  

def create_data(file_path: str, transforms: dict, image_column: str, mask_column: str, train_split: float):
    """
    Create datasets for DeepGlobe.
    Since DeepGlobe doesn't have labels in val/test, we split the train set into train/val.
    """
    if not os.path.exists(file_path):
        print(f"Data not found at {file_path}. Downloading...")
        data_path = download_data("balraj98/deepglobe-road-extraction-dataset", file_path)
    else:
        data_path = file_path

    # Load all training data
    full_data = Dataset(data_path, None, split="train", image_column=image_column, mask_column=mask_column)
    
    # Split train into train and val
    train_size = int(train_split * len(full_data))
    val_size = len(full_data) - train_size

    # Fixed seed
    generator = Generator().manual_seed(42)

    
    train_indices, val_indices = random_split(
        range(len(full_data)),
        [train_size, val_size],
        generator=generator
    )

    # Provided validation set doesn't have labels so I use the training set for both train and val
    train_data = Subset(
        Dataset(data_path, transforms["train"], "train", image_column, mask_column),
        train_indices.indices
    )

    val_data = Subset(
        Dataset(data_path, transforms["test"], "train", image_column, mask_column),
        val_indices.indices
    )

    test_data = Dataset(data_path, transform=transforms['test'], split="test", image_column=image_column, mask_column=mask_column, has_labels=False)
    
    print(f"DeepGlobe dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
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

if __name__ == "__main__":
    # Example usage
    transforms = {
        'train': None,
        'val': None,
        'test': None
    }
    create_data("data/deepglobe", transforms, image_column="sat_image_path", mask_column="mask_path", train_split=0.8)