import os
import matplotlib.pyplot as plt
from src.datasets.dataset_utils import download_data, unnormalize, Dataset  

CACHE_DIR = "/work3/s252653/.cache"
os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR


def create_data(file_path: str, transforms: dict, image_column: str, mask_column: str):
    """
    Create datasets for Massachusetts Roads.
    This dataset has labels for train, val, and test splits.
    """
    if not os.path.exists(file_path):
        print(f"Data not found at {file_path}. Downloading...")
        data_path = download_data("balraj98/massachusetts-roads-dataset", file_path)
    else:
        data_path = file_path

    train_data = Dataset(data_path, transform=transforms['train'], split="train", image_column=image_column, mask_column=mask_column)
    val_data = Dataset(data_path, transform=transforms['val'], split="val", image_column=image_column, mask_column=mask_column)
    test_data = Dataset(data_path, transform=transforms['test'], split="test", image_column=image_column, mask_column=mask_column)
    
    print(f"Massachusetts dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
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
        'train': None,  # Replace with actual transforms
        'val': None,
        'test': None
    }
    create_data("data/massasschussetts-roads", transforms, image_column="tiff_image_path", mask_column="tif_label_path")

