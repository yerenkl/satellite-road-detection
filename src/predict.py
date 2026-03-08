"""
Prediction script for satellite road detection.
Loads a pretrained model and generates predictions on test data.
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import hydra


def unnormalize(tensor):
    """Unnormalize tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    res = tensor * std + mean
    return res.clamp(0, 1)


def load_model(checkpoint_path, model_config, device):
    """Load model from checkpoint."""
    # Instantiate model from config
    model = hydra.utils.instantiate(model_config.init)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_from_dataset(model, cfg, device, output_folder, threshold=0.5, num_samples=8):
    """Run predictions on test dataset and save a grid of results."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Load dataset
    _, _, test_data = hydra.utils.instantiate(cfg.dataset.init)
    
    print(f"Test dataset size: {len(test_data)}, saving {num_samples} samples")
    
    # Collect samples
    images_list = []
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            img, _ = test_data[i]
            img = img.unsqueeze(0).to(device)
            
            output = model(img)
            pred = torch.sigmoid(output)
            pred_binary = (pred > threshold).float()
            
            images_list.append((unnormalize(img.squeeze()), pred_binary.squeeze()))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i, (img_tensor, pred_mask) in enumerate(images_list):
        # Input image
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")
        
        # Prediction mask
        pred_np = pred_mask.cpu().numpy().squeeze()
        axes[1, i].imshow(pred_np, cmap="gray")
        axes[1, i].set_title(f"Prediction {i+1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, "test_predictions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict road segmentation masks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Model config override (e.g., model=unet_monai)")
    parser.add_argument("--output", type=str, default="results/predictions",
                        help="Output folder for predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    hydra.initialize(config_path="../configs", version_base=None)
    
    overrides = []
    if args.model_config:
        overrides.append(args.model_config)
    
    cfg = hydra.compose(config_name="run", overrides=overrides)
    print(f"Loaded config with model: {cfg.model.name}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, cfg.model, device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run predictions - save 8 samples from test dataset
    predict_from_dataset(
        model, cfg, device, args.output,
        threshold=args.threshold,
        num_samples=8
    )


if __name__ == "__main__":
    main()
