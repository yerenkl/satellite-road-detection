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
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Support checkpoints saved with torch.nn.DataParallel
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def _apply_ops(tensor, ops):
    """Apply geometric ops to BCHW tensor."""
    out = tensor
    for op in ops:
        if op == "r90":
            out = torch.rot90(out, k=1, dims=(-2, -1))
        elif op == "v":
            out = torch.flip(out, dims=(-2,))
        elif op == "h":
            out = torch.flip(out, dims=(-1,))
    return out


def _invert_ops(tensor, ops):
    """Invert geometric ops on BCHW tensor."""
    out = tensor
    for op in reversed(ops):
        if op == "r90":
            out = torch.rot90(out, k=-1, dims=(-2, -1))
        elif op == "v":
            out = torch.flip(out, dims=(-2,))
        elif op == "h":
            out = torch.flip(out, dims=(-1,))
    return out


def predict_with_tta(model, img):
    """8-view TTA prediction matching rotate/flip ensembling logic."""
    tta_ops = [
        [],
        ["r90"],
        ["v"],
        ["r90", "v"],
        ["h"],
        ["r90", "h"],
        ["v", "h"],
        ["r90", "v", "h"],
    ]

    pred_sum = None
    for ops in tta_ops:
        aug_img = _apply_ops(img, ops)
        aug_pred = torch.sigmoid(model(aug_img))
        aug_pred = _invert_ops(aug_pred, ops)
        pred_sum = aug_pred if pred_sum is None else pred_sum + aug_pred

    return pred_sum / len(tta_ops)


def predict_from_dataset(model, cfg, device, output_folder, output_file="test_predictions.png", num_samples=8, threshold=0.5, use_tta=False):
    """Run predictions on test dataset and save a grid of results."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Load dataset
    _, _, test_data = hydra.utils.instantiate(cfg.dataset.init)
    
    num_samples = min(num_samples, len(test_data))
    print(f"Test dataset size: {len(test_data)}, saving {num_samples} samples")
    
    # Collect samples
    images_list = []
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            img, _ = test_data[i]
            img = img.unsqueeze(0).to(device)

            if use_tta:
                pred = predict_with_tta(model, img)
            else:
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
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict road segmentation masks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Model config override (e.g., model=unet_monai)")
    parser.add_argument("--dataset-config", type=str, default=None,
                        help="Dataset config override (e.g., dataset=deepglobe)")
    parser.add_argument("--output", type=str, default="results/predictions",
                        help="Output folder for predictions")
    parser.add_argument("--output-file", type=str, default="test_predictions.png",
                        help="Output file for predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction mask")
    parser.add_argument("--tta", type=bool, default=False,
                        help="Enable 8-view test-time augmentation")
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
    if args.dataset_config:
        overrides.append(args.dataset_config)
    
    cfg = hydra.compose(config_name="run", overrides=overrides)
    print(f"Loaded config with model: {cfg.model.name} and dataset: {cfg.dataset.name}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, cfg.model, device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run predictions - save 8 samples from test dataset
    predict_from_dataset(
        model, cfg, device, args.output,
        num_samples=8,
        threshold=args.threshold,
        output_file=args.output_file,
        use_tta=args.tta,
    )


if __name__ == "__main__":
    main()
