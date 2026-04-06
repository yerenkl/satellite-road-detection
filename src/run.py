import hydra
import torch
import csv
from datetime import datetime
from omegaconf import OmegaConf
import os
from src.utils import seed_everything
from torch.utils.data import DataLoader


def save_results_to_csv(cfg, results, csv_path="results/experiment_results.csv"):
    """Append training results to a CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    file_exists = os.path.exists(csv_path)
    
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': cfg.model.name,
        'dataset': cfg.dataset.name,
        'epochs': cfg.training.epochs,
        'batch_size': cfg.training.batch_size,
        'lr': cfg.training.optimizer.lr,
        'seed': cfg.seed,
        'best_epoch': results.get('best_epoch', ''),
        'best_iou': results.get('best_iou', ''),
        'final_train_loss': results.get('final_train_loss', ''),
        'final_train_dice': results.get('final_train_dice', ''),
        'final_train_iou': results.get('final_train_iou', ''),
        'final_val_loss': results.get('final_val_loss', ''),
        'final_val_dice': results.get('final_val_dice', ''),
        'final_val_iou': results.get('final_val_iou', ''),
    }
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Results saved to {csv_path}")


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"]='1'
    print(OmegaConf.to_yaml(cfg))

    # If resuming training, use the checkpoint's directory as result_dir
    if cfg.trainer.train.resume_from:
        resume_dir = os.path.dirname(cfg.trainer.train.resume_from)
        cfg.result_dir = resume_dir
        print(f"Resuming training - using result_dir: {resume_dir}")

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    if cfg.logger.disable == False:
        logger = hydra.utils.instantiate(cfg.logger)
        hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger.init_run(hparams)

    dataset_result = hydra.utils.instantiate(cfg.dataset.init)
    train_data, val_data = dataset_result[0], dataset_result[1]

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer,
        params=model.parameters()
    )

    # Create scheduler
    scheduler = None
    if hasattr(cfg.training, 'scheduler') and cfg.training.scheduler is not None:
        scheduler = hydra.utils.instantiate(
            cfg.training.scheduler,
            optimizer=optimizer
        )

    criterion = hydra.utils.instantiate(cfg.training.loss)
    
    trainer = hydra.utils.instantiate(cfg.trainer.init, model=model, logger=logger, device=device, 
                                        criterion=criterion,
                                        scheduler=scheduler,
                                        train_loader=train_loader, 
                                        val_loader=val_loader, 
                                        optimizer=optimizer)

    results = trainer.train(**cfg.trainer.train)
    print(f"\nFinal Results: {results}")
    
    # Save results to CSV
    save_results_to_csv(cfg, results)
    
    return results



if __name__ == "__main__":
    main()