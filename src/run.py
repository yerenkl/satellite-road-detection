import hydra
import torch
from omegaconf import OmegaConf
import os
from src.utils import seed_everything
from src.trainer import combined_loss
from torch.utils.data import DataLoader

@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"]='1'
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    if cfg.logger.disable == False:
        logger = hydra.utils.instantiate(cfg.logger)
        hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger.init_run(hparams)

    train_data, val_data, _ = hydra.utils.instantiate(cfg.dataset.init)

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
    
    return results



if __name__ == "__main__":
    main()