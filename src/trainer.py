import os
import torch
from tqdm import tqdm
import wandb
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete, Compose

class Trainer:
    """Trainer class for semantic segmentation."""
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        logger=None,
        scheduler=None,
        result_dir="./results"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.result_dir = result_dir
        
        # Create results directory
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Track best model
        self.best_iou = 0.0
        self.best_epoch = 0

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.iou_metric = MeanIoU(include_background=True, reduction="mean")

        self.conf_metric = ConfusionMatrixMetric(
            metric_name=["f1 score", "precision", "sensitivity"],
            include_background=True,
            reduction="mean"
        )

        self.post_pred = Compose([
            Activations(sigmoid=True),  # convert logits to probabilities
            AsDiscrete(threshold=0.5)   # convert probabilities to 0/1
        ])
        self.post_label = AsDiscrete(threshold=0.5)  # binarize the labels
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_iou = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            preds = self.post_pred(outputs)
            labels = self.post_label(masks)

            self.dice_metric(preds, labels)
            self.iou_metric(preds, labels)
            self.conf_metric(preds, labels)
            
            total_loss += loss.item()
        
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
            
            # Log to wandb
            if self.logger and not self.logger.disable:
                wandb.log({
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(self.train_loader)
        dice = self.dice_metric.aggregate().item()
        iou = self.iou_metric.aggregate().item()
        conf = self.conf_metric.aggregate()

        self.dice_metric.reset()
        self.iou_metric.reset()
        self.conf_metric.reset()
    
        return avg_loss, dice, iou, conf
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss and metrics
                loss = self.criterion(outputs, masks)
                
                preds = self.post_pred(outputs)
                labels = self.post_label(masks)

                self.dice_metric(preds, labels)
                self.iou_metric(preds, labels)
                self.conf_metric(preds, labels)
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        dice = self.dice_metric.aggregate().item()
        iou = self.iou_metric.aggregate().item()
        conf = self.conf_metric.aggregate()

        self.dice_metric.reset()
        self.iou_metric.reset()
        self.conf_metric.reset()
        
        # Log to wandb
        if self.logger and not self.logger.disable:
            wandb.log({
                'val/loss': avg_loss,
                'val/iou': iou,
                'val/dice': dice,
                'val/f1': conf[0].item(),
                'val/precision': conf[1].item(),
                'val/sensitivity': conf[2].item(),
                'val/epoch': epoch
            })
        
        return avg_loss, dice, iou, conf
    
    def save_best_model(self, val_iou):
        """Save best model state dict."""
        path = os.path.join(self.result_dir, 'best_model.pth')
        torch.save(self.model.state_dict(), path)
        print(f"Saved best model (IoU: {val_iou:.4f})")
    
    def save_checkpoint(self, epoch, val_iou):
        """Save full training checkpoint for resuming."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch,
            'val_iou': val_iou,
        }
        path = os.path.join(self.result_dir, 'checkpoint.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_iou = checkpoint['best_iou']
        self.best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch']
        print(f"Resumed from checkpoint at epoch {start_epoch} (best IoU: {self.best_iou:.4f})")
        return start_epoch
    
    def load_model(self, checkpoint_path):
        """Load model state dict."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {checkpoint_path}")
    
    def train(self, epochs, resume_from=None):
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Continuing training from epoch {start_epoch}")
        
        print(f"\nStarting training for {epochs} epochs (from epoch {start_epoch})...")
        print(f"{'='*60}")
        
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"{'-'*60}")
            
            # Train
            train_loss, train_dice, train_iou, train_conf = self.train_epoch(epoch)
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            
            # Validate
            val_loss, val_dice, val_iou, val_conf = self.validate(epoch)
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss) 
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
                
                if self.logger and not self.logger.disable:
                    wandb.log({'train/lr': current_lr, 'train/epoch': epoch})
            
            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.best_epoch = epoch
                self.save_best_model(val_iou)
            
            # Save checkpoint for resuming
            self.save_checkpoint(epoch, val_iou)
            
            # Log epoch metrics
            if self.logger and not self.logger.disable:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'train/epoch_dice': train_dice,
                    'train/epoch_iou': train_iou,
                    'epoch': epoch
                })
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation IoU: {self.best_iou:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        return {
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch,
            'final_train_loss': train_loss,
            'final_train_dice': train_dice,
            'final_train_iou': train_iou,
            'final_val_loss': val_loss,
            'final_val_dice': val_dice,
            'final_val_iou': val_iou
        }
