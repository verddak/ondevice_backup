# training/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

class Trainer:
    """
    PyTorch model trainer with early stopping and augmentation support
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : nn.Module
        Loss function
    device : str or torch.device
        Device to train on
    patience : int
        Early stopping patience
    augmentation : nn.Module or nn.Sequential, optional
        Augmentation pipeline to apply during training
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        device='cuda', 
        patience=15,
        augmentation: Optional[Union[nn.Module, nn.Sequential]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        
        # Augmentation
        self.augmentation = augmentation
        if self.augmentation is not None:
            self.augmentation = self.augmentation.to(device)
            self.augmentation.eval()  # Augmentation은 항상 eval 모드
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
        # Timestamp
        self.start_time = datetime.now()
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': []
        }
    
    def set_augmentation(self, augmentation: Optional[Union[nn.Module, nn.Sequential]]):
        """
        Set or update augmentation pipeline
        
        Parameters
        ----------
        augmentation : nn.Module, nn.Sequential, or None
            New augmentation pipeline
        """
        self.augmentation = augmentation
        if self.augmentation is not None:
            self.augmentation = self.augmentation.to(self.device)
            self.augmentation.eval()
    
    def train_epoch(self, train_loader):
        """Train for one epoch with optional augmentation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Apply augmentation (GPU에서 배치 단위로 처리)
            if self.augmentation is not None:
                with torch.no_grad():  # Augmentation은 gradient 계산 불필요
                    # print(f"Window warping 인덱스 수정 하셨나요?")
                    # print(f"증강기법을 HBM 대상이 아닌 On-device용으로 사용 중이신가요?")
                    batch_x = self.augmentation(batch_x)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate the model (without augmentation)"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Validation에는 augmentation 적용 안함!
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def check_early_stopping(self, val_loss, val_acc):
        """Check if should stop early"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                return True
        return False
    
    def train(self, train_loader, val_loader, epochs, verbose=True):
        """
        Train the model
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of epochs to train
        verbose : bool
            Whether to print progress
        
        Returns
        -------
        history : dict
            Training history
        """
        aug_info = "with augmentation" if self.augmentation is not None else "without augmentation"
        print(f"Starting training on {self.device} ({aug_info})")
        print(f"Total epochs: {epochs}, Early stopping patience: {self.patience}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Time
            epoch_time = time.time() - start_time
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            
            # Print
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if self.check_early_stopping(val_loss, val_acc):
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with val_loss: {self.best_val_loss:.4f}, val_acc: {self.best_val_acc:.2f}%")
        
        print("=" * 60)
        print("Training completed!")
        
        return self.history
    
    def save_checkpoint(self, checkpoint_dir, model_name, save_format='timestamp_acc'):
        """
        Save model checkpoint with custom naming
        
        Parameters
        ----------
        checkpoint_dir : Path or str
            Directory to save checkpoint
        model_name : str
            Base model name
        save_format : str
            Format for filename: 'timestamp_acc', 'acc_only', or 'simple'
        
        Returns
        -------
        save_path : Path
            Path where checkpoint was saved
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific subdirectory
        model_dir = checkpoint_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Generate filename based on format
        if save_format == 'timestamp_acc':
            # Format: model_name/20241104_143022_acc92.50.pt
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_acc{self.best_val_acc:.2f}.pt"
        elif save_format == 'acc_loss':
            # Format: model_name/acc92.50_loss0.1234.pt
            filename = f"acc{self.best_val_acc:.2f}_loss{self.best_val_loss:.4f}.pt"
        elif save_format == 'simple':
            # Format: model_name/best.pt
            filename = "best.pt"
        else:
            raise ValueError(f"Unknown save_format: {save_format}")
        
        save_path = model_dir / filename
        
        # Save checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat(),
            'training_duration': (datetime.now() - self.start_time).total_seconds()
        }, save_path)
        
        print(f"Checkpoint saved to: {save_path}")
        return save_path
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Checkpoint loaded from {path}")
        print(f"  Val Loss: {self.best_val_loss:.4f}")
        print(f"  Val Acc: {self.best_val_acc:.2f}%")