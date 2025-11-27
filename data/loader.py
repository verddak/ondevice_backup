# data/loader.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from .dataset import TimeSeriesDataset


def load_hbm_data(data_dir):
    """
    Load HBM data from .npy files
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing .npy files
        
    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
    """
    data_dir = Path(data_dir)
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_valid.npy')
    y_val = np.load(data_dir / 'y_valid.npy')
    
    print(f"Loaded data from {data_dir}")
    print(f"  X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  X_val: {X_val.shape}, dtype: {X_val.dtype}")
    print(f"  y_val: {y_val.shape}, dtype: {y_val.dtype}")
    
    return X_train, y_train, X_val, y_val

def create_dataloaders(X_train, y_train, X_val, y_val, 
                      batch_size=64, 
                      train_transforms=None,
                      seed=42,
                      num_workers=4):
    """
    Create train and validation dataloaders with reproducibility
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data and labels
    X_val, y_val : array-like
        Validation data and labels
    batch_size : int
        Batch size for training
    train_transforms : list, optional
        List of augmentation transforms (only applied to train)
    seed : int
        Random seed for reproducibility
    num_workers : int
        Number of data loading workers
        
    Returns
    -------
    train_loader, val_loader : DataLoader
        PyTorch data loaders
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, transforms=train_transforms)
    val_dataset = TimeSeriesDataset(X_val, y_val, transforms=None)  # No augmentation!
    
    # Seed worker function for reproducibility
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Generator for shuffle reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g  # 이게 핵심!
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,  # Validation은 shuffle 안 함
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return train_loader, val_loader