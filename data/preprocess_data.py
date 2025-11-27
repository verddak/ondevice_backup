"""
Time Series Data Preprocessing Script
Supports both train/valid split and K-fold cross-validation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
from datetime import datetime
import argparse


class TimeSeriesPreprocessor:
    def __init__(self, raw_data_dir, output_dir, seed=42):
        """
        Args:
            raw_data_dir: Path to raw data directory (e.g., 'ondevice_raw/2mm_#1')
            output_dir: Path to output directory for processed data
            seed: Random seed for reproducibility
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.class_names = ['no_defect', 'defect_0.2mm', 'defect_0.5mm', 'defect_1.0mm']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        np.random.seed(seed)
        
    def load_all_data(self):
        """Load all CSV files and create dataset"""
        X_list = []
        y_list = []
        file_info = []  # Store (class_name, file_name) for metadata
        
        for class_name in self.class_names:
            class_dir = self.raw_data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist. Skipping...")
                continue
            
            csv_files = sorted([f for f in class_dir.glob('*.csv')])
            print(f"Loading {len(csv_files)} files from {class_name}...")
            
            for csv_file in csv_files:
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file)
                    
                    # Extract voltage values (case-insensitive column name)
                    voltage_col = None
                    for col in df.columns:
                        if col.lower() == 'voltage':
                            voltage_col = col
                            break
                    
                    if voltage_col is None:
                        print(f"Warning: No 'Voltage' column found in {csv_file}. Columns: {df.columns.tolist()}")
                        continue
                    
                    voltage = df[voltage_col].values
                    
                    X_list.append(voltage)
                    y_list.append(self.class_to_idx[class_name])
                    file_info.append((class_name, csv_file.name))
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
                    continue
        
        X = np.array(X_list)  # Shape: (n_samples, n_timesteps)
        y = np.array(y_list)  # Shape: (n_samples,)
        
        # Add channel dimension for univariate time series
        # Shape: (n_samples, n_timesteps) -> (n_samples, 1, n_timesteps)
        X = X[:, np.newaxis, :]  # or X.reshape(X.shape[0], 1, X.shape[1])
        
        print(f"\nDataset loaded:")
        print(f"  X shape: {X.shape}  # (n_samples, n_channels, seq_len)")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y, return_counts=True)}")
        
        return X, y, file_info
    
    def create_train_valid_split(self, X, y, file_info, valid_size=0.2):
        """Create train/valid split with stratification"""
        indices = np.arange(len(X))
        
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=valid_size,
            random_state=self.seed,
            stratify=y
        )
        
        X_train = X[train_idx]
        X_valid = X[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        
        # Create metadata
        metadata = {
            'seed': self.seed,
            'valid_size': valid_size,
            'n_train': len(train_idx),
            'n_valid': len(valid_idx),
            'train_class_distribution': {
                int(cls): int(count) for cls, count in zip(*np.unique(y_train, return_counts=True))
            },
            'valid_class_distribution': {
                int(cls): int(count) for cls, count in zip(*np.unique(y_valid, return_counts=True))
            },
            'train_files': [file_info[i] for i in train_idx],
            'valid_files': [file_info[i] for i in valid_idx],
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        return X_train, X_valid, y_train, y_valid, metadata
    
    def create_kfold_splits(self, X, y, file_info, n_splits=5):
        """Create K-fold cross-validation splits"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        folds_metadata = {
            'seed': self.seed,
            'n_splits': n_splits,
            'n_samples': len(X),
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat(),
            'folds': []
        }
        
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            fold_info = {
                'fold': fold_idx,
                'n_train': len(train_idx),
                'n_valid': len(valid_idx),
                'train_class_distribution': {
                    int(cls): int(count) for cls, count in zip(*np.unique(y[train_idx], return_counts=True))
                },
                'valid_class_distribution': {
                    int(cls): int(count) for cls, count in zip(*np.unique(y[valid_idx], return_counts=True))
                },
                'train_files': [file_info[i] for i in train_idx],
                'valid_files': [file_info[i] for i in valid_idx],
            }
            folds_metadata['folds'].append(fold_info)
            
            # Save fold data
            fold_dir = self.output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)
            
            np.save(fold_dir / 'X_train.npy', X[train_idx])
            np.save(fold_dir / 'X_valid.npy', X[valid_idx])
            np.save(fold_dir / 'y_train.npy', y[train_idx])
            np.save(fold_dir / 'y_valid.npy', y[valid_idx])
            
            print(f"Fold {fold_idx}: train={len(train_idx)}, valid={len(valid_idx)}")
        
        return folds_metadata
    
    def save_data(self, X_train, X_valid, y_train, y_valid, metadata):
        """Save processed data and metadata"""
        # Save numpy arrays
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_valid.npy', X_valid)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_valid.npy', y_valid)
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nData saved to {self.output_dir}")
        print(f"  X_train: {X_train.shape}  # (n_samples, n_channels, seq_len)")
        print(f"  X_valid: {X_valid.shape}  # (n_samples, n_channels, seq_len)")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_valid: {y_valid.shape}")
    
    def run_train_valid_split(self, valid_size=0.2):
        """Run complete preprocessing pipeline with train/valid split"""
        print(f"Starting preprocessing with seed={self.seed}")
        print(f"Raw data directory: {self.raw_data_dir}")
        print(f"Output directory: {self.output_dir}\n")
        
        # Load data
        X, y, file_info = self.load_all_data()
        
        # Create split
        X_train, X_valid, y_train, y_valid, metadata = self.create_train_valid_split(
            X, y, file_info, valid_size=valid_size
        )
        
        # Save data
        self.save_data(X_train, X_valid, y_train, y_valid, metadata)
        
        print("\nPreprocessing completed successfully!")
        return metadata
    
    def run_kfold(self, n_splits=5):
        """Run complete preprocessing pipeline with K-fold CV"""
        print(f"Starting K-fold preprocessing with seed={self.seed}")
        print(f"Number of folds: {n_splits}")
        print(f"Raw data directory: {self.raw_data_dir}")
        print(f"Output directory: {self.output_dir}\n")
        
        # Load data
        X, y, file_info = self.load_all_data()
        
        # Create K-fold splits
        folds_metadata = self.create_kfold_splits(X, y, file_info, n_splits=n_splits)
        
        # Save metadata
        with open(self.output_dir / 'kfold_metadata.json', 'w') as f:
            json.dump(folds_metadata, f, indent=2)
        
        print(f"\nK-fold preprocessing completed successfully!")
        print(f"All folds saved to {self.output_dir}")
        return folds_metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess time series data')
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help='Path to raw data directory (e.g., ondevice_raw/2mm_#1)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['split', 'kfold'], default='split',
                        help='Preprocessing mode: split or kfold')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Validation set size (only for split mode)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds (only for kfold mode)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TimeSeriesPreprocessor(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run preprocessing
    if args.mode == 'split':
        preprocessor.run_train_valid_split(valid_size=args.valid_size)
    elif args.mode == 'kfold':
        preprocessor.run_kfold(n_splits=args.n_splits)


if __name__ == '__main__':
    main()