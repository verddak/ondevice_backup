# data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    시계열 분류를 위한 PyTorch Dataset
    
    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Shape: (n_samples, n_channels, n_timesteps)
    y : np.ndarray or torch.Tensor
        Shape: (n_samples,)
    transforms : list or callable, optional
        Augmentation transforms to apply
    """
    def __init__(self, X, y, transforms=None):
        # numpy array를 tensor로 변환
        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X)
        else:
            self.X = X.float()
        
        # y 처리: 문자열이면 숫자로 변환
        if isinstance(y, np.ndarray):
            # 문자열 타입 체크
            if y.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, object
                # 고유한 클래스를 찾아서 0, 1, 2, ... 로 매핑
                unique_classes = np.unique(y)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
                y_numeric = np.array([self.class_to_idx[cls] for cls in y])
                self.y = torch.LongTensor(y_numeric)
                print(f"Converted string labels to numeric: {dict(list(self.class_to_idx.items())[:5])}")
            else:
                self.y = torch.LongTensor(y)
        else:
            self.y = y.long()
        
        self.transforms = transforms
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플 반환
        
        Returns
        -------
        x : torch.Tensor
            Shape: (n_channels, n_timesteps)
        y : torch.Tensor
            Shape: ()
        """
        x = self.X[idx].clone()  # clone()으로 원본 보호
        y = self.y[idx]
        
        # Apply augmentations
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for transform in self.transforms:
                    if transform is not None:
                        x = transform(x)
            else:
                x = self.transforms(x)
        
        return x, y