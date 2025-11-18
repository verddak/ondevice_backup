# models/simple_cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for time series classification
    
    Parameters
    ----------
    n_vars : int
        Number of input channels (variables)
    n_classes : int
        Number of output classes
    seq_len : int
        Sequence length (timesteps)
    """
    def __init__(self, n_vars, n_classes, seq_len):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv1d(n_vars, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # seq_len -> seq_len//2
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # seq_len//2 -> seq_len//4
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # -> 1
        
        # FC layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # x: (batch, n_vars, seq_len)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # (batch, 256, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # FC layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x