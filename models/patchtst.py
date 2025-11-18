# models/patchtst_hf.py
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification


class PatchTSTClassifier(nn.Module):
    """
    PatchTST for classification using HuggingFace official implementation
    
    Parameters
    ----------
    n_vars : int
        Number of input variables (channels)
    n_classes : int
        Number of output classes
    seq_len : int
        Sequence length (context_length)
    patch_len : int, default=16
        Length of each patch
    stride : int, default=8
        Stride for patching
    d_model : int, default=128
        Hidden dimension
    n_heads : int, default=16
        Number of attention heads
    n_layers : int, default=3
        Number of encoder layers
    d_ff : int, default=256
        FFN dimension
    dropout : float, default=0.1
        Dropout rate
    head_dropout : float, default=0.1
        Classification head dropout
    use_cls_token : bool, default=True
        Whether to use [CLS] token for classification
    """
    def __init__(self, n_vars, n_classes, seq_len,
                 patch_len=16, stride=8,
                 d_model=128, n_heads=16, n_layers=3,
                 d_ff=256, dropout=0.1, head_dropout=0.1,
                 use_cls_token=True):
        super().__init__()
        
        self.n_vars = n_vars
        self.n_classes = n_classes
        self.seq_len = seq_len
        
        # HuggingFace PatchTST Config for Classification
        config = PatchTSTConfig(
            num_input_channels=n_vars,
            num_targets=n_classes,
            context_length=seq_len,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            ffn_dim=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            use_cls_token=use_cls_token,
            do_mask_input=False,  # Classification에선 masking 안함
        )
        
        # PatchTSTForClassification 사용
        self.model = PatchTSTForClassification(config)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch, n_vars, seq_len] or [batch, seq_len, n_vars]
        
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, n_classes]
        """
        # HuggingFace expects [batch, seq_len, n_vars]
        if x.shape[1] == self.n_vars and x.shape[2] == self.seq_len:
            # [batch, n_vars, seq_len] -> [batch, seq_len, n_vars]
            x = x.permute(0, 2, 1)
        
        # Forward pass (training mode에서는 Trainer가 loss 계산)
        outputs = self.model(
            past_values=x,
            return_dict=True
        )
        
        return outputs.prediction_logits