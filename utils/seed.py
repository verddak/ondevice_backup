# utils/seed.py
import random
import numpy as np
import torch
import os

def set_seed(seed):
    """
    재현성을 위한 모든 시드 고정
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 환경변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set random seed to {seed}")