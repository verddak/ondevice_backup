# augment/torch_augmentations.py
"""
Pure PyTorch implementation of time series augmentation methods.
No dependency on tsai.
"""
import torch
import torch.nn as nn
import random
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional


class BaseAugmentation(nn.Module):
    """Base class for all augmentation methods"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        """
        Args:
            prob: Probability of applying augmentation (0.0 - 1.0)
            magnitude: Strength of augmentation (0 - 9)
        """
        super().__init__()
        self.prob = prob
        self.magnitude = magnitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_vars, seq_len)
        
        Returns:
            Augmented tensor of same shape
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob={self.prob:.2f}, magnitude={self.magnitude})"


class JITTERING(BaseAugmentation):
    """Add Gaussian noise to time series"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        super().__init__(prob, magnitude)
        # magnitude 0~9 → 0~0.025 범위로 정규화
        self.noise_std = magnitude * 0.025 / 9.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
        
        batch_size = x.shape[0]
        
        # 각 샘플별로 확률적으로 적용 여부 결정
        apply_mask = torch.rand(batch_size, device=x.device) < self.prob
        
        # 노이즈 생성 (평균=0, 표준편차=noise_std/3)
        std = self.noise_std / 3
        noise = torch.normal(mean=0.0, std=std, size=x.shape, device=x.device)
        
        # 마스크를 적용하여 선택된 샘플에만 노이즈 추가
        apply_mask = apply_mask.view(-1, 1, 1)
        noise = noise * apply_mask
        
        return x + noise


class HOMOGENEOUS(BaseAugmentation):
    """Scale time series by random factor"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        super().__init__(prob, magnitude)
        # magnitude 0~9 → 0~0.5 범위로 정규화
        self.scale_range = magnitude * 0.5 / 9.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
        
        batch_size = x.shape[0]
        result = x.clone()
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
                result[i] = x[i] * scale
        
        return result


class HOMOGENEOUS_UP(BaseAugmentation):
    """Scale time series up by random factor"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        super().__init__(prob, magnitude)
        self.scale_range = magnitude * 0.5 / 9.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
        
        batch_size = x.shape[0]
        result = x.clone()
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                scale = random.uniform(1, 1 + self.scale_range)
                result[i] = x[i] * scale
        
        return result


class HOMOGENEOUS_DOWN(BaseAugmentation):
    """Scale time series down by random factor"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        super().__init__(prob, magnitude)
        self.scale_range = magnitude * 0.5 / 9.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
        
        batch_size = x.shape[0]
        result = x.clone()
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                scale = random.uniform(1 - self.scale_range, 1)
                result[i] = x[i] * scale
        
        return result


class SEGMENT_MASKING(BaseAugmentation):
    """Mask segments of time series with marginal values"""
    def __init__(
        self, 
        prob: float = 0.5, 
        magnitude: int = 0,
        start_idx: int = 4500,
        end_idx: int = 5250,
        block_size_range: tuple = (30,50)
    ):
        super().__init__(prob, magnitude)
        # magnitude 0~9 → 0~0.5 범위로 정규화
        self.mask_ratio = magnitude * 0.5 / 9.0
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.block_size_range = block_size_range
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
            
        if self.mask_ratio <= 0:
            return x
            
        batch_size, channels, seq_len = x.shape
        result = x.clone()
        
        # 보호 구간 유효성 검사
        start_idx = max(0, self.start_idx)
        end_idx = min(seq_len, self.end_idx)
        
        if start_idx >= end_idx:
            return x
        
        # 마스킹 가능한 인덱스들 (보호 구간 제외)
        available_indices = []
        if start_idx > 0:
            available_indices.extend(range(0, start_idx))
        if end_idx < seq_len:
            available_indices.extend(range(end_idx, seq_len))
            
        if not available_indices:
            return x
        
        # 마스킹할 총 길이
        total_mask_length = int(len(available_indices) * self.mask_ratio)
        min_block_size = self.block_size_range[0]
        
        if total_mask_length < min_block_size:
            return x
            
        # 각 샘플에 확률적으로 마스킹 적용
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                mask_positions = self._generate_blocks(
                    available_indices.copy(), 
                    total_mask_length
                )
                
                # 모든 채널에 동일한 마스킹 (marginal value 사용)
                for ch in range(channels):
                    for pos in mask_positions:
                        if pos > 0:
                            result[i, ch, pos] = result[i, ch, pos - 1]
                        elif pos < seq_len - 1:
                            result[i, ch, pos] = result[i, ch, pos + 1]
                            
        return result
    
    def _generate_blocks(self, available_indices, total_mask_length):
        """보호 구간을 피해서 블록 마스킹 위치 생성"""
        min_size, max_size = self.block_size_range
        mask_positions = []
        remaining = total_mask_length
        
        np.random.shuffle(available_indices)
        
        while remaining >= min_size and available_indices:
            # 블록 크기 결정
            block_size = min(
                max_size, 
                remaining, 
                np.random.randint(min_size, min(max_size, remaining) + 1)
            )
            
            # 연속된 블록을 만들 수 있는 위치 찾기
            valid_starts = []
            for i in range(len(available_indices) - block_size + 1):
                if all(available_indices[i+j] == available_indices[i] + j 
                       for j in range(block_size)):
                    valid_starts.append(i)
            
            if not valid_starts:
                # 연속 블록 불가능하면 개별 위치 선택
                positions = available_indices[:remaining]
                mask_positions.extend(positions[:block_size])
                remaining -= len(positions[:block_size])
                break
            
            # 랜덤 시작 위치 선택
            start_idx = valid_starts[np.random.randint(len(valid_starts))]
            block_positions = available_indices[start_idx:start_idx + block_size]
            
            mask_positions.extend(block_positions)
            remaining -= block_size
            
            # 사용된 위치들 제거
            for pos in block_positions:
                available_indices.remove(pos)

        return sorted(set(mask_positions))


class WINDOW_WARPING(BaseAugmentation):
    """Warp a specific window of time series"""
    def __init__(
        self, 
        prob: float = 0.5, 
        magnitude: int = 0,
        start_idx: int = 4500,
        end_idx: int = 5250
    ):
        super().__init__(prob, magnitude)
        # magnitude 0~9 → 0~0.3 범위로 정규화
        self.warp_strength = magnitude * 0.3 / 9.0
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
            
        if self.warp_strength <= 0:
            return x
            
        batch_size, channels, seq_len = x.shape
        result = x.clone()
        
        # 구간 유효성 검사
        start_idx = max(0, self.start_idx)
        end_idx = min(seq_len, self.end_idx)
        window_len = end_idx - start_idx
        
        if window_len <= 1:
            return x
        
        # 각 샘플에 확률적으로 window warping 적용
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                # 상수 스케일링 팩터 생성
                scale_factor = np.random.uniform(
                    1 - self.warp_strength, 
                    1 + self.warp_strength
                )
                
                # window 구간에 선형 mapping
                warped_indices = np.linspace(0, window_len - 1, window_len) * scale_factor
                warped_indices = np.clip(warped_indices, 0, window_len - 1)
                
                # 각 채널에 보간 적용
                for ch in range(channels):
                    window_signal = x[i, ch, start_idx:end_idx].cpu().numpy()
                    f = CubicSpline(np.arange(window_len), window_signal)
                    warped_values = f(warped_indices)
                    
                    result[i, ch, start_idx:end_idx] = torch.tensor(
                        warped_values, dtype=x.dtype, device=x.device
                    )
        
        return result


class TIME_WARPING(BaseAugmentation):
    """Warp entire time series"""
    def __init__(self, prob: float = 0.5, magnitude: int = 0):
        super().__init__(prob, magnitude)
        # magnitude 0~9 → 0~0.3 범위로 정규화
        self.warp_strength = magnitude * 0.3 / 9.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
            
        if self.warp_strength <= 0:
            return x
            
        batch_size, channels, seq_len = x.shape
        result = x.clone()
        
        # 각 샘플에 확률적으로 전체 time warping 적용
        for i in range(batch_size):
            if torch.rand(1).item() < self.prob:
                # 상수 스케일링 팩터 생성
                scale_factor = np.random.uniform(
                    1 - self.warp_strength, 
                    1 + self.warp_strength
                )
                
                # 전체 시퀀스에 선형 mapping
                warped_indices = np.linspace(0, seq_len - 1, seq_len) * scale_factor
                warped_indices = np.clip(warped_indices, 0, seq_len - 1)
                
                # 각 채널에 보간 적용
                for ch in range(channels):
                    signal = x[i, ch].cpu().numpy()
                    f = CubicSpline(np.arange(seq_len), signal)
                    warped_values = f(warped_indices)
                    
                    result[i, ch] = torch.tensor(
                        warped_values, dtype=x.dtype, device=x.device
                    )
        
        return result


# Augmentation registry for easy access
AUGMENTATION_REGISTRY = {
    'JITTERING': JITTERING,
    'HOMOGENEOUS': HOMOGENEOUS,
    'HOMOGENEOUS_UP': HOMOGENEOUS_UP,
    'HOMOGENEOUS_DOWN': HOMOGENEOUS_DOWN,
    'SEGMENT_MASKING': SEGMENT_MASKING,
    'WINDOW_WARPING': WINDOW_WARPING,
    'TIME_WARPING': TIME_WARPING,
}


def get_augmentation(name: str, prob: float = 0.5, magnitude: int = 0, **kwargs):
    """
    Factory function to get augmentation by name.
    
    Args:
        name: Name of augmentation method
        prob: Probability of applying augmentation
        magnitude: Strength of augmentation (0-9)
        **kwargs: Additional arguments for specific augmentations
    
    Returns:
        Augmentation instance
    """
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. "
                        f"Available: {list(AUGMENTATION_REGISTRY.keys())}")
    
    return AUGMENTATION_REGISTRY[name](prob=prob, magnitude=magnitude, **kwargs)

def get_default_augmentations():
    """
    Get default list of augmentation classes for PBA search
    
    Returns:
        List of augmentation classes (not instances!)
    """
    return [
        JITTERING,
        HOMOGENEOUS,
        HOMOGENEOUS_UP,
        HOMOGENEOUS_DOWN,
        SEGMENT_MASKING,
        WINDOW_WARPING,
        TIME_WARPING,
    ]