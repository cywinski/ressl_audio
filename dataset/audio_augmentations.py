from torch import nn
import torch
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    AddGaussianSNR,
    Gain,
    Reverse,
    RoomSimulator,
)


class NormalizeBatch(nn.Module):
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=None):
        if axis is None:
            axis = [0, 2, 3]
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(
            X.std(dim=self.axis, keepdims=True), torch.finfo().eps, torch.finfo().max
        )
        return (X - _mean) / _std

    def __repr__(self):
        return f"{self.__class__.__name__}(axis={self.axis})"


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.

    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats, axis=None):
        if axis is None:
            axis = [1, 2]
        super().__init__()
        self.axis = axis
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, axis={self.axis})"


def get_contrastive_augment():
    return Compose(
        [
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # AddGaussianSNR(min_snr_db=5.0, max_snr_db=20.0, p=1.0),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
            # Reverse(p=0.5),
            # RoomSimulator()
        ]
    )


def get_weak_augment():
    return Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            # Reverse(p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # Shift(p=0.5),
        ]
    )
