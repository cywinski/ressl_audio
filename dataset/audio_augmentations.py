from torch import nn
import torch
import audiomentations


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
    return audiomentations.Compose(
        [
            # audiomentations.PolarityInversion(),
            audiomentations.HighPassFilter(),
            audiomentations.LowPassFilter(),
            audiomentations.AddGaussianNoise(),
            audiomentations.TimeStretch(),
            audiomentations.PitchShift(),
            # audiomentations.Shift(),
            # audiomentations.Normalize(),
        ]
    )


def get_weak_augment():
    return audiomentations.Compose(
        [
            # audiomentations.PolarityInversion(),
            audiomentations.AddGaussianNoise(),
            # audiomentations.PitchShift(),
            # audiomentations.Normalize(),
        ]
    )


def get_augment_by_class_names(aug_classes_list):
    aug_list = [audiomentations.AddGaussianNoise()] + [
        getattr(audiomentations, aug_class)() for aug_class in eval(aug_classes_list)
    ] + [audiomentations.Normalize()]
    return audiomentations.Compose(aug_list)
