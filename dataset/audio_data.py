import numpy as np
import torch
import torch.nn.functional as F
import librosa


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_rate=16000,
        random_crop=False,
        contrastive_aug=None,
        weak_aug=None,
    ):
        self.sample_rate = sample_rate
        self.unit_samples = int(sample_rate * 0.95)
        self.random_crop = random_crop
        self.contrastive_aug = contrastive_aug
        self.weak_aug = weak_aug

    def __len__(self):
        raise NotImplementedError("implement me")

    def get_audio(self, index):
        raise NotImplementedError("implement me")

    def get_label(self, index):
        return None  # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index)  # shape is expected to be (cfg.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        unit_samples = self.unit_samples
        if l > unit_samples:
            start = np.random.randint(l - unit_samples) if self.random_crop else 0
            wav = wav[start : start + unit_samples]
        elif l < unit_samples:
            wav = np.pad(wav, (0, unit_samples - l), mode="constant", constant_values=0)

        if self.contrastive_aug is not None:
            pos_1 = self.contrastive_aug(wav, sample_rate=self.sample_rate)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(wav, sample_rate=self.sample_rate)
            else:
                pos_2 = self.contrastive_aug(wav, sample_rate=self.sample_rate)

            pos_1 = torch.from_numpy(pos_1).float()
            pos_2 = torch.from_numpy(pos_2).float()
        else:
            pos_1 = torch.from_numpy(wav).float()
            pos_2 = torch.from_numpy(wav).float()

        # Return pair of waveforms
        label = self.get_label(index)
        return (pos_1, pos_2) if label is None else (pos_1, label, pos_2, label)


class WavDatasetPair(BaseRawAudioDataset):
    def __init__(
        self,
        sample_rate,
        audio_files,
        labels,
        random_crop=False,
        contrastive_aug=None,
        weak_aug=None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            random_crop=random_crop,
            contrastive_aug=contrastive_aug,
            weak_aug=weak_aug,
        )
        self.files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def get_audio(self, index):
        filename = self.files[index]
        wav, sr = librosa.load(filename, sr=self.sample_rate)
        assert (
            sr == 16000
        ), f"Convert .wav files to {self.sample_rate} Hz. {filename} has {sr} Hz."
        return wav

    def get_label(self, index):
        return None if self.labels is None else torch.tensor(self.labels[index])
