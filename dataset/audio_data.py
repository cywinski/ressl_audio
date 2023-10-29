import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, sample_rate=16000, tfms=None, random_crop=False):
        self.sample_rate = sample_rate
        self.unit_samples = int(sample_rate * 0.95)
        self.tfms = tfms
        self.random_crop = random_crop

    def __len__(self):
        raise NotImplementedError('implement me')

    def get_audio(self, index):
        raise NotImplementedError('implement me')

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index) # shape is expected to be (cfg.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        unit_samples = self.unit_samples
        if l > unit_samples:
            start = np.random.randint(l - unit_samples) if self.random_crop else 0
            wav = wav[start:start + unit_samples]
        elif l < unit_samples:
            wav = F.pad(wav, (0, unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)

        # TODO: Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        label = self.get_label(index)
        return (wav, wav) if label is None else (wav, label, wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self, sample_rate, audio_files, labels, tfms=None, random_crop=False):
        super().__init__(sample_rate=sample_rate, tfms=tfms, random_crop=random_crop)
        self.files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def get_audio(self, index):
        filename = self.files[index]
        wav, sr = torchaudio.load(filename)
        assert sr == 16000, f'Convert .wav files to {self.sample_rate} Hz. {filename} has {sr} Hz.'
        return wav[0]

    def get_label(self, index):
        return None if self.labels is None else torch.tensor(self.labels[index])
