from evar.ar_base import (
    BaseAudioRepr,
    ToLogMelSpec,
    calculate_norm_stats,
    normalize_spectrogram,
    temporal_pooling,
)
from evar.model_utils import load_pretrained_weights
import logging
from network.base_model import *
from evar.common import torch
from network.audio_model import AudioNTT2022Encoder
import audiomentations
from dataset.audio_augmentations import NormalizeBatch


class AR_RESSL(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)
        self.body = AudioNTT2022Encoder(n_mels=64)
        self.normalize_wav = audiomentations.Normalize()
        self.post_norm = NormalizeBatch()

        prefix = "net."
        if cfg.weight_file is not None:
            state_dict = torch.load(cfg.weight_file, map_location="cuda")["model"]
            for k in list(state_dict.keys()):
                if not k.startswith(prefix):
                    del state_dict[k]
                if k.startswith(prefix):
                    state_dict[k[len(prefix) :]] = state_dict[k]
                    del state_dict[k]
            self.body.load_state_dict(state_dict)

    def precompute(self, device, data_loader):
        # self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)
        pass

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = x.unsqueeze(1)  # -> B,1,F,T
        # x = self.post_norm(x)
        x = self.body(x)  # -> B,T,D=C*F
        x = x.transpose(1, 2)  # -> B,D,T
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x
