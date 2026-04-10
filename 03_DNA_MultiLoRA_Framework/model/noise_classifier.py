"""
Noise Classifier Wrapper
- 01_Noise_Classification 모듈 로드 + 노이즈 도메인 예측
- 모델별 입력 feature 자동 추출
"""

import os
import sys
import types
import importlib
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

NC_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../01_Noise_Classification"))

DEFAULT_CLASS_TO_DOMAIN = {
    0: 0, 1: 1, 2: 1, 3: 6, 4: 3,
    5: 4, 6: 5, 7: 5, 8: 7, 9: 2,
}

# 3FF models: spec + mfcc + f0
THREE_FF_MODELS = [
    "cnn8rnn_3ff_crossmodal", "cnn8rnn_3ff_base",
    "cnn8rnn_3ff_gating", "cnn8rnn_3ff_sigmoid", "cnn8rnn_3ff_softmax",
    "cnnlstm_3ff", "cnnlstm_3ff_crossmodal", "cnnlstm_3ff_gating",
    "cnnlstm_3ff_interaction", "cnnlstm_3ff_softmax", "cnnlstm_3ff_weight",
]

# SSAST model
SSAST_MODELS = ["ssast_model"]

# Mel-spectrogram single input models (22050Hz, [B, 1, n_mels, T])
MELSPEC_MODELS = ["cnnlstm"]

# HTS-AT model (32kHz)
HTSAT_MODELS = ["htsat_model"]

# 3FF feature config (from dataset_cnnlstm_3ff.py)
TARGET_SR_3FF = 22050
N_MELS = 128
N_MFCC = 40
N_FFT_3FF = 1024
HOP_3FF = 512
CLIP_DURATION = 10.0


def _load_nc_module(module_name):
    """Load NC model module, handling model package conflict."""
    saved = {}
    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model."):
            saved[k] = sys.modules.pop(k)

    fake_pkg = types.ModuleType("model")
    fake_pkg.__path__ = [os.path.join(NC_BASE, "model")]
    fake_pkg.__package__ = "model"
    sys.modules["model"] = fake_pkg

    sys.path.insert(0, NC_BASE)
    try:
        mod = importlib.import_module(f"model.{module_name}")
    finally:
        sys.path.remove(NC_BASE)
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model."):
                del sys.modules[k]
        sys.modules.update(saved)

    return mod


class NoiseClassifierWrapper:
    def __init__(self, module_name, checkpoint_path, nc_config_name=None,
                 class_to_domain=None, device="cuda"):
        self.device = device
        self.class_to_domain = class_to_domain or DEFAULT_CLASS_TO_DOMAIN
        self.module_name = module_name
        self.is_3ff = module_name in THREE_FF_MODELS
        self.is_melspec = module_name in MELSPEC_MODELS
        self.is_ssast = module_name in SSAST_MODELS
        self.is_htsat = module_name in HTSAT_MODELS

        # Load NC config for model-specific params
        import yaml
        cfg_name = nc_config_name or module_name
        nc_cfg_path = os.path.join(NC_BASE, "config", f"{cfg_name}.yaml")
        if os.path.exists(nc_cfg_path):
            with open(nc_cfg_path) as f:
                self.nc_cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.nc_cfg = {}

        # Load NC model
        model_module = _load_nc_module(module_name)

        # Find the main model class: the one that accepts num_classes or label_dim
        model_class = None
        candidates = []
        for attr_name in dir(model_module):
            attr = getattr(model_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr is not nn.Module:
                candidates.append((attr_name, attr))

        import inspect
        for name, cls in candidates:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())
            if 'num_classes' in params or 'label_dim' in params:
                model_class = cls
                break
        if model_class is None and candidates:
            model_class = candidates[-1][1]

        if model_class is None:
            raise ValueError(f"No nn.Module found in model.{module_name}")

        # Build model with appropriate kwargs
        if self.is_ssast:
            pretrained_mdl = self.nc_cfg.get("pretrained_mdl_path", "")
            if not os.path.isabs(pretrained_mdl):
                # Config paths are relative to config/ dir
                pretrained_mdl = os.path.normpath(os.path.join(NC_BASE, "config", pretrained_mdl))
            self.model = model_class(
                label_dim=10,
                fshape=self.nc_cfg.get("fshape", 16),
                tshape=self.nc_cfg.get("tshape", 16),
                fstride=self.nc_cfg.get("fstride", 10),
                tstride=self.nc_cfg.get("tstride", 10),
                input_fdim=128, input_tdim=self.nc_cfg.get("target_length", 512),
                model_size=self.nc_cfg.get("model_size", "tiny"),
                pretrain_stage=False,
                load_pretrained_mdl_path=pretrained_mdl,
            )
        elif self.is_htsat:
            htsat_pretrained = self.nc_cfg.get("pretrained_mdl_path", None)
            if htsat_pretrained and not os.path.isabs(htsat_pretrained):
                htsat_pretrained = os.path.normpath(os.path.join(NC_BASE, "config", htsat_pretrained))
            self.model = model_class(num_classes=10, load_pretrained_path=htsat_pretrained)
        else:
            self.model = model_class(num_classes=10)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(device)
        self.model.eval()

        # Setup mel transforms for melspec models
        if self.is_melspec:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=TARGET_SR_3FF, n_fft=N_FFT_3FF,
                hop_length=HOP_3FF, n_mels=N_MELS
            )
            self.amplitude_to_db = T.AmplitudeToDB(stype='power')
            self.clip_samples_3ff = int(CLIP_DURATION * TARGET_SR_3FF)

        # Setup 3FF transforms
        if self.is_3ff:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=TARGET_SR_3FF, n_fft=N_FFT_3FF,
                hop_length=HOP_3FF, n_mels=N_MELS
            )
            self.amplitude_to_db = T.AmplitudeToDB(stype='power')
            self.mfcc_transform = T.MFCC(
                sample_rate=TARGET_SR_3FF, n_mfcc=N_MFCC,
                melkwargs={'n_fft': N_FFT_3FF, 'hop_length': HOP_3FF, 'n_mels': N_MELS}
            )
            self.clip_samples_3ff = int(CLIP_DURATION * TARGET_SR_3FF)

        # SSAST params
        if self.is_ssast:
            self.ssast_target_length = self.nc_cfg.get("target_length", 512)
            self.ssast_norm_mean = self.nc_cfg.get("norm_mean", -4.2677393)
            self.ssast_norm_std = self.nc_cfg.get("norm_std", 4.5689974)

        # HTSAT params
        if self.is_htsat:
            self.htsat_sr = self.nc_cfg.get("sample_rate", 32000)
            self.htsat_clip_duration = self.nc_cfg.get("clip_duration", 10.0)

        mode = "3FF" if self.is_3ff else "melspec" if self.is_melspec else "SSAST" if self.is_ssast else "HTSAT" if self.is_htsat else "waveform"
        print(f"[NC] Loaded {module_name} ({mode})")

    def _extract_3ff(self, waveform_16k):
        """Extract mel-spec, mfcc, f0 from 16kHz waveform tensor [1, T]."""
        # Resample to 22050
        waveform = torchaudio.functional.resample(waveform_16k, 16000, TARGET_SR_3FF)

        # Clip/pad
        n = waveform.shape[1]
        if n >= self.clip_samples_3ff:
            waveform = waveform[:, :self.clip_samples_3ff]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples_3ff - n))

        spec = self.mel_transform(waveform)          # [1, 128, T]
        spec = self.amplitude_to_db(spec)
        mfcc = self.mfcc_transform(waveform)         # [1, 40, T]

        # F0 (simple YIN-like, avoid crepe dependency)
        T_spec = spec.shape[2]
        wav_np = waveform.squeeze(0).numpy()
        n_frames = (len(wav_np) - N_FFT_3FF) // HOP_3FF + 1
        f0_values = []
        for i in range(n_frames):
            start = i * HOP_3FF
            frame = wav_np[start:start + N_FFT_3FF]
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            min_lag = int(TARGET_SR_3FF / 500)
            max_lag = min(int(TARGET_SR_3FF / 50), len(autocorr))
            if max_lag > min_lag:
                peaks = autocorr[min_lag:max_lag]
                lag = np.argmax(peaks) + min_lag if len(peaks) > 0 else 0
                f0_values.append(TARGET_SR_3FF / lag if lag > 0 else 0.0)
            else:
                f0_values.append(0.0)
        f0_np = np.array(f0_values)
        f0_np = np.where(f0_np > 0, np.log(f0_np + 1e-8), 0.0)
        if f0_np.std() > 0:
            f0_np = (f0_np - f0_np.mean()) / (f0_np.std() + 1e-8)
        f0 = torch.from_numpy(f0_np).float().unsqueeze(0)  # [1, T_f0]

        # Align time dims
        T_target = T_spec
        if mfcc.shape[2] != T_target:
            mfcc = torch.nn.functional.interpolate(
                mfcc.unsqueeze(0), size=(N_MFCC, T_target),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        if f0.shape[1] != T_target:
            f0 = torch.nn.functional.interpolate(
                f0.unsqueeze(0), size=T_target,
                mode='linear', align_corners=False
            ).squeeze(0)

        return spec, mfcc, f0

    def _extract_ssast_fbank(self, waveform_16k):
        """Extract fbank for SSAST from 16kHz waveform [1, T]."""
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_16k, htk_compat=True, sample_frequency=16000,
            use_energy=False, window_type='hanning',
            num_mel_bins=128, dither=0.0, frame_shift=10
        )  # [T, 128]
        # Normalize
        fbank = (fbank - self.ssast_norm_mean) / (self.ssast_norm_std * 2)
        # Pad/crop
        tgt = self.ssast_target_length
        if fbank.shape[0] < tgt:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, tgt - fbank.shape[0]))
        else:
            fbank = fbank[:tgt]
        return fbank  # [T, 128]

    def _extract_htsat(self, waveform_16k):
        """Resample to 32kHz for HTSAT."""
        waveform = torchaudio.functional.resample(waveform_16k, 16000, self.htsat_sr)
        clip_samples = int(self.htsat_clip_duration * self.htsat_sr)
        if waveform.shape[1] >= clip_samples:
            waveform = waveform[:, :clip_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, clip_samples - waveform.shape[1]))
        return waveform.squeeze(0)  # [T]

    @torch.no_grad()
    def predict(self, waveforms):
        """
        Args:
            waveforms: [batch, length] tensor (16kHz raw audio)
        Returns:
            domain_ids: [batch] tensor (0~7)
            class_ids: [batch] tensor (0~9)
        """
        if self.is_3ff:
            specs, mfccs, f0s = [], [], []
            for i in range(waveforms.shape[0]):
                wav = waveforms[i].unsqueeze(0)  # [1, T]
                spec, mfcc, f0 = self._extract_3ff(wav)
                specs.append(spec)
                mfccs.append(mfcc)
                f0s.append(f0)
            spec_batch = torch.stack(specs).to(self.device)
            mfcc_batch = torch.stack(mfccs).to(self.device)
            f0_batch = torch.stack(f0s).to(self.device)
            logits = self.model(spec_batch, mfcc_batch, f0_batch)

        elif self.is_melspec:
            specs = []
            for i in range(waveforms.shape[0]):
                wav = waveforms[i].unsqueeze(0)  # [1, T]
                wav_22k = torchaudio.functional.resample(wav, 16000, TARGET_SR_3FF)
                n = wav_22k.shape[1]
                if n >= self.clip_samples_3ff:
                    wav_22k = wav_22k[:, :self.clip_samples_3ff]
                else:
                    wav_22k = torch.nn.functional.pad(wav_22k, (0, self.clip_samples_3ff - n))
                spec = self.mel_transform(wav_22k)
                spec = self.amplitude_to_db(spec)
                specs.append(spec)
            spec_batch = torch.stack(specs).to(self.device)  # [B, 1, 128, T]
            logits = self.model(spec_batch)

        elif self.is_ssast:
            fbanks = []
            for i in range(waveforms.shape[0]):
                fbank = self._extract_ssast_fbank(waveforms[i].unsqueeze(0))
                fbanks.append(fbank)
            fbank_batch = torch.stack(fbanks).to(self.device)
            logits = self.model(fbank_batch, task='ft_avgtok')

        elif self.is_htsat:
            wavs = []
            for i in range(waveforms.shape[0]):
                wav = self._extract_htsat(waveforms[i].unsqueeze(0))
                wavs.append(wav)
            wav_batch = torch.stack(wavs).to(self.device)
            logits = self.model(wav_batch)

        else:
            # cnnlstm, cnn8rnn single input
            waveforms = waveforms.to(self.device)
            logits = self.model(waveforms)

        if isinstance(logits, tuple):
            logits = logits[0]
        class_ids = logits.argmax(dim=-1)

        domain_ids = torch.tensor(
            [self.class_to_domain.get(c.item(), 0) for c in class_ids],
            device=self.device
        )
        return domain_ids, class_ids
