import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
import pickle
import hashlib
from pathlib import Path

################################
# gen_list: 프로토콜 파일 파싱
################################
def gen_list(protocol_file, is_train=False, is_dev=False, is_eval=False):
    label_mapping = {
        "clean": 0,
        "background_noise": 1,
        "background_music": 2,
        "gaussian_noise": 3,
        "band_pass_filter": 4,
        "manipulation": 5,
        "auto_tune": 6,
        "echo": 7,
        "reverberation": 8
    }
    
    d_meta = {}
    file_list = []

    if is_train:
        with open(protocol_file, "r") as f:
            for line in f:
                speaker, file_path, label1, label2, subset = line.strip().split(" ")
                if subset.lower() == "train":
                    file_list.append(file_path)
                    d_meta[file_path] = label_mapping.get(label2, -1)
        return d_meta, file_list

    if is_dev:
        with open(protocol_file, "r") as f:
            for line in f:
                speaker, file_path, label1, label2, subset = line.strip().split(" ")
                if subset.lower() == "dev":
                    file_list.append(file_path)
                    d_meta[file_path] = label_mapping.get(label2, -1)
        return d_meta, file_list

    if is_eval:
        with open(protocol_file, "r") as f:
            for line in f:
                speaker, file_path, label1, label2, subset = line.strip().split(" ")
                if subset.lower() == "eval":
                    file_list.append(file_path)
        return file_list


################################
# Feature Extraction
################################
def fix_length(waveform, target_len=64000, random_start=True):
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    cur_len = waveform.shape[-1]
    if cur_len > target_len:
        start = np.random.randint(0, cur_len - target_len) if random_start else 0
        waveform = waveform[start: start + target_len]
    elif cur_len < target_len:
        repeat = (target_len // cur_len) + 1
        waveform = waveform.repeat(repeat)[:target_len]
    return waveform


################################
# MultiFeatureDataset (Train/Dev/Eval)
################################
class MultiFeatureDataset(Dataset):
    def __init__(self, list_IDs, labels=None, sr=16000, target_len=64000, is_train=True, is_eval=False, 
                 cache_dir=None, enable_cache=True, train_random_start=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.sr = sr
        self.target_len = target_len
        self.is_train = is_train
        self.is_eval = is_eval
        self.enable_cache = enable_cache
        self.train_random_start = train_random_start  # 추가: 첫 epoch만 True, 이후 False로 사용

        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 128
        self.window = 'hamming'
        
        # 캐시 디렉토리 설정
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
        self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Feature cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, utt_id):
        """Generate cache file path for given utterance ID"""
        # 파일 경로를 hash로 변환해서 캐시 파일명 생성
        path_hash = hashlib.md5(utt_id.encode()).hexdigest()
        # feature extraction 설정도 포함해서 캐시 무효화
        config_str = f"{self.sr}_{self.target_len}_{self.n_fft}_{self.hop_length}_{self.n_mels}_{self.n_mfcc}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_filename = f"{path_hash}_{config_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, cache_path):
        """Load features from cache file"""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_path, features):
        """Save features to cache file"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")
    
    def _extract_features(self, wav_np):
        """Extract all features from waveform"""
        # Spectrogram
        stft = librosa.stft(wav_np, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)
        spec_db = librosa.amplitude_to_db(np.abs(stft))
        spec_db = torch.from_numpy(spec_db).float().unsqueeze(0)

        # MFCC
        frame_size = int(0.025 * self.sr)  # 25ms
        hop_size = int(0.010 * self.sr)    # 10ms
        mfcc = librosa.feature.mfcc(y=wav_np, sr=self.sr, n_mfcc=13,
                                    n_fft=frame_size, hop_length=hop_size, n_mels=self.n_mels)
        mfcc = torch.from_numpy(mfcc).float().unsqueeze(0)

        # F0 Extraction
        f0 = librosa.yin(wav_np, fmin=50, fmax=600, sr=self.sr,
                        frame_length=self.n_fft, hop_length=self.hop_length)
        f0 = np.nan_to_num(f0)
        f0 = torch.from_numpy(f0).float().unsqueeze(0)
        
        return spec_db, mfcc, f0

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        # train 모드일 때 train_random_start가 False면 캐시 사용
        use_cache = self.enable_cache and (not self.is_train or (self.is_train and not self.train_random_start))
        # 캐시 활성화 시 캐시에서 먼저 확인
        if use_cache:
            cache_path = self._get_cache_path(utt_id)
            cached_features = self._load_from_cache(cache_path)
            if cached_features is not None:
                spec_db, mfcc, f0 = cached_features
                if idx < 3:
                    print(f"[CACHE HIT] {os.path.basename(utt_id)} - spec_db: {spec_db.shape}, mfcc: {mfcc.shape}, f0: {f0.shape}")
            else:
                wav, fs = librosa.load(utt_id, sr=self.sr)
                wav = fix_length(torch.tensor(wav), self.target_len, random_start=self.is_train)
                wav_np = wav.numpy()
                spec_db, mfcc, f0 = self._extract_features(wav_np)
                self._save_to_cache(cache_path, (spec_db, mfcc, f0))
                if idx < 3:
                    print(f"[EXTRACTED & CACHED] {os.path.basename(utt_id)} - spec_db: {spec_db.shape}, mfcc: {mfcc.shape}, f0: {f0.shape}")
        else:
            wav, fs = librosa.load(utt_id, sr=self.sr)
            wav = fix_length(torch.tensor(wav), self.target_len, random_start=self.is_train)
            wav_np = wav.numpy()
            spec_db, mfcc, f0 = self._extract_features(wav_np)
            if idx < 3:
                print(f"[NO CACHE] {os.path.basename(utt_id)} - spec_db: {spec_db.shape}, mfcc: {mfcc.shape}, f0: {f0.shape}")
        if self.is_eval:
            return spec_db, mfcc, f0, utt_id
        else:
            label = self.labels[utt_id]
            return spec_db, mfcc, f0, label

    def clear_cache(self):
        """Clear all cached features"""
        if self.enable_cache and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cache cleared: {self.cache_dir}")
    
    def get_cache_info(self):
        """Get cache statistics"""
        if not self.enable_cache or not self.cache_dir.exists():
            return {"enabled": False}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_cached_files": len(cache_files),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cache_files": len(cache_files)
        }
