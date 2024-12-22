import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from tqdm import tqdm
from fft import fft
from stft import stft
from mfcc import mfcc, mel_spectrogram, spectrogram

class AudioDataset(Dataset):
    def __init__(self, file_dir='./audio', query_fold=5, n_mels=128, window_length=1700, hop_size=425, cache_dir="cache"):
        """
        初始化音频数据集，加载音频文件并提取特征。
        
        :param file_dir: 音频文件所在目录。
        :param query_fold: 用于划分查询集的折号（fold）。
        :param n_mels: 梅尔滤波器组的数量。
        :param window_length: 用于特征提取的窗口大小。
        :param hop_size: 用于特征提取的跳跃大小。
        :param cache_dir: 特征缓存目录。
        """
        self.file_dir = file_dir
        self.query_fold = query_fold
        self.n_mels = n_mels
        self.window_length = window_length
        self.hop_size = hop_size
        self.cache_dir = cache_dir
        self.samples = []
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_dataset()

    def _load_dataset(self):
        """
        加载数据集，将每个样本的路径、特征和目标标签存储为元组。
        """
        file_names = os.listdir(self.file_dir)
        for file_name in tqdm(file_names, desc="Loading dataset", dynamic_ncols=True, unit="file"):
            file_path = os.path.join(self.file_dir, file_name)
            fold, target = self._parse_file_name(file_name)
            cache_path = os.path.join(self.cache_dir, f"{file_name}_melspec.pkl")
            
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    features = pickle.load(f)
            else:
                signal, sr = self._load_audio(file_path)
                features = self._extract_features(signal, sr)
                with open(cache_path, "wb") as f:
                    pickle.dump(features, f)
            
            self.samples.append((features, target, fold))

    def _load_audio(self, file_path, sr=44100):
        signal, _ = librosa.load(file_path, sr=sr, mono=True)
        return signal, sr

    def _parse_file_name(self, file_name):
        fold, _, _, target = file_name.split("-")
        target = int(target.split(".")[0])
        return int(fold), target

    def _extract_features(self, signal, sr, target_length=512):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, 
            n_fft=self.window_length, hop_length=self.hop_size,
            n_mels=self.n_mels
        )

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为dB单位
        n_frames = mel_spec.shape[1]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, p))
        elif p < 0:
            mel_spec = mel_spec[:, :target_length]

        return mel_spec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target, fold = self.samples[idx]
        is_query = (fold == self.query_fold)
        # if not is_query:
        #     freq_masked = frequency_masking(features, freq_mask_param=10, num_masks=2)
        #     time_masked = time_masking(freq_masked, time_mask_param=20, num_masks=2)
        #     features = time_masked

        return features, target, is_query


def create_dataloaders(file_dir, query_fold=5, batch_size=48, num_workers=2, **kwargs):
    """
    创建训练集和查询集的DataLoader。
    
    :param file_dir: 音频文件所在目录。
    :param query_fold: 用于划分查询集的折号（fold）。
    :param batch_size: 每批数据的大小。
    :param num_workers: 数据加载的并行线程数。
    :param kwargs: 传递给AudioDataset的其他参数。
    :return: 训练集和查询集的DataLoader。
    """
    dataset = AudioDataset(file_dir, query_fold, **kwargs)

    train_samples = [(features, target) for features, target, is_query in dataset if not is_query]
    query_samples = [(features, target) for features, target, is_query in dataset if is_query]

    train_dataset = torch.utils.data.TensorDataset(
        torch.stack([torch.tensor(item[0]) for item in train_samples]),
        torch.tensor([item[1] for item in train_samples])
    )
    query_dataset = torch.utils.data.TensorDataset(
        torch.stack([torch.tensor(item[0]) for item in query_samples]),
        torch.tensor([item[1] for item in query_samples])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # shape: [batch_size, n_mels, time_frames]
    return train_loader, query_loader

def frequency_masking(mel_spec, freq_mask_param, num_masks=1):
    """
    实现频率掩码（Frequency Masking）。
    
    :param mel_spec: 输入 Mel 频谱图（形状: [n_mels, time_frames]）。
    :param freq_mask_param: 掩码的最大频率范围。
    :param num_masks: 掩码的数量。
    :return: 经过频率掩码的数据。
    """
    mel_spec = mel_spec.copy()  # 避免修改原始频谱
    n_mels, _ = mel_spec.shape

    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask_param)  # 随机生成掩码大小
        f_start = np.random.randint(0, n_mels - f)  # 随机选择起始频率
        mel_spec[f_start:f_start + f, :] = 0  # 将对应频率范围置零

    return mel_spec

def time_masking(mel_spec, time_mask_param, num_masks=1):
    """
    实现时间掩码（Time Masking）。
    
    :param mel_spec: 输入 Mel 频谱图（形状: [n_mels, time_frames]）。
    :param time_mask_param: 掩码的最大时间范围。
    :param num_masks: 掩码的数量。
    :return: 经过时间掩码的数据。
    """
    mel_spec = mel_spec.copy()  # 避免修改原始频谱
    _, time_frames = mel_spec.shape

    for _ in range(num_masks):
        t = np.random.randint(0, time_mask_param)  # 随机生成掩码大小
        t_start = np.random.randint(0, time_frames - t)  # 随机选择起始时间
        mel_spec[:, t_start:t_start + t] = 0  # 将对应时间范围置零

    return mel_spec

