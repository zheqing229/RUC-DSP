import os
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import librosa

from torch.utils.data import Dataset, DataLoader
from itertools import chain

from .process import dsp_pipline, max_bins


def random_clip(audio):
    N = audio.shape[0]
    audio, _ = librosa.effects.trim(audio)
    N2 = audio.shape[0]
    clip_result = np.zeros(N + N2)
    start = np.random.randint(N)
    clip_result[start:start + N2] = audio
    return clip_result[N2 // 2: N + N2 // 2]


# cut N pieces, averaged at the end for testing
def cut_all(spec, origin_N, pieces=10):
    N, D = spec.shape
    start = np.linspace(0, origin_N, pieces).astype(np.int32)
    result = []
    for s in start:
        clip_option = np.ones((origin_N + N, D)) * (-80 + 55.141) / 18.852
        clip_option[s:s + N] = spec
        result.append(clip_option[N // 2: origin_N + N // 2])
    return result


# randomly mix two sound, scaled according to power
def mix(audio1, audio2):
    if len(audio2) > len(audio1):
        audio1, audio2 = audio2, audio1
    if len(audio2) != len(audio1):
        N1, N2 = len(audio1), len(audio2)
        pad_front = np.random.randint(N1 - N2)
        audio2 = np.concatenate([np.zeros(pad_front), audio2, np.zeros(N1 - N2 - pad_front)])

    power_ratio = (np.sum(audio1 ** 2) / np.sum(audio2 ** 2)) ** 0.5
    r = np.random.uniform()
    p = 1 / (1 + power_ratio * (1 - r) / r)
    mix_audio = (p * audio1 + (1 - p) * audio2) / (p ** 2 + (1 - p) ** 2) ** 0.5
    return mix_audio, r

class ESC50Dataset(Dataset):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        self.train_mix = self.config.get("train_mix",0)
        self.dataset_path = self.config.get("dataset_path")
        self.dataset_type = self.config.get("dataset_type")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.device = self.config.get("device", "cpu")

        self.meta = pd.read_csv(os.path.join(self.dataset_path, "meta", "esc50.csv"))
        # names of the sounds
        self.labels = self.meta.groupby('target')["category"].sample(1).tolist()
        
        if self.dataset_type == "train":
            self.meta = self.meta[self.meta.fold != 5]
        elif self.dataset_type == "test":
            self.meta = self.meta[self.meta.fold == 5]
        else:
            raise ValueError

        # because the dataset is small, preload them into ram
        self.targets = self.meta["target"].tolist()
        self.raw_audio = []
        for file in tqdm(self.meta["filename"]):
            signal,_ = librosa.load(os.path.join(self.dataset_path, "audio", file), sr=self.sample_rate)
            self.raw_audio.append(signal)
        self.raw_audio = np.stack(self.raw_audio)
        print(f"Done initializing {self.dataset_type} dataset.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):

        if self.dataset_type == "test":
            test_cut_pieces = self.config.get("test_cut_pieces", 10)
            signal = self.raw_audio[index]
            N = signal.shape[0]
            signal,_ = librosa.effects.trim(y=signal) # remove silence
            # pad audio with zero
            signal = np.concatenate([np.zeros(N//2), signal, np.zeros(N//2)])
            spec = dsp_pipline(signal, **self.config)  # dsp
            spec = normalize(spec)
            specs = cut_all(spec, max_bins(N, **self.config), test_cut_pieces)
            specs = [torch.tensor(spec, device=self.device) for spec in specs] 
            target = torch.scalar_tensor(self.targets[index], device=self.device)
            return specs, target
        else:
            index2 = np.random.randint(len(self))
            if  self.train_mix == 0:
                index2 = index
            signal1 = self.raw_audio[index]
            signal2 = self.raw_audio[index2]
            # remove silence, and random clip
            signal1 = random_clip(signal1)
            signal2 = random_clip(signal2)
            mix_audio, r = mix(signal1, signal2)
            #dsp
            spec = dsp_pipline(mix_audio, **self.config)
            spec = normalize(spec)
            target = torch.zeros(50, dtype=torch.float32, device=self.device)
            target[self.targets[index]] += r
            target[self.targets[index2]] += 1 - r
            spec = torch.tensor(spec, device=self.device)

        return spec, target
    
class ESC50Dataset1(Dataset):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        self.n_cut = self.config.get("test_cut_pieces",10)
        self.train_mix = self.config.get("train_mix",0)
        self.dataset_path = self.config.get("dataset_path")
        self.dataset_type = self.config.get("dataset_type")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.device = self.config.get("device", "cpu")

        self.meta = pd.read_csv(os.path.join(self.dataset_path, "meta", "esc50.csv"))
        # names of the sounds
        self.labels = self.meta.groupby('target')["category"].sample(1).tolist()
        
        if self.dataset_type == "train":
            self.meta = self.meta[self.meta.fold != 5]
        elif self.dataset_type == "test":
            self.meta = self.meta[self.meta.fold == 5]
        else:
            raise ValueError

        # because the dataset is small, preload them into ram
        self.targets = self.meta["target"].tolist()
        self.raw_audio = []
        for file in tqdm(self.meta["filename"]):
            signal,_ = librosa.load(os.path.join(self.dataset_path, "audio", file), sr=self.sample_rate)
            self.raw_audio.append(signal)
        self.raw_audio = np.stack(self.raw_audio)
        print(f"Done initializing {self.dataset_type} dataset.")

    def __len__(self):
        return len(self.meta)
    
    def process(self,index,index2):
        signal1 = self.raw_audio[index]
        signal2 = self.raw_audio[index2]
        # remove silence, and random clip
        signal1 = random_clip(signal1)
        signal2 = random_clip(signal2)
        mix_audio, r = mix(signal1, signal2)
        #dsp
        spec = dsp_pipline(mix_audio, **self.config)
        spec = normalize(spec)
        target = torch.zeros(50, dtype=torch.float32, device=self.device)
        target[self.targets[index]] += r
        target[self.targets[index2]] += 1 - r
        return spec, target
    
    def __getitem__(self, index):

        index2 = np.random.randint(len(self))
        if self.dataset_type == "test" or self.train_mix==0:
            index2 = index
        
        spec,target = self.process(index,index2)
        if self.dataset_type != "test":
            spec = torch.tensor(spec, device=self.device)
        if self.train_mix==0:
            target = torch.tensor(self.targets[index], device=self.device)
        if self.dataset_type == "test" and self.n_cut>=1:
            specs = [spec]
            for _ in range(self.n_cut-1):
                spec,_ = self.process(index,index2)
                specs.append(spec)
            specs = [torch.tensor(spec, device=self.device) for spec in specs]
            spec = specs
        return spec, target
    
# calculated from the dataset
def normalize(spec):
    std = 18.852
    mean = -55.141
    return (spec - mean) / std

def build_dataset(config, select=0):

    def train_collate(batch_input):
        specs, labels = list(zip(*batch_input))
        specs = torch.stack(specs, dim=0)
        labels = torch.stack(labels, dim=0)
        return specs, labels

    def test_collate(batch_input):
        specs, labels = list(zip(*batch_input))
        specs = list(chain(*specs))
        specs = torch.stack(specs, dim=0)
        labels = torch.stack(labels, dim=0)
        return specs, labels

    batch_size = config.get("batch_size")
    if select == 0:
        dataset = ESC50Dataset(config)
    else:
        dataset = ESC50Dataset1(config)
    if dataset.dataset_type == "train":
        return DataLoader(dataset, batch_size=batch_size, collate_fn=train_collate, shuffle=True)
    else :
        return DataLoader(dataset, batch_size=batch_size, collate_fn=test_collate)
    

