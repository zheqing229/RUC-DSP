import ctypes
import numpy as np

# python
import numpy
from fft import fft
from stft import stft
from mfcc import power_to_db

def preemphasis(audio, coef=0.97):
    return audio[1:] - coef * audio[:-1]

def filter_bank(spec, n_mel, sample_rate=16000):
    max_freq = sample_rate / 2
    N = spec.shape[1]
    interval = max_freq / (N - 1)
    max_mel_freq = 2595 * np.log10(1 + max_freq / 700)
    mel_points = np.linspace(0, max_mel_freq, n_mel + 2)
    mel_points_freq = 700 * (10 ** (mel_points / 2595) - 1) / interval
    mel_filter = np.zeros((n_mel, N))
    for i in range(n_mel):
        ref1 = int(np.ceil(mel_points_freq[i]))
        ref2 = int(np.ceil(mel_points_freq[i + 1]))
        ref3 = int(np.ceil(mel_points_freq[i + 2]))
        for j in range(ref1, ref2):
            mel_filter[i, j] = (j - mel_points_freq[i]) / (mel_points_freq[i + 1] - mel_points_freq[i])
        for j in range(ref2, ref3):
            mel_filter[i, j] = (mel_points_freq[i + 2] - j) / (mel_points_freq[i + 2] - mel_points_freq[i + 1])
    mel_spectrum = spec @ mel_filter.T
    return mel_spectrum

def max_bins(N, n_fft=512, hop_length=128, **kwargs):
    return (N - n_fft - 1) // hop_length + 2


def dsp_pipline(audio, sample_rate=16000, n_fft=512, hop_length=128, n_mel=128, preemphasis_coef=0.97, **kwargs):
    audio = preemphasis(audio, coef=preemphasis_coef)
    power_spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
    mel_spec = filter_bank(power_spec, n_mel=n_mel, sample_rate=sample_rate)
    mel_spec = power_to_db(mel_spec)
    return mel_spec