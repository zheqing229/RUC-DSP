import numpy as np
from stft import stft

def hz_to_mel(hz):
    '''
    input:
    hz: float, frequency in Hz or np.ndarray
    output:
    mel: float, frequency in mel or np.ndarray
    '''
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    '''
    input:
    mel: float, frequency in mel or np.ndarray
    output:
    hz: float, frequency in Hz or np.ndarray
    '''
    return 700 * (10 ** (mel / 2595) - 1)

def power_to_db(power):
    '''
    input:
    power: float, power or np.ndarray
    output:
    db: float, decibel or np.ndarray
    '''
    return 10 * np.log10(power+1e-5)

def mfcc(x:np.ndarray, sr:int=22050, n_mfcc:int=20, window_length:int=None, hop_length:int=None, window_type:str='hann')->np.ndarray:
    '''
    input:
    x: 1D array, audio signal
    sr: int, sample rate
    n_mfcc: int, number of mfcc coefficients
    window_length: int, window length in samples
    hop_length: int, hop length in samples
    window_type: str, window type
    output:
    mfcc: 2D array, mfcc coefficients ()
    '''
    S = power_to_db(mel_spectrogram(x, sr, window_length=window_length, hop_length=hop_length, window_type=window_type))
    return dct_manual(S, axis=-2)[..., :n_mfcc, :]

def spectrogram(x:np.ndarray, n_fft:int=2048, sr:int=22050, hop_length:int=None, window_length:int=None, window_type:str='hann')->np.ndarray:
    '''
    input:
    x: 1D array, audio signal
    sr: int, sample rate
    n_fft: int, number of fft points
    hop_length: int, hop length in samples
    window_length: int, window length in samples
    window_type: str, window type
    output:
    melspectrogram: 2D array, melspectrogram ()
    '''
    return np.abs(
        stft(x, n_fft=n_fft, window_length=window_length, hop_length=hop_length, window_type=window_type)
    )

def mel_spectrogram(x:np.ndarray, sr:int=22050, n_fft:int=2048, hop_length:int=None, window_length:int=None, window_type:str='hann', n_mels:int=20, fmin:float=0.0, fmax:float=None)->np.ndarray:
    '''
    input:
    x: 1D array, audio signal
    sr: int, sample rate
    n_fft: int, number of fft points
    hop_length: int, hop length in samples
    window_length: int, window length in samples
    window_type: str, window type
    n_mels: int, number of mel filters
    fmin: float, low frequency in Hz
    fmax: float, high frequency in Hz
    output:
    melspectrogram: 2D array, melspectrogram ()
    '''
    S = spectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, window_length=window_length, window_type=window_type)
    mel_basis = mel_filter(n_fft, n_mels, sr, fmin, fmax)
    return np.dot(mel_basis, S)

def fft_frequencies(n:int=2048, sr:int=22050)->np.ndarray:
    '''
    input:
    n: int, number of fft points
    sr: int, sample rate
    output:
    frequencies: 1D array, frequencies ()
    '''
    d = 1.0 / sr
    val = 1.0/(n*d)
    N = n//2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val

def mel_frequencies(n_mels:int=128, fmin:float=0.0, fmax:float=11025)->np.ndarray:
    '''
    input:
    n_mels: int, number of mel filters
    fmin: float, low frequency in Hz
    fmax: float, high frequency in Hz
    output:
    mel_frequencies: 1D array, mel frequencies ()
    '''
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels)
    return hz


def mel_filter(n_fft:int, n_mels:int=20, sr:int=22050, low_freq:float=0.0, high_freq:float=None)->np.ndarray:
    '''
    input:
    n_fft: int, number of fft points
    n_mels: int, number of mel filters
    sr: int, sample rate
    low_freq: float, low frequency in Hz
    high_freq: float, high frequency in Hz
    output:
    mel_filter: 2D array, mel filterbank ()
    '''
    if high_freq is None:
        high_freq = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=float)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(n=n_fft, sr=sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=low_freq, fmax=high_freq)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    # ramps = mel_f.reshape(mel_f[0], 1) - fft_frequencies.reshape(1, fft_frequencies[0])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights

def dct_manual(x:np.ndarray, axis:int=-1)->np.ndarray:
    '''
    input:
    x: np.ndarray, input signal
    norm: float, normalization factor
    output:
    dct: np.ndarray, dct coefficients
    '''
    x = np.asarray(x)
    N = x.shape[axis]
    n = np.arange(N)
    k = np.arange(N)
    cos_matrix = np.cos(np.pi / N * (n + 0.5)[:, None] * k)
    
    factor = np.sqrt(2 / N) * np.ones(N)
    factor[0] = np.sqrt(1 / N) 

    x_moved = np.moveaxis(x, axis, 0)
    result = np.dot(x_moved.T, cos_matrix) * factor
    return result