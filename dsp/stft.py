import numpy as np
from fft import fft

def stft(x: np.ndarray, n_fft:int=2048, window_length: int = None, hop_length: int = None, window_type: str = 'hann') -> np.ndarray:
    '''
    Compute the Short-Time Fourier Transform (STFT) of the input signal.

    input:
        x: np.ndarray, shape (seq_len,) or (N, seq_len)
            Input audio signal(s).
        n_fft: int
            Number of FFT points. Default is 2048.
        window_length: int
            Length of the window function to apply. Default is the n_fft.
        hop_length: int
            Step size between adjacent windows. Default is window_length // 4.
        window_type: str
            Type of window function to use ("hann", "hamming", "blackman", "rectangular").

    output:
        y: np.ndarray, shape (N, n_fft // 2 + 1, n_frames)
            STFT result.
    '''
    # Ensure 2D input
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    N, seq_len = x.shape

    # Set default window and hop lengths
    if window_length is None:
        window_length = n_fft
    if hop_length is None:
        hop_length = int(window_length // 4)  # Default to 75% overlap

    # Padding the signal to handle edge cases
    padding_size = (n_fft - (seq_len % n_fft))
    padding_size = max(0, padding_size)
    x_padded = np.pad(x, ((0, 0), (0, padding_size)), mode='constant')

    # Determine the number of frames
    num_frames = int(1 + np.ceil((x_padded.shape[1] - window_length) // hop_length))
    
    if window_type == 'hann':
        window = np.hanning(window_length)
    elif window_type == 'hamming':
        window = np.hamming(window_length)
    elif window_type == 'blackman':
        window = np.blackman(window_length)
    elif window_type == 'rectangular':
        window = np.ones(window_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Perform STFT
    stft_result = []
    for i in range(N):  # Iterate over batch dimension
        frames = []
        for j in range(num_frames):
            start_idx = j * hop_length
            end_idx = start_idx + window_length
            if end_idx > x_padded.shape[1]:
                frame = np.zeros(window_length)  # Padding if the window goes beyond signal length
                frame[:x_padded[i, start_idx:].shape[0]] = x_padded[i, start_idx:] * window[:x_padded[i, start_idx:].shape[0]]
            else:
                frame = x_padded[i, start_idx:end_idx] * window  # Apply window function
            frame_fft = fft(frame, n=n_fft)  # Compute FFT (real-to-complex)
            frames.append(frame_fft[:n_fft // 2 + 1])
        stft_result.append(np.array(frames))

    
    stft_result = np.array(stft_result)
    stft_result = np.transpose(stft_result, axes=(0, 2, 1)) # Shape (N, n_fft // 2 + 1, n_frames)
    return stft_result
