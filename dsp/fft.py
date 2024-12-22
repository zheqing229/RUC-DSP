import numpy as np
import cmath

def _fft(x:np.ndarray, n:int=None, is_forward:bool=True) -> np.ndarray:
    '''
    _fft is the function to calculate both fft and ifft.

    input:
    x: input array
    n: length of the transformed axis
    is_forward: if True, compute forward FFT, otherwise compute inverse FFT

    output:
    y: output array
    '''
    if x.ndim != 1:
        raise ValueError("The input array must be a one-dimensional array.")

    x = np.asarray(x, dtype=complex)

    if n is None:
        n = len(x)
    else:
        if n < len(x):
            x = x[:n]
        elif n > len(x):
            x = np.pad(x, (0, n - len(x)), 'constant')

    x = pad_to_power_of_two(x)
    n = len(x)

    if not is_forward:
        x = np.conj(x)

    bit_reversed_indices, bit_len = bit_reverse_indices(n)
    x = x[bit_reversed_indices]

    # Cooley-Tukey FFT 
    for s in range(1, bit_len + 1):
        m = 2 ** s
        m_half = m // 2
        twiddle_factor = np.exp(-2j * cmath.pi / m)
        for k in range(0, n, m):
            for j in range(m_half):
                t = twiddle_factor ** j * x[k + j + m_half]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + m_half] = u - t

    if not is_forward:
        x = np.conj(x) / n
        x = x.real
    return x

def bit_reverse_indices(n:int) -> np.ndarray:
    '''
    Calculate the bit-reversed indices of an integer.

    input:
    n: integer

    output:
    bit_reversed_indices: bit-reversed indices
    '''
    bit_len = int(np.log2(n))
    indices = np.arange(n)
    bit_reversed = np.array([int('{:0{width}b}'.format(i, width=bit_len)[::-1], 2) for i in indices])
    return bit_reversed, bit_len

def pad_to_power_of_two(x:np.ndarray) -> np.ndarray:
    '''
    pad_to_power_of_two is the function to pad the input array to the next power of 2.

    input:
    x: input array

    output:
    x: output array
    '''
    N = len(x)
    if (N & (N - 1)) == 0:
        return x

    next_pow_two = 1 << (N - 1).bit_length()
    
    padded_x = np.pad(x, (0, next_pow_two - N), mode='constant')

    return padded_x

def fft(x:np.ndarray, n:int=None) -> np.ndarray:
    '''
    fft is the function to calculate the forward FFT.

    input:
    x: input array
    n: length of the transformed axis

    output:
    y: output array
    '''
    return _fft(x, n, is_forward=True)

def ifft(x:np.ndarray, n:int=None) -> np.ndarray:
    '''
    ifft is the function to calculate the inverse FFT.

    input:
    x: input array
    n: length of the transformed axis

    output:
    y: output array
    '''
    return _fft(x, n, is_forward=False)

