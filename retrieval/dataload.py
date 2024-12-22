import os    
import librosa
import numpy as np
import pickle
from tqdm import tqdm
from fft import fft
from stft import stft
from mfcc import mfcc
def load_audio(file_path, sr=44100):
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr

def parse_file_name(file_name):
    '''
    get fold and target class from filename
    '''
    fold, _, _, target = file_name.split("-")
    target = int(target.split(".")[0]) 
    return int(fold), target

def extract_features(signal, sr, n_mfcc=20, window_length=2048, hop_size=512):
    mfcc_features = mfcc(x=signal, sr=sr, n_mfcc=n_mfcc, window_length=window_length, hop_length=hop_size)
    # mfcc_features = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc_features

    
def process_file(file_name, file_dir, query_fold, n_mfcc= 20, window_length=2048, hop_size=512, cache_dir="cache"):
    
    file_path = os.path.join(file_dir, file_name)
    fold, target = parse_file_name(file_name)
    cache_path = os.path.join(cache_dir, f"{file_name}_mfcc.pkl")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            features = pickle.load(f)
    else:
        signal, sr = load_audio(file_path)
        features = extract_features(signal, sr, n_mfcc=20, window_length=window_length, hop_size=hop_size)
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
            
    
    return (features, target, fold)

def load_dataset(file_dir, query_fold=5, n_mfcc = 20,window_length=2048, hop_size=512, cache_dir="cache"):
    '''
    Loads the dataset and divides it into database and query set.

    file_dir: Path to the directory containing the audio files.
    query_fold: The fold to be used for querying, default is 5.
    n_mfcc: The number of MFCC features to extract, default is 20.
    window_length: The window size used for feature extraction, default is 2048.
    hop_size: The hop size (frame shift) used for feature extraction, default is 512.
    cache_dir: The directory to store the cached feature files, default is "cache".

    Returns:
    - database: A list containing database samples, where each element is a tuple (features, target label).
    - query_set: A list containing query set samples, where each element is a tuple (features, target label).
    '''
    os.makedirs(cache_dir, exist_ok=True)  
    database, query_set = [], []
    
    file_names = os.listdir(file_dir)
    for file_name in tqdm(file_names, desc="Loading dataset", dynamic_ncols=True, unit="file"):
        features, target, fold = process_file(file_name, file_dir, query_fold, n_mfcc=n_mfcc,window_length=window_length, hop_size=hop_size, cache_dir=cache_dir)
        if fold == query_fold:
            query_set.append((features, target))
        else:
            database.append((features, target))
    return database, query_set