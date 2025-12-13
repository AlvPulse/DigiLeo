# features.py
import librosa
import numpy as np

def extract_mfcc(audio, sr, config):
    # 1. Normalize (Crucial for Volume Invariance)
    if config.normalize_audio:
        audio = librosa.util.normalize(audio)
        
    # 2. Compute
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.n_mfcc)
    
    # 3. Drop First Coeff (Crucial for "Loudness Invariance")
    if config.drop_first_coeff:
        mfcc = mfcc[1:]
        
    # 4. Global Average Pooling
    return np.mean(mfcc.T, axis=0)