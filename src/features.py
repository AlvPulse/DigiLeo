# features.py
import librosa
import numpy as np

def extract_features(X, config):
    """
    Generic feature extractor that switches between MFCC and Mel-spectrogram
    based on config.feature_type.
    
    Args:
        X (list or np.array): List of raw audio arrays (1D).
        config (ExperimentConfig): Configuration object.
        
    Returns:
        np.array: 2D array of features (n_samples, n_features).
    """
    #print(f"   Extracting Features ({config.feature_type})...")
    feats = []
    
    # Pre-calculate params to avoid lookup in loop
    is_mfcc = (config.feature_type == 'mfcc')
    #sr = 16000 # Hardcoded for consistency across project
    
    for x in X:
        # Re-normalize before feature extraction (safe practice)
        if config.normalize_audio:
            x = librosa.util.normalize(x)
            
        if is_mfcc:
            # --- MFCC ---
            ft = librosa.feature.mfcc(y=x, sr=config.sr, n_mfcc=config.n_mfcc)
            if config.drop_first_coeff:
                ft = ft[1:]
        else:
            # --- MEL SPECTROGRAM ---
            # Power spectrogram (amplitude squared)
            ft = librosa.feature.melspectrogram(
                y=x, sr=config.sr, 
                n_mels=config.n_mels, 
                fmax=config.fmax
            )
            # Convert to log scale (dB) - standard for audio DL/ML
            ft = librosa.power_to_db(ft, ref=np.max)
            
        # Global Average Pooling (Time averaging)
        # Result shape: (n_features,)
        feats.append(np.mean(ft.T, axis=0))
        
    return np.array(feats)
