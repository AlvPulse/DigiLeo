# features.py
import librosa
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def extract_features(X, config):
    """
    Extracts features from raw audio.
    
    Args:
        X (list or np.array): List of raw audio arrays (1D).
        config (ExperimentConfig): Configuration object.
        
    Returns:
        np.array:
            - If return_2d_features is True: (n_samples, n_features, n_timesteps)
              or (n_samples, 1, n_features, n_timesteps) for PyTorch CNNs?
              Typically CNNs want (N, C, H, W).
              Here we likely return (N, n_features, n_timesteps) and let the model reshape/unsqueeze.
            - If return_2d_features is False (1D): (n_samples, enriched_feature_vector_size)
    """
    feats = []
    
    # Pre-calculate params
    is_mfcc = (config.feature_type == 'mfcc')
    return_2d = getattr(config, 'return_2d_features', False)
    
    for x in X:
        if config.normalize_audio:
            x = librosa.util.normalize(x)
            
        if is_mfcc:
            # --- MFCC ---
            ft = librosa.feature.mfcc(y=x, sr=config.sr, n_mfcc=config.n_mfcc)
            if config.drop_first_coeff:
                ft = ft[1:]
        else:
            # --- MEL SPECTROGRAM ---
            ft = librosa.feature.melspectrogram(
                y=x, sr=config.sr, 
                n_mels=config.n_mels, 
                fmax=config.fmax
            )
            ft = librosa.power_to_db(ft, ref=np.max)
            
        # ft shape: (n_features, n_timesteps)

        if return_2d:
            # For CNN: Return the full matrix
            # We assume constant time dimension (handled by padding/trimming in dataset_loader)
            feats.append(ft)
        else:
            # For 1D Models: Feature Enrichment
            # Flattened stats for each frequency bin across time

            # 1. Mean (Original)
            mu = np.mean(ft, axis=1)

            # 2. Std Dev
            sigma = np.std(ft, axis=1)

            # 3. Min/Max
            min_val = np.min(ft, axis=1)
            max_val = np.max(ft, axis=1)

            # 4. Skewness & Kurtosis
            sk = skew(ft, axis=1)
            ku = kurtosis(ft, axis=1)

            # 5. Entropy (Shannon) - normalize to make it probability-like first?
            # Creating a distribution over time for each band
            # Softmax or simple normalization
            # Let's use simple probability distribution over time
            # Add small epsilon to avoid log(0)
            ft_norm = np.abs(ft)
            ft_prob = ft_norm / (np.sum(ft_norm, axis=1, keepdims=True) + 1e-8)
            ent = entropy(ft_prob, axis=1)

            # 6. Deltas (Rate of change)
            # Delta over time (axis 1)
            dt = librosa.feature.delta(ft)
            dt_mu = np.mean(dt, axis=1)

            # Delta-Delta
            ddt = librosa.feature.delta(ft, order=2)
            ddt_mu = np.mean(ddt, axis=1)

            # Concatenate all features
            # Shape per sample: (n_features * 8, )
            combined = np.concatenate([mu, sigma, min_val, max_val, sk, ku, ent, dt_mu, ddt_mu])
            feats.append(combined)
        
    return np.array(feats)
