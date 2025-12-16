# augmentation.py
import numpy as np
import librosa
import random
import glob
import os

# Cache background files to avoid disk lag
BACKGROUND_FILES = []

def load_background_files(folder_path):
    global BACKGROUND_FILES
    if not BACKGROUND_FILES:
        print(f"📂 Caching background noise from {folder_path}...")
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        BACKGROUND_FILES = files
    return BACKGROUND_FILES

def mix_background(signal, sr, noise_path, min_snr=0, max_snr=15):
    """
    Overlays a random background noise file onto the signal 
    at a random Signal-to-Noise Ratio (SNR).
    """
    try:
        # 1. Load a random noise file
        bg_files = load_background_files(noise_path)
        if not bg_files: return signal
        
        noise_file = random.choice(bg_files)
        noise, _ = librosa.load(noise_file, sr=sr)
        
        # 2. Loop/Crop noise to match signal length
        if len(noise) < len(signal):
            repeats = int(np.ceil(len(signal) / len(noise)))
            noise = np.tile(noise, repeats)[:len(signal)]
        else:
            start = random.randint(0, len(noise) - len(signal))
            noise = noise[start : start + len(signal)]
            
        # 3. Calculate RMS Energy (Volume)
        signal_rms = np.sqrt(np.mean(signal**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        
        if noise_rms == 0: return signal
        
        # 4. Pick a random SNR (Signal-to-Noise Ratio)
        snr_db = np.random.uniform(min_snr, max_snr)
        snr_linear = 10 ** (snr_db / 20)
        
        # 5. Scale noise to achieve target SNR
        # Target Noise RMS = Signal RMS / SNR
        target_noise_rms = signal_rms / snr_linear
        scaled_noise = noise * (target_noise_rms / noise_rms)
        
        # 6. Mix
        return signal + scaled_noise
        
    except Exception:
        return signal # Fail safe: return original

def add_pink_noise(data, noise_factor=0.005):
    """
    Pink noise (1/f) is more realistic for environmental background 
    than standard white (Gaussian) noise.
    """
    # Generate white noise
    white = np.random.randn(len(data))
    # FFT to frequency domain
    X_white = np.fft.rfft(white) / len(white)
    # Apply 1/f filter
    S = np.sqrt(np.arange(X_white.size) + 1.) # +1 to avoid div by zero
    X_pink = X_white / S
    # IFFT back to time domain
    pink = np.fft.irfft(X_pink)
    # Normalize pink noise to match signal scale
    pink = pink * (np.max(np.abs(data)) / (np.max(np.abs(pink)) + 1e-9))
    return data + (pink * noise_factor)

def augment_audio(audio,config, sr=16000,Target_length = 8000):
    """
    Returns a SINGLE augmented version of the input audio.
    Guarantees exact output length.
    """
    
    y = audio.copy()
    
    # Randomly choose ONE effect (Expanded List)
    choice = random.choice(['pitch', 'speed', 'white_noise', 'pink_noise', 'gain', 'polarity','mix_bg'])
    
    try:
        if choice == 'mix_bg' and config.use_background_noise:
            y = mix_background(y, sr, config.background_noise_path, 
                            config.min_snr_db, config.max_snr_db)
        if choice == 'pitch':
            # Pitch Shift: Changes tone without changing speed
            steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            
        elif choice == 'speed':
            # Time Stretch: Changes speed without changing tone (changes length!)
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            
        elif choice == 'white_noise':
            # Electronic Hiss
            noise_amp = np.random.uniform(0.001, 0.01)
            y = y + np.random.normal(0, noise_amp, len(y))
            
        elif choice == 'pink_noise':
            # Natural/Wind Noise
            noise_amp = np.random.uniform(0.001, 0.01)
            y = add_pink_noise(y, noise_factor=noise_amp)
            
        elif choice == 'gain':
            # Volume variation (Distance simulation)
            y = y * np.random.uniform(0.5, 1.5)
            
        elif choice == 'time_shift':
            # Roll the audio (circular shift) so the event isn't always in the center
            shift_amount = np.random.randint(0, len(y))
            y = np.roll(y, shift_amount)
            
        elif choice == 'polarity':
            # Invert Phase (Simple but effective data variation)
            y = -y

    except Exception as e:
        # Fallback if an augmentation fails math (rare)
        pass

    # --- THE FIX: STRICT LENGTH ENFORCEMENT ---
    # This replaces manual padding and guarantees 16000 samples.
    # librosa.util.fix_length centers the audio if padding, or crops if too long.
    y = librosa.util.fix_length(y, size=Target_length)
        
    return y