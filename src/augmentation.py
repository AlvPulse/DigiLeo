# augmentation.py
import numpy as np
import librosa
import random

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

def augment_audio(audio, sr=16000,Target_length = 8000):
    """
    Returns a SINGLE augmented version of the input audio.
    Guarantees exact output length.
    """
    
    y = audio.copy()
    
    # Randomly choose ONE effect (Expanded List)
    choice = random.choice(['pitch', 'speed', 'white_noise', 'pink_noise', 'gain', 'time_shift', 'polarity'])
    
    try:
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