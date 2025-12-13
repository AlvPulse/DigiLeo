# preprocessing.py
import numpy as np

def handle_long_audio(audio, target_len, strategy="split"):
    """
    Handles audio longer than target_len. Returns a LIST of chunks.
    """
    curr_len = len(audio)
    if curr_len <= target_len:
        return [audio]

    if strategy == "split":
        # Divide into multiple valid chunks
        num_chunks = curr_len // target_len
        chunks = []
        for i in range(num_chunks):
            start = i * target_len
            chunks.append(audio[start : start + target_len])
        return chunks
        
    elif strategy == "trim_start":
        return [audio[:target_len]]
        
    elif strategy == "trim_random":
        start = np.random.randint(0, curr_len - target_len)
        return [audio[start : start + target_len]]
        
    return [audio[:target_len]]

def handle_short_audio(audio, target_len, strategy="loop_pad"):
    """
    Handles audio shorter than target_len. Returns single array or None.
    """
    curr_len = len(audio)
    if curr_len >= target_len:
        return audio[:target_len]

    if strategy == "drop":
        return None
        
    elif strategy == "zero_pad":
        pad_width = target_len - curr_len
        return np.pad(audio, (0, pad_width), mode='constant')
        
    elif strategy == "loop_pad":
        repeats = int(np.ceil(target_len / curr_len))
        tiled = np.tile(audio, repeats)
        # Add tiny noise to avoid robotic looping artifacts
        tiled = tiled + np.random.normal(0, 0.0001, len(tiled))
        return tiled[:target_len]
    
    return None