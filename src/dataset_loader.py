# dataset_loader.py
import numpy as np
import librosa
import io
import random
from datasets import load_dataset, Audio,load_from_disk
from src.preprocessing import handle_long_audio, handle_short_audio
from src.augmentation import augment_audio

def load_raw_dataset(config):
    """
    Step 1: Stream data, clean it, split/pad it. 
    NO AUGMENTATION HERE. Returns pure raw data.
    """
    print(f"📦 Loading Raw Data (Limit: {config.max_raw_samples}/class)...")
    if "hf" in config.dataset_path:
        ds = load_dataset(config.dataset_path, split="train", streaming=True)
    else:
        ds= load_from_disk(config.dataset_path)
    # Force raw bytes to avoid decoder errors
    ds = ds.cast_column("audio", Audio(decode=False))
    
    X_raw = []
    y_raw = []
    counts = {0: 0, 1: 0}
    
    for item in ds:
        # Stop condition
        if counts[0] >= config.max_raw_samples and counts[1] >= config.max_raw_samples:
            break
            
        label = item['label']
        if counts[label] >= config.max_raw_samples: continue
        
        try:
            # 1. Decode
            audio_bytes = item['audio']['bytes']
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            
            # 2. Drop Garbage
            if len(audio) < config.min_audio_len: continue
            
            # 3. Normalize
            if config.normalize_audio:
                audio = librosa.util.normalize(audio)
            
            # 4. Handle Long Files (Returns List)
            chunks = handle_long_audio(audio, config.window_samples, config.long_file_strategy)
            
            for chunk in chunks:
                # 5. Handle Short Files (Returns Array or None)
                processed = handle_short_audio(chunk, config.window_samples, config.short_file_strategy)
                
                if processed is not None:
                    X_raw.append(processed)
                    y_raw.append(label)
            
            # Only count the ORIGINAL file towards the limit
            counts[label] += 1
            print(f"   Collected: C0={counts[0]} | C1={counts[1]}", end='\r')
            
        except Exception:
            continue
            
    print(f"\n✅ Raw Load Complete. Total Chunks: {len(X_raw)}")
    return np.array(X_raw), np.array(y_raw)

def balance_training_set(X_train, y_train, config):
    """
    Step 2: Takes the TRAINING set, identifies the minority class,
    and augments it until it equals the majority class.
    """
    if not config.balance_classes:
        return X_train, y_train
        
    print("\n⚖️ Balancing Training Set...")
    
    # Separate
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    
    n0, n1 = len(X0), len(X1)
    print(f"   Original Train Counts -> Class 0: {n0}, Class 1: {n1}")
    
    X_balanced = []
    y_balanced = []
    
    # Add all Class 0 (Assuming it's majority)
    X_balanced.extend(X0)
    y_balanced.extend([0] * n0)
    
    # Add Original Class 1
    X_balanced.extend(X1)
    y_balanced.extend([1] * n1)
    
    # Calculate deficit
    target_count = n0
    current_count = n1
    needed = target_count - current_count
    
    if abs(needed) > 0:
        print(f"   🚀 Generating {needed} augmented samples for Class 1...")
        generated = 0
        if(needed>0):
            reference= X1
            SampleNum=n1
        else:
            reference= X0
            SampleNum= n0
        
        while generated < abs(needed):
            # Randomly pick a Class 1 sample to clone
            idx = random.randint(0, SampleNum - 1)
            original_sample = reference[idx]
            
            # Create new version
            new_sample = augment_audio(original_sample)
            #print("original_sample",original_sample.size, "new_sample", new_sample.size)
            X_balanced.append(new_sample)
            y_balanced.append(1)
            generated += 1
    
    return np.array(X_balanced), np.array(y_balanced)

    