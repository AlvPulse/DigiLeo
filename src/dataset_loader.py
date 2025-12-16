# dataset_loader.py
import numpy as np
import librosa
import io
import os
import glob
import random
from datasets import load_dataset, Audio, load_from_disk
from src.preprocessing import handle_long_audio, handle_short_audio
from src.augmentation import augment_audio

def load_raw_dataset(config):
    """
    Step 1: Iterate through ALL configured data sources, stream/load data, 
    clean it, split/pad it. 
    NO AUGMENTATION HERE. Returns pure raw data.
    """
    X_raw = []
    y_raw = []
    
    # Global limits logic is tricky with multiple sources. 
    # Current implementation: limit applies PER SOURCE to ensure mixing.
    # Or should it be total? Let's assume per source for diversity.
    
    for source in config.data_sources:
        print(f"\n📦 Processing Source: {source['path']} ({source['type']})")
        
        counts = {0: 0, 1: 0}
        
        if source['type'] == 'disk_hf':
            # Existing logic for HF Disk dataset
            try:
                ds = load_from_disk(source['path'])
                # No cast_column("audio", Audio(decode=False)) because we want raw bytes or decoded audio
                # If we decode=False, we get bytes. If we decode=True, we get array.
                # Let's try decode=True but be robust.
                # Actually, the original code used decode=False and io.BytesIO.
                # The issue "The least populated classes in y have only 1 member" suggests
                # we are not loading enough data. Maybe min_audio_len is too high?
                # or max_raw_samples is limiting too much?
                
                # Let's stick to the working logic but add better error logging and fallback
                ds = ds.cast_column("audio", Audio(decode=False))
                
                for item in ds:
                    if counts[0] >= config.max_raw_samples and counts[1] >= config.max_raw_samples:
                        break
                        
                    label = item['label']
                    if counts[label] >= config.max_raw_samples: continue
                    
                    try:
                        audio_bytes = item['audio']['bytes']
                        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                        _process_and_add(audio, label, config, X_raw, y_raw, counts)
                    except Exception as e:
                        # print(f"Skipping file due to load error: {e}")
                        continue
                        
            except Exception as e:
                print(f"   ❌ Failed to load disk_hf source: {e}")

        elif source['type'] == 'folder':
            # Logic for "Binary Wav ds" (folders yes/no)
            base_path = source['path']
            # Map folder names to labels
            # Assuming 'yes' = 1 (Event), 'no' = 0 (Background)
            label_map = {'yes': 1, 'no': 0}
            
            for folder_name, label in label_map.items():
                folder_path = os.path.join(base_path, folder_name)
                files = glob.glob(os.path.join(folder_path, "*.wav"))
                print(f"   Found {len(files)} files in {folder_name}...")
                
                random.shuffle(files) # Shuffle to get random sample if we hit limit
                
                for fpath in files:
                    if counts[label] >= config.max_raw_samples: break
                    
                    try:
                        audio, _ = librosa.load(fpath, sr=16000)
                        _process_and_add(audio, label, config, X_raw, y_raw, counts)
                    except Exception as e:
                        print(f"   Error reading {fpath}: {e}")

        print(f"   Collected from this source: C0={counts[0]} | C1={counts[1]}")

    print(f"\n✅ Total Raw Load Complete. Total Chunks: {len(X_raw)}")
    return np.array(X_raw), np.array(y_raw)

def _process_and_add(audio, label, config, X_list, y_list, counts):
    """Helper to apply preprocessing and add to list"""
    # 1. Drop Garbage
    if len(audio) < config.min_audio_len: return
    
    # 2. Normalize
    if config.normalize_audio:
        audio = librosa.util.normalize(audio)
    
    # 3. Handle Long Files
    chunks = handle_long_audio(audio, config.window_samples, config.long_file_strategy)
    
    added_any = False
    for chunk in chunks:
        # 4. Handle Short Files
        processed = handle_short_audio(chunk, config.window_samples, config.short_file_strategy)
        
        if processed is not None:
            X_list.append(processed)
            y_list.append(label)
            added_any = True
            
    # Only count the ORIGINAL file towards the limit if we used at least one chunk
    if added_any:
        counts[label] += 1
        # print(f"   Collected: C0={counts[0]} | C1={counts[1]}", end='\r')


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
    
    # Handle edge case where one class might be empty
    if n0 == 0 or n1 == 0:
        print("   ⚠️ One class is empty! Skipping balancing.")
        return X_train, y_train

    X_balanced = []
    y_balanced = []
    
    # Add all Class 0 
    X_balanced.extend(X0)
    y_balanced.extend([0] * n0)
    
    # Add all Class 1
    X_balanced.extend(X1)
    y_balanced.extend([1] * n1)
    
    # Target is the larger count
    target_count = max(n0, n1)
    
    # Balance 0 if needed
    if n0 < target_count:
        needed = target_count - n0
        print(f"   🚀 Generating {needed} augmented samples for Class 0...")
        _augment_to_list(X0, needed, 0, X_balanced, y_balanced,config)

    # Balance 1 if needed
    if n1 < target_count:
        needed = target_count - n1
        print(f"   🚀 Generating {needed} augmented samples for Class 1...")
        _augment_to_list(X1, needed, 1, X_balanced, y_balanced,config)
    
    return np.array(X_balanced), np.array(y_balanced)

def _augment_to_list(reference_samples, needed, label, X_out, y_out,config):
    generated = 0
    n_refs = len(reference_samples)
    while generated < needed:
        idx = random.randint(0, n_refs - 1)
        original_sample = reference_samples[idx]
        new_sample = augment_audio(original_sample,config)
        X_out.append(new_sample)
        y_out.append(label)
        generated += 1
