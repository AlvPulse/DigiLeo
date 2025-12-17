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
    
    for source in config.data_sources:
        print(f"\n📦 Processing Source: {source['path']} ({source['type']})")
        
        # Counts now refer to CHUNKS, not files
        counts = {0: 0, 1: 0}
        
        if source['type'] == 'disk_hf':
            try:
                ds = load_from_disk(source['path'])
                ds = ds.cast_column("audio", Audio(decode=False))
                
                # Convert to list and shuffle to ensure random file access
                # CAUTION: If dataset is huge, this list(range) is fine, but don't list(ds)
                indices = list(range(len(ds)))
                random.shuffle(indices)

                for idx in indices:
                    # Check if we have enough chunks for both classes
                    if counts[0] >= config.max_raw_samples and counts[1] >= config.max_raw_samples:
                        break
                        
                    item = ds[idx]
                    label = item['label']

                    # Optimization: Don't even load audio if this class is full
                    if counts[label] >= config.max_raw_samples: continue
                    
                    try:
                        audio_bytes = item['audio']['bytes']
                        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

                        # Process and get chunks
                        new_chunks = _process_and_get_chunks(audio, config)

                        # Sub-sample chunks from this file to avoid "one file dominates"
                        # e.g. Max 20 chunks per file
                        if (len(new_chunks) > config.max_chunks_per_file and label ==0 ):
                            new_chunks = random.sample(new_chunks, config.max_chunks_per_file)

                        # Add to lists
                        for chunk in new_chunks:
                            if counts[label] >= config.max_raw_samples: break
                            X_raw.append(chunk)
                            y_raw.append(label)
                            counts[label] += 1

                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"   ❌ Failed to load disk_hf source: {e}")

        elif source['type'] == 'folder':
            base_path = source['path']
            label_map = {'yes': 1, 'no': 0}
            
            for folder_name, label in label_map.items():
                folder_path = os.path.join(base_path, folder_name)
                files = glob.glob(os.path.join(folder_path, "*.wav"))
                print(f"   Found {len(files)} files in {folder_name}...")
                
                random.shuffle(files) # Random file order
                
                for fpath in files:
                    if counts[label] >= config.max_raw_samples: break
                    
                    try:
                        audio, _ = librosa.load(fpath, sr=16000)

                        new_chunks = _process_and_get_chunks(audio, config)

                        if (len(new_chunks) > config.max_chunks_per_file and label==0):
                            new_chunks = random.sample(new_chunks, config.max_chunks_per_file)

                        for chunk in new_chunks:
                            if counts[label] >= config.max_raw_samples: break
                            X_raw.append(chunk)
                            y_raw.append(label)
                            counts[label] += 1

                    except Exception as e:
                        print(f"   Error reading {fpath}: {e}")

        print(f"   Collected from this source: C0={counts[0]} | C1={counts[1]}")

    print(f"\n✅ Total Raw Load Complete. Total Chunks: {len(X_raw)}")
    return np.array(X_raw), np.array(y_raw)

def _process_and_get_chunks(audio, config):
    """
    Helper to apply preprocessing and return a list of valid chunks.
    Does NOT modify global lists directly.
    """
    valid_chunks = []

    # 1. Drop Garbage
    if len(audio) < config.min_audio_len: return []
    
    # 2. Normalize
    if config.normalize_audio:
        audio = librosa.util.normalize(audio)
    
    # 3. Handle Long Files (Splitting)
    raw_chunks = handle_long_audio(audio, config.window_samples, config.long_file_strategy)
    
    for chunk in raw_chunks:
        # 4. Handle Short Files (Padding)
        processed = handle_short_audio(chunk, config.window_samples, config.short_file_strategy)
        if processed is not None:
            valid_chunks.append(processed)
            
    return valid_chunks


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
