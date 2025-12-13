# config.py
# config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ExperimentConfig:
    # --- 1. DATASET CONTROL ---
    dataset_path: str = "./my_offline_ds"
    max_raw_samples: int = 3000     # <--- YOU CONTROL THIS
    min_audio_len: int = 16000        # Drop raw files shorter than this (samples)
    test_size: float = 0.2
    
    # --- 2. SHAPING STRATEGIES ---
    window_samples: int = 8000      # 1.0 Second
    
    # How to handle files > 1.0s? 
    # Options: 'split' (keep all chunks), 'trim_start' (keep 1st), 'trim_random'
    long_file_strategy: str = "split" 
    
    # How to handle files < 1.0s?
    # Options: 'loop_pad' (repeat sound), 'zero_pad' (silence), 'drop' (ignore)
    short_file_strategy: str = "drop"
    
    # --- 3. AUGMENTATION & BALANCING ---
    # These only apply to the TRAINING set after splitting
    balance_classes: bool = True     # If True, oversamples minority class to match majority
    augment_prob: float = 1.0        # Probability to augment a sample if selected for balancing
    
    # --- 4. FEATURE PARAMETERS ---
    n_mfcc: int = 40
    drop_first_coeff: bool = True    # Ignore volume/energy
    normalize_audio: bool = True     # Normalize volume before feature extraction
    
    # --- 5. MODEL PARAMETERS ---
    model_type: str = "rf"           # 'rf', 'svm', 'log_reg'
    
    # GRID FOR SVM:
    if(model_type=='svm'):
        param_grid: Dict[str, List] = field(default_factory=lambda: {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        })
    elif (model_type=='rf'):
    # GRID FOR RANDOM FOREST (Example):
        param_grid: Dict[str, List] = field(default_factory=lambda: {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        })
        model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 150,
        'max_depth': 12,
        'class_weight': 'balanced',
        'n_jobs': -1
    })