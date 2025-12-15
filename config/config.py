# config.py
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import json

@dataclass
class ExperimentConfig:
    # --- 1. DATASET CONTROL ---
    # We now support multiple sources
    data_sources: List[Dict[str, str]] = field(default_factory=lambda: [
        {'type': 'disk_hf', 'path': './my_offline_ds'},
        {'type': 'folder', 'path': 'Binary Wav ds'} # "yes"/"no" folders inside
    ])
    sr= 16000
    
    # Validation datasets for Evaluation.py
    validation_sources: Dict[str, str] = field(default_factory=lambda: {
        'class0': './EvalDatasets/ESC-50',
        'class1': './EvalDatasets/AudioDD'
    })

    max_raw_samples: int = 3000     # Limit per class per source (or total, depending on logic)
    min_audio_len: int = 16000      # Drop raw files shorter than this (samples)
    test_size: float = 0.2
    
    # --- 2. SHAPING STRATEGIES ---
    window_samples: int = 8000      # 1.0 Second (at 16k, wait 8000 is 0.5s) -> Default was 8000
    # Note: User said 16000 in Evaluation.py, but config said 8000. 
    # We should probably align this or allow it to be tuned.
    
    # How to handle files > window_samples? 
    # Options: 'split' (keep all chunks), 'trim_start' (keep 1st), 'trim_random'
    long_file_strategy: str = "split" 
    
    # How to handle files < window_samples?
    # Options: 'loop_pad' (repeat sound), 'zero_pad' (silence), 'drop' (ignore)
    short_file_strategy: str = "loop_pad"
    
    # --- 3. AUGMENTATION & BALANCING ---
    # These only apply to the TRAINING set after splitting
    balance_classes: bool = True     # If True, oversamples minority class to match majority
    augment_prob: float = 1.0        # Probability to augment a sample if selected for balancing
    
    # --- 4. FEATURE PARAMETERS ---
    feature_type: str = "mfcc"       # 'mfcc' or 'mel'
    
    # MFCC Params
    n_mfcc: int = 40
    drop_first_coeff: bool = True    # Ignore volume/energy
    
    # Mel Params
    n_mels: int = 128
    fmax: int = 8000                 # Nyquist for 16k
    
    normalize_audio: bool = True     # Normalize volume before feature extraction
    
    # --- 5. MODEL PARAMETERS ---
    model_type: str = "rf"           # 'rf', 'svm', 'log_reg'
    
    # Default model params (used if not tuning)
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 150,
        'max_depth': 12,
        'class_weight': 'balanced',
        'n_jobs': -1
    })

    # Search spaces for Randomized Search (used by randomized_search.py)
    # This defines the "Universe" of possible experiments
    search_space: Dict[str, List] = field(default_factory=lambda: {
        # Preprocessing
        'feature_type': ['mfcc', 'mel'],
        'n_mfcc': [13, 20, 40],
        'n_mels': [64, 128],
        'window_samples': [8000, 16000, 32000], # 0.5s, 1s, 2s
        
        # Models
        'model_type': ['rf', 'svm'],
        
        # Model Hyperparams (Nested strategy requires custom handling or flat keys)
        'rf_n_estimators': [50, 100, 200],
        'rf_max_depth': [10, 20, None],
        
        'svm_C': [0.1, 1.0, 10.0],
        'svm_kernel': ['rbf', 'poly']
    })

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ExperimentConfig(**d)
