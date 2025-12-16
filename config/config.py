# config.py
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import json

@dataclass
class ExperimentConfig:
    # --- 1. DATASET CONTROL ---
    data_sources: List[Dict[str, str]] = field(default_factory=lambda: [
        {'type': 'disk_hf', 'path': './my_offline_ds'},
        {'type': 'folder', 'path': 'Binary Wav ds'}
    ])
    sr: int = 16000
    use_background_noise: bool = True
    background_noise_path: str = "./EvalDatasets/ESC-50"

    validation_sources: Dict[str, str] = field(default_factory=lambda: {
        'class0': './EvalDatasets/TAU',
        'class1': './EvalDatasets/AudioDD'
    })

    max_raw_samples: int = 3e4     # Limit per class per source (or total, depending on logic)
    max_chunks_per_file: int = 10   # Prevent one file from dominating
    min_audio_len: int = 16000      # Drop raw files shorter than this (samples)
    test_size: float = 0.1
    
    # --- 2. SHAPING STRATEGIES ---
    window_samples: int = 8000
    long_file_strategy: str = "split" 
    short_file_strategy: str = "drop"
    
    # --- 3. AUGMENTATION & BALANCING ---
    balance_classes: bool = True
    augment_prob: float = 1.0
    
    # --- 4. FEATURE PARAMETERS ---
    feature_type: str = "mfcc"       # 'mfcc' or 'mel'
    
    # DL Control
    return_2d_features: bool = False # Set True for CNN

    # MFCC Params
    n_mfcc: int = 40
    drop_first_coeff: bool = True
    
    # Mel Params
    n_mels: int = 128
    fmax: int = 8000
    
    normalize_audio: bool = True
    
    # --- 5. MODEL PARAMETERS ---
    model_type: str = "rf"           # 'rf', 'svm', 'cnn', 'dnn', 'ensemble'
    
    # Default model params
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 150,
        'max_depth': 12,
        'class_weight': 'balanced',
        'n_jobs': -1
    })

    # Example Ensemble Configuration (copy this structure to model_params when using ensemble)
    # model_params = {
    #     'voting': 'soft',
    #     'estimators': [
    #         ('rf', 'rf', {'n_estimators': 100}),
    #         ('svm', 'svm', {'probability': True}),
    #         ('cnn', 'cnn', {'epochs': 5, 'batch_size': 32})
    #     ]
    # }

    # Search spaces for Randomized Search
    search_space: Dict[str, List] = field(default_factory=lambda: {
        'feature_type': ['mfcc', 'mel'],
        'n_mfcc': [20, 40],
        'model_type': ['rf', 'svm', 'cnn'],
        'rf_n_estimators': [50, 100],
        'svm_C': [0.1, 1.0],
        'cnn_lr': [0.001, 0.0001]
    })

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ExperimentConfig(**d)
