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
    n_mfcc: int = 13
    drop_first_coeff: bool = True
    
    # Mel Params
    n_mels: int = 128
    fmax: int = 8000
    
    normalize_audio: bool = True
    
    # --- 5. MODEL PARAMETERS ---
    
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
    model_type: str = "cnn"  # Change this to 'rf', 'svm', 'cnn', 'log_reg'
    # Leave this EMPTY. It will be auto-filled by __post_init__
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        This runs automatically after the class is initialized.
        If model_params is empty, it loads the defaults for the chosen model_type.
        """
        if not self.model_params:
            self.model_params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Defines the Default Parameter Dictionary for each architecture."""
        m_type = self.model_type.lower()
        
        # --- SIMPLE MODELS (Sklearn) ---
        if m_type == 'rf':
            return {
                'n_estimators': 150,
                'max_depth': 12,
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': 42
            }
            
        elif m_type == 'svm':
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,  # Critical for your evaluation plots
                'class_weight': 'balanced',
                'random_state': 42
            }
            
        elif m_type == 'log_reg':
            return {
                'solver': 'lbfgs',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }

        # --- DEEP MODELS (PyTorch) ---
        elif m_type in ['cnn', 'dnn']:
            return {
                # Wrapper / Training Params
                'epochs': 20,
                'batch_size': 32,
                'lr': 0.001,
                'device': 'cpu',  # or 'cpu'
                
                # Architecture Params
                'input_channels': 1,     # MFCC is treated as 1 channel image usually
                'n_classes': 2,
                'hidden_dim': 64,        # for DNN
                'dropout': 0.3
            }
            
        elif m_type == 'ensemble':
            return {
                'estimators': ['rf', 'svm'],
                'voting': 'soft'
            }
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ExperimentConfig(**d)
