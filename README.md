# Rational Event Pipeline

This repository implements a machine learning pipeline for audio event detection. It supports classical Machine Learning (RF, SVM), Deep Learning (CNN, DNN), and Ensemble methods.

## Key Features

*   **Robust Data Loading**: Handles large datasets efficiently by chunking long audio files and using random sampling to prevent memory overflows.
*   **Feature Enrichment**:
    *   **1D Models (RF/SVM)**: Enriched features including Mean, Std Dev, Min/Max, Skewness, Kurtosis, Entropy, and Delta/Delta-Deltas.
    *   **DL Models**: 2D MFCC or Mel-Spectrogram inputs.
*   **Deep Learning Support**: PyTorch-based CNN and DNN models integrated seamlessly via a Scikit-Learn wrapper.
*   **Ensembling**: Easy-to-configure Voting Classifiers to combine multiple model types.
*   **Experiment Tracking**: MLflow integration for logging params, metrics, and models.

## Installation

1.  **Clone the repository.**
2.  **Install dependencies**:
    ```bash
    pip install torch torchaudio scikit-learn scipy librosa pandas numpy datasets mlflow
    ```
    *(Note: PyTorch installation might vary based on your CUDA version. See [pytorch.org](https://pytorch.org))*

## Configuration

All configuration is managed in `config/config.py`.

### 1. Data Loading Strategies
*   `max_raw_samples`: Max number of *chunks* to load per class per source.
*   `max_chunks_per_file`: Max chunks to take from a single file (prevents long files from dominating).
*   `long_file_strategy`: 'split' (keep all chunks) or 'trim'.

### 2. Feature Extraction
*   `feature_type`: 'mfcc' or 'mel'.
*   `return_2d_features`:
    *   `True`: Returns (N, Features, Time) for CNN.
    *   `False`: Returns enriched 1D statistical vectors for RF/SVM/DNN.

### 3. Model Types
Supported `model_type` values: `'rf'`, `'svm'`, `'log_reg'`, `'cnn'`, `'dnn'`, `'ensemble'`.

#### Deep Learning (CNN/DNN)
To use a CNN:
1.  Set `model_type = 'cnn'`.
2.  Set `return_2d_features = True`.
3.  Configure `model_params` (e.g., `epochs`, `batch_size`, `lr`).

#### Ensembling
To create a Voting Classifier:
1.  Set `model_type = 'ensemble'`.
2.  Set `return_2d_features = False` (unless all sub-models support 2D, but RF/SVM do not). *Note: Currently, ensembling assumes all models take the same input format. If mixing 1D and 2D models, custom code adaptation is needed.*
3.  Define `model_params`:
    ```python
    model_params = {
        'voting': 'soft',
        'estimators': [
            ('rf', 'rf', {'n_estimators': 100}),
            ('svm', 'svm', {'probability': True})
            # Add 'dnn' here if input is 1D
        ]
    }
    ```

## Running Experiments

**Train a single model:**
```bash
python Train_pipeline.py
```

**Run Hyperparameter Search:**
```bash
python RandomizedSearch.py
```

**Evaluate:**
```bash
python Evaluation.py
```

## Project Structure

*   `src/dataset_loader.py`: Efficient streaming and sampling of audio data.
*   `src/features.py`: Feature extraction (Stats for 1D, Matrix for 2D).
*   `src/models.py`: Model factory (RF, SVM, Ensemble).
*   `src/dl_models.py`: PyTorch CNN/DNN implementations.
*   `Train_pipeline.py`: Main training loop with MLflow logging.
