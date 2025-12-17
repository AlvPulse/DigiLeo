# train_pipeline.py
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import json
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, DetCurveDisplay
import matplotlib.pyplot as plt
import os
import pathlib

# Modular Imports
from config.config import ExperimentConfig
from src.dataset_loader import load_raw_dataset, balance_training_set
from src.features import extract_features
from src.models import ModelFactory 

# ADJUST THIS PATH to point to your specific bin folder containing the .dll files
ffmpeg_bin_path = pathlib.Path(r"C:\program files\ffmpeg\bin")

def train_model(cfg, experiment_name="Rational_Drone_Pipeline", parent_run_id=None):
    """
    Refactored Training Logic accepting a Config Object.
    Can be called standalone or from randomized search.
    """
    mlflow.set_experiment(experiment_name)
    
    # Nested run support if parent_run_id is provided
    # If parent_run_id is None, start a new root run
    with mlflow.start_run(run_id=None, nested=(parent_run_id is not None)) as run:
        # If we are in a nested run (hyperopt), we might want to log tags
        if parent_run_id:
            mlflow.set_tag("parent_run", parent_run_id)
            
        print(f"\n🚀 Starting Run: {run.info.run_id}")
        mlflow.log_params(cfg.to_dict())
        
        # 2. Load Raw Data
        X_raw, y_raw = load_raw_dataset(cfg)
        
        if len(X_raw) == 0:
            print("❌ No data loaded! Check dataset paths.")
            return
        
        # 3. Rational Split (Stratified)
        print("✂️ Splitting Train/Test...")
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_raw, y_raw, 
                test_size=cfg.test_size, 
                stratify=y_raw, 
                random_state=42
            )
        except ValueError as e:
            print(f"❌ Split failed (maybe too few samples?): {e}")
            return
        
        # 4. Balance & Augment (TRAIN ONLY)
        X_train_aug, y_train_aug = balance_training_set(X_train_raw, y_train_raw, cfg)
        
        # 5. Feature Extraction & Smart Configuration
        # Apply specific settings based on model family
        classical_models = ['rf', 'svm', 'log_reg', 'ensemble']
        dl_models = ['cnn', 'dnn', 'lstm', 'sara_cnn']

        if cfg.model_type in dl_models:
             # Deep Learning: Force FBE (Mel) and 2D Input
             # User requested FBE (Mel) for DL, MFCC for others.
             print(f"🧠 Deep Learning Model ({cfg.model_type}) detected.")
             print("   -> Switching to Mel-Spectrogram (FBE) features (n_mels={cfg.n_mels}).")
             print("   -> Enabling 2D Feature return.")
             cfg.feature_type = 'mel'
             cfg.return_2d_features = True

        elif cfg.model_type in classical_models:
             # Classical: Prefer Enriched MFCC 1D
             print(f"🤖 Classical Model ({cfg.model_type}) detected.")
             if cfg.feature_type != 'mfcc':
                 print("   -> Switching to MFCC features (standard for RF/SVM).")
                 cfg.feature_type = 'mfcc'
             cfg.return_2d_features = False

        # 1. Save Config as Artifact (Crucial for Evaluation.py)
        # We save it HERE, after Smart Configuration has potentially modified it.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(cfg.to_dict(), tmp, indent=4)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="config")
        os.remove(tmp_path)

        print(f"feat ({cfg.feature_type}) Train...")
        X_train_vec = extract_features(X_train_aug, cfg)
        print(f"feat ({cfg.feature_type}) Test...")
        X_test_vec = extract_features(X_test_raw, cfg)
        
        # 5b. NaN Cleaning (Crucial step added)
        # Check for NaNs/Infs in features (if any slipped through extract_features or 2D extraction)
        if np.isnan(X_train_vec).any() or np.isinf(X_train_vec).any():
            print("⚠️ NaNs/Infs detected in Training Data! Cleaning...")
            mask = ~np.isnan(X_train_vec).any(axis=tuple(range(1, X_train_vec.ndim))) & \
                   ~np.isinf(X_train_vec).any(axis=tuple(range(1, X_train_vec.ndim)))
            print(f"   Dropping {len(X_train_vec) - mask.sum()} corrupted samples.")
            X_train_vec = X_train_vec[mask]
            y_train_aug = y_train_aug[mask]

        if np.isnan(X_test_vec).any() or np.isinf(X_test_vec).any():
             print("⚠️ NaNs/Infs detected in Test Data! Cleaning...")
             mask = ~np.isnan(X_test_vec).any(axis=tuple(range(1, X_test_vec.ndim))) & \
                    ~np.isinf(X_test_vec).any(axis=tuple(range(1, X_test_vec.ndim)))
             X_test_vec = X_test_vec[mask]
             y_test_raw = y_test_raw[mask] # Note: y_test_raw matches X_test_raw indices, hope extract_features didn't shuffle
             # Actually extract_features output matches input list order.
             # So this masking aligns X_test_vec and y_test_raw correctly.

        # 6. Scaling
        # Skip scaling for 3D data (DL models have internal BatchNorm)
        # OR Flatten -> Scale -> Reshape
        if len(X_train_vec.shape) > 2:
            print("   ℹ️ 3D Data detected. Skipping StandardScaler (relying on Batch Norm).")
            X_train_sc = X_train_vec
            X_test_sc = X_test_vec
            scaler = None # Signal that no scaler is used
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_vec)
            X_test_sc = scaler.transform(X_test_vec)
        
        # 7. Model Training
        print(f"🛠️ Training {cfg.model_type}...")
        model = ModelFactory.get_model(cfg.model_type, cfg.model_params)
        model.fit(X_train_sc, y_train_aug)
        
        # 8. Evaluation
        print("📝 Evaluating...")
        y_pred = model.predict(X_test_sc)

        # Get Probabilities for DET Curve (if supported)
        try:
            y_scores = model.predict_proba(X_test_sc)[:, 1] # Probability of Class 1
        except:
            # Fallback for models without probability (e.g. some SVMs if prob=False)
            # Use decision_function if available
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test_sc)
            else:
                y_scores = y_pred # Hard fallback

        acc = accuracy_score(y_test_raw, y_pred)
        
        print(f"✅ Test Accuracy: {acc:.2%}")
        #print(f"")
        print(classification_report(y_test_raw, y_pred)) # Can be noisy in loops
        
        # DET Curve Plotting
        try:
            print("   📊 Generating DET Curve...")
            DetCurveDisplay.from_predictions(y_test_raw, y_scores)
            plt.title(f"DET Curve - {cfg.model_type}")
            plt.savefig("det_curve_val.png")
            mlflow.log_artifact("det_curve_val.png")
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Failed to plot DET Curve: {e}")

        # 9. Logging
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        if scaler is not None:
            joblib.dump(scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl")
        else:
            # Log a dummy file to indicate no scaler, or just skip it.
            # Better to skip, and handle missing scaler in Evaluation.py
            pass
        
        return run.info.run_id

if __name__ == "__main__":
    if ffmpeg_bin_path.exists():
        os.add_dll_directory(str(ffmpeg_bin_path))
        os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]
    else:
        # Only warn, don't crash, might be on Linux/Mac
        pass
        
    # Default behavior: Load default config and run once
    cfg = ExperimentConfig()
    train_model(cfg)
