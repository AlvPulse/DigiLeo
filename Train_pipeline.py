# train_pipeline.py
import mlflow
import mlflow.sklearn
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from src.models import ModelFactory 

# Modular Imports
from config.config import ExperimentConfig
from src.dataset_loader import load_raw_dataset, balance_training_set
from src.models import ModelFactory # Use the factory from previous step
import os
import pathlib
# ADJUST THIS PATH to point to your specific bin folder containing the .dll files
ffmpeg_bin_path = pathlib.Path(r"C:\program files\ffmpeg\bin")

def extract_features(X, config):
    """Converts Raw Audio to MFCC Vectors"""
    print("   Extracting Features...")
    feats = []
    for x in X:
        # Re-normalize before feature extraction (safe practice)
        if config.normalize_audio:
            x = librosa.util.normalize(x)
            
        mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=config.n_mfcc)
        
        if config.drop_first_coeff:
            mfcc = mfcc[1:]
            
        feats.append(np.mean(mfcc.T, axis=0))
    return np.array(feats)

def run_pipeline():
    # 1. Initialize Config
    cfg = ExperimentConfig()
    
    mlflow.set_experiment("Rational_Drone_Pipeline")
    
    with mlflow.start_run() as run:
        mlflow.log_params(vars(cfg))
        
        # 2. Load Raw Data (Clean, No Augmentation)
        X_raw, y_raw = load_raw_dataset(cfg)
        
        # 3. Rational Split (Stratified)
        print("✂️ Splitting Train/Test...")
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, 
            test_size=cfg.test_size, 
            stratify=y_raw, 
            random_state=42
        )
        
        # 4. Balance & Augment (TRAIN ONLY)
        # This prevents leakage because X_test_raw is untouched
        X_train_aug, y_train_aug = balance_training_set(X_train_raw, y_train_raw, cfg)
        
        # 5. Feature Extraction
        print("featurizing Train...")
        X_train_vec = extract_features(X_train_aug, cfg)
        print("featurizing Test...")
        X_test_vec = extract_features(X_test_raw, cfg)
        
        # 6. Scaling
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_vec)
        X_test_sc = scaler.transform(X_test_vec)
        
        # 7. Model Training
        print(f"🛠️ Training {cfg.model_type}...")
        # (Assuming you kept models.py from before, or use direct import)
        
        model = ModelFactory.get_model(cfg.model_type, cfg.model_params)
        model.fit(X_train_sc, y_train_aug)
        
        # 8. Evaluation
        print("📝 Evaluating...")
        y_pred = model.predict(X_test_sc)
        acc = accuracy_score(y_test_raw, y_pred)
        
        print(f"✅ Test Accuracy: {acc:.2%}")
        print(classification_report(y_test_raw, y_pred))
        
        # 9. Logging
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")

if __name__ == "__main__":
    if ffmpeg_bin_path.exists():
        # This explicitly adds the folder to the DLL search path for the current process
        os.add_dll_directory(str(ffmpeg_bin_path))
        
        # Optional: Update PATH for subprocess calls (like cmd commands)
        os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]
    else:
        print(f"Warning: FFmpeg bin directory not found at {ffmpeg_bin_path}")
    run_pipeline()