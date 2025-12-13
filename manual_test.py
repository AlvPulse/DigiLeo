# test_manual.py
import mlflow.sklearn
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import os
from config import ExperimentConfig

# --- CONFIGURATION ---
TEST_FILE = "EvalDataset/Jamal.wav"  # <--- Your file here
RUN_ID = "REPLACE_WITH_RUN_ID_FROM_TRAIN_OUTPUT" # e.g. "a1b2c3d4..."

# If you don't want to copy-paste the ID every time, 
# you can set RUN_ID = None to auto-pick the latest run (see below)

def download_artifacts(run_id):
    print(f"📥 Downloading artifacts from Run: {run_id}...")
    
    # 1. Download Scaler
    local_scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl"
    )
    scaler = joblib.load(local_scaler_path)
    
    # 2. Load Model
    model_uri = f"runs:/{run_id}/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return scaler, model

def test_file_on_model(file_path, run_id):
    # Auto-detect latest run if ID not provided
    if run_id is None:
        last_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]
        run_id = last_run.run_id
        print(f"🔄 Auto-selected latest Run ID: {run_id}")

    # 1. Load the exact tools used in training
    scaler, model = download_artifacts(run_id)
    
    # 2. Load Config (Ideally loaded from MLflow, but we use local class for now)
    # We must match the preprocessing logic (Window size, etc.)
    cfg = ExperimentConfig() 
    
    print(f"🎧 Analyzing {file_path}...")
    y, sr = librosa.load(file_path, sr=16000)
    
    # Sliding Window
    window_samples = cfg.window_samples # 16000
    step_samples = int(window_samples * 0.1) # 10% step (High Res)
    
    timestamps = []
    probs = []
    
    for i in range(0, len(y) - window_samples, step_samples):
        chunk = y[i : i + window_samples]
        
        # --- REPLICATE TRAINING PREPROCESSING ---
        # 1. Normalize (If config says so)
        if cfg.normalize_audio:
            chunk = librosa.util.normalize(chunk)
            
        # 2. MFCC
        mfcc = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=cfg.n_mfcc)
        
        # 3. Drop First Coeff (If config says so)
        if cfg.drop_first_coeff:
            mfcc = mfcc[1:]
            
        # 4. Flatten
        feat = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
        # --- PREDICT ---
        feat_scaled = scaler.transform(feat)
        prob = model.predict_proba(feat_scaled)[0][1] # Probability of Class 1
        
        timestamps.append(i / 16000)
        probs.append(prob)

    # --- VISUALIZE ---
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, probs, label="Event Probability", color='#2ca02c')
    plt.axhline(0.6, color='red', linestyle='--', label="Threshold")
    plt.fill_between(timestamps, probs, 0, where=(np.array(probs)>0.6), color='red', alpha=0.3)
    
    plt.title(f"Detection Result (Run: {run_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Please create a test file at: {TEST_FILE}")
    else:
        test_file_on_model(TEST_FILE, RUN_ID)