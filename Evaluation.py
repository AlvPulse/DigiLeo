import os
import glob
import numpy as np
import librosa
import joblib
import mlflow
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config.config import ExperimentConfig
from src.features import extract_features

# --- 1. MODEL & CONFIG LOADER ---
def load_artifacts(run_id=None):
    """
    Loads the model, scaler, AND config from MLflow. 
    If run_id is None, it picks the latest run.
    """
    if run_id is None:
        print("🔄 Auto-selecting latest MLflow run...")
        last_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]
        run_id = last_run.run_id
    
    print(f"📥 Loading artifacts from Run: {run_id}")
    
    # 1. Load Scaler
    local_scaler = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler = joblib.load(local_scaler)
    
    # 2. Load Model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # 3. Load Config (JSON)
    try:
        local_config = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config")
        with open(local_config, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
        print("✅ Loaded ExperimentConfig from artifact")
    except Exception as e:
        print(f"⚠️ Could not load config artifact ({e}). Using default config (may be risky!)")
        config = ExperimentConfig()
    
    # Identify Model Type for logging
    model_type = type(model).__name__
    print(f"✅ Loaded {model_type}")
    
    return scaler, model, config, run_id, model_type

# --- 2. FILE PROCESSOR (The "Chopper") ---
def process_file(filepath, label, scaler, model, config, vote_threshold=0.3, overlap=0.0):
    """
    Reads a file, chops it into windows, extracts features (using config), 
    and returns the MAJORITY VOTE prediction.
    """
    try:
        # Load Audio (resample to 16k)
        y, sr = librosa.load(filepath, sr=16000)
        
        # Normalize (Crucial if model was trained with normalization)
        if config.normalize_audio:
            y = librosa.util.normalize(y)
        
        # Parameters
        win_len = config.window_samples
        step = int(win_len * (1 - overlap))
        
        # Scan the file
        chunks = []
        
        # Create chunks first
        for i in range(0, len(y) - win_len + 1, step):
            chunks.append(y[i : i + win_len])
        #print(len(chunks))
        if not chunks:
            return None, None # File too short
            
        # Feature Extraction (Using SHARED logic from src.features)
        # extract_features expects a list of arrays
        feats = extract_features(chunks, config)
        
        # Scale
        feats_sc = scaler.transform(feats)
        
        # Predict all chunks at once
        try:
            chunk_preds = model.predict(feats_sc)
        except:
            chunk_preds = [0] * len(feats_sc) # Fallback
            
        # --- VOTING LOGIC ---
        # Calculate percentage of "Event" chunks
        event_ratio = np.mean(chunk_preds)
        
        # Final Decision
        final_pred = 1 if event_ratio >= vote_threshold else 0
        
        return final_pred, event_ratio,chunk_preds

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None, None

# --- 3. BENCHMARK RUNNER ---
def run_benchmark(run_id=None):
    # 1. Load Tools
    scaler, model, config, run_id, model_name = load_artifacts(run_id)
    
    # 2. Gather Files (Using paths from config, or defaults if missing)
    # We use the config's validation sources if available
    
    sources = config.validation_sources
    path_0 = sources.get('class0', './EvalDatasets/ESC-50')
    path_1 = sources.get('class1', './EvalDatasets/AudioDD')
    
    files_0 = glob.glob(os.path.join(path_0, "*.wav"))
    files_1 = glob.glob(os.path.join(path_1, "*.wav"))
    
    print(f"📂 Found {len(files_0)} Class 0 files ({path_0})")
    print(f"📂 Found {len(files_1)} Class 1 files ({path_1})")
    
    y_true = []
    y_pred = []
    ratios = []
    
    print("🚀 Starting Evaluation...")
    
    # Process Class 0
    for f in files_0:
        pred, ratio,chunk_preds = process_file(f, 0, scaler, model, config)
        if pred is not None:
            
            # y_true.append(0)
            # y_pred.append(pred)
            # ratios.append(ratio)
            y_true.extend([0]* len(chunk_preds))
            y_pred.extend(chunk_preds)
    
    # Process Class 1
    for f in files_1:
        pred, ratio,chunk_preds = process_file(f, 1, scaler, model, config)
        if pred is not None:
            # y_true.append(1)
            # y_pred.append(pred)
            # ratios.append(ratio)
            y_true.extend([1]* len(chunk_preds))
            y_pred.extend(chunk_preds)
            
    if not y_true:
        print("❌ No valid files processed. Aborting.")
        return
    #print(y_true.size(),y_pred.size())
    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*40)
    print(f"   BENCHMARK REPORT: {model_name}")
    print("="*40)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%} (Low false alarms?)")
    print(f"Recall:    {rec:.2%} (Caught all events?)")
    print(f"F1 Score:  {f1:.2f}")
    
    # 4. Log to Existing Run
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("benchmark_accuracy", acc)
            mlflow.log_metric("benchmark_recall", rec)
            mlflow.log_metric("benchmark_f1", f1)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Background', 'Event'], yticklabels=['Background', 'Event'])
            plt.title(f"Benchmark: {model_name}\n(Threshold > 30%)")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            
            plt.savefig("benchmark_confusion_matrix.png")
            mlflow.log_artifact("benchmark_confusion_matrix.png")
            plt.close()
        print(f"✅ Results logged to Run ID: {run_id}")
    except Exception as e:
        print(f"⚠️ Could not log to MLflow run {run_id}: {e}")

if __name__ == "__main__":
    # You can pass a specific run_id string here, e.g., "a1b2c3..."
    # Or leave None to test the latest model
    run_benchmark("cf05e4d7f48e4d769603833ad7d0ac5b")

