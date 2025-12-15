import os
import glob
import numpy as np
import librosa
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURATION ---
BENCHMARK_CONFIG = {
    'folder_class0': './EvalDatasets/ESC-50',    # Update path to your Zeros folder
    'folder_class1': './EvalDatasets/droneAudio',# Update path to your Ones folder
    'window_samples': 16000,
    'overlap': 0.0,                      # 0% overlap for strict testing
    'vote_threshold': 0.3                # If >30% of chunks are 'Event', file is Event
}

# --- 1. MODEL LOADER ---
def load_artifacts(run_id=None):
    """
    Loads the model and scaler from MLflow. 
    If run_id is None, it picks the latest run.
    """
    if run_id is None:
        print("🔄 Auto-selecting latest MLflow run...")
        last_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]
        run_id = last_run.run_id
    
    print(f"📥 Loading artifacts from Run: {run_id}")
    
    # Download & Load Scaler
    local_scaler = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler = joblib.load(local_scaler)
    
    # Load Model (Generic sklearn loader works for RF, SVM, etc.)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Identify Model Type for logging
    model_type = type(model).__name__
    print(f"✅ Loaded {model_type}")
    
    return scaler, model, run_id, model_type

# --- 2. FILE PROCESSOR (The "Chopper") ---
def process_file(filepath, label, scaler, config):
    """
    Reads a file, chops it into 1s windows, extracts features, 
    and returns the MAJORITY VOTE prediction.
    """
    try:
        # Load Audio (resample to 16k)
        y, sr = librosa.load(filepath, sr=16000)
        
        # Normalize (Crucial if model was trained with normalization)
        y = librosa.util.normalize(y)
        
        # Parameters
        win_len = config['window_samples']
        step = int(win_len * (1 - config['overlap']))
        
        # Scan the file
        chunk_preds = []
        
        for i in range(0, len(y) - win_len + 1, step):
            chunk = y[i : i + win_len]
            
            # Feature Extraction (Must match Training Logic!)
            mfcc = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=13)
            # Drop 1st Coeff (Volume)
            mfcc = mfcc[1:] 
            feat = np.mean(mfcc.T, axis=0).reshape(1, -1)
            
            # Scale
            feat_sc = scaler.transform(feat)
            
            # Predict (1 chunk)
            # Some models don't support predict_proba, so we try/except
            try:
                # Use probability if available for softer voting
                prob = 1.0 if hasattr(model, "predict_proba") else model.predict(feat_sc)[0]
                pred = 1 if prob > 0.5 else 0 
                # Actually, simpler: just use hard predict for chunks
                pred = model.predict(feat_sc)[0]
            except:
                pred = model.predict(feat_sc)[0]
                
            chunk_preds.append(pred)
            
        if not chunk_preds:
            return None, None # File too short
            
        # --- VOTING LOGIC ---
        # Calculate percentage of "Event" chunks
        event_ratio = sum(chunk_preds) / len(chunk_preds)
        
        # Final Decision
        final_pred = 1 if event_ratio >= config['vote_threshold'] else 0
        
        return final_pred, event_ratio

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None, None

# --- 3. BENCHMARK RUNNER ---
def run_benchmark(run_id=None):
    # 1. Load Tools
    scaler, model, run_id, model_name = load_artifacts(run_id)
    
    # 2. Gather Files
    files_0 = glob.glob(os.path.join(BENCHMARK_CONFIG['folder_class0'], "*.wav"))
    files_1 = glob.glob(os.path.join(BENCHMARK_CONFIG['folder_class1'], "*.wav"))
    
    print(f"📂 Found {len(files_0)} Background files (ESC-50)")
    print(f"📂 Found {len(files_1)} Event files (My Data)")
    
    y_true = []
    y_pred = []
    ratios = []
    
    print("🚀 Starting Evaluation...")
    
    # Process Class 0
    for f in files_0:
        pred, ratio = process_file(f, 0, scaler, BENCHMARK_CONFIG)
        if pred is not None:
            y_true.append(0)
            y_pred.append(pred)
            ratios.append(ratio)
            
    # Process Class 1
    for f in files_1:
        pred, ratio = process_file(f, 1, scaler, BENCHMARK_CONFIG)
        if pred is not None:
            y_true.append(1)
            y_pred.append(pred)
            ratios.append(ratio)
            
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
    
    # 4. Log to Existing Run (or create new 'eval' run)
    # We prefer logging to the SAME run so we can see 'test_acc' vs 'benchmark_acc'
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("benchmark_accuracy", acc)
        mlflow.log_metric("benchmark_recall", rec)
        mlflow.log_metric("benchmark_f1", f1)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Background', 'Event'], yticklabels=['Background', 'Event'])
        plt.title(f"Benchmark: {model_name}\n(Threshold > {BENCHMARK_CONFIG['vote_threshold']*100:.0f}%)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        plt.savefig("benchmark_confusion_matrix.png")
        mlflow.log_artifact("benchmark_confusion_matrix.png")
        plt.close()
        
    print(f"✅ Results logged to Run ID: {run_id}")

if __name__ == "__main__":
    # You can pass a specific run_id string here, e.g., "a1b2c3..."
    # Or leave None to test the latest model
    run_benchmark(run_id="44b5ee424f2b4e16b5d2ff3d27dc4eab")