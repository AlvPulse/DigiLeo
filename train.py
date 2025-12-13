# train.py
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom Modules
from config.config import ExperimentConfig
from src.dataset_loader import load_and_process
from src.models import ModelFactory # Use the factory we created in the previous step
import os
import pathlib
# ADJUST THIS PATH to point to your specific bin folder containing the .dll files
ffmpeg_bin_path = pathlib.Path(r"C:\program files\ffmpeg\bin")



def run_optimization_pipeline():
    # 1. INITIALIZE CONFIG
    cfg = ExperimentConfig() # Edits to config.py determine the behavior
    
    # 2. MLFLOW SETUP
    mlflow.set_experiment("Drone_Hyperparameter_Search")
    
    with mlflow.start_run() as run:
        print(f"🚀 Starting Optimization Run: {run.info.run_id}")
        mlflow.log_params(vars(cfg)) # Log the config "meta-data"

        # 3. LOAD DATA (Fixed Preprocessing)
        # Grid Search needs a unified X_train/y_train to do its internal CV splitting
        # We assume Fold 0 logic for the outer split, or load ALL data if doing internal CV
        #sources = [{'type': 'hf', 'path': 'geronimobasso/drone-audio-detection-samples'}]
        sources= [{'type': 'disk', 'path': './my_offline_ds'}]
        print("📦 Loading Data...")
        X_train, y_train = load_and_process(sources, cfg, fold_idx=0, mode='train')
        X_test, y_test = load_and_process(sources, cfg, fold_idx=0, mode='test')

        # 4. SCALING
        print("⚖️ Scaling...")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # 5. GET BASE MODEL
        # We ask the factory for a "blank" model of the requested type
        base_model = ModelFactory.get_model(cfg.model_type, {})
        
        # 6. HYPERPARAMETER SEARCH LOGIC
        final_model = None
        
        if cfg.search_type in ['grid', 'random'] and cfg.param_grid:
            print(f"🔎 Running {cfg.search_type.upper()} SEARCH on {cfg.model_type}...")
            print(f"   Space: {cfg.param_grid}")
            
            # Choose Search Strategy
            if cfg.search_type == 'grid':
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=cfg.param_grid,
                    cv=3,              # 3-Fold internal validation
                    scoring='accuracy',
                    n_jobs=-1,         # Use all cores
                    verbose=1
                )
            else:
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=cfg.param_grid,
                    n_iter=10,         # Limit random trials
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                
            # EXECUTE SEARCH
            search.fit(X_train_sc, y_train)
            
            print(f"🏆 Best Params Found: {search.best_params_}")
            print(f"   Best CV Score: {search.best_score_:.4f}")
            
            # Log the winner to MLflow
            mlflow.log_params(search.best_params_) # Logs winner params as top-level keys
            mlflow.log_metric("best_cv_score", search.best_score_)
            
            final_model = search.best_estimator_
            
        else:
            # MANUAL / SINGLE RUN MODE
            print(f"🛠️ Single Run Mode (No Search). Training {cfg.model_type}...")
            base_model.fit(X_train_sc, y_train)
            final_model = base_model

        # 7. FINAL EVALUATION (On the unseen Test Set)
        print("📝 Evaluating Final Model on Test Set...")
        y_pred = final_model.predict(X_test_sc)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Test Set Accuracy: {acc:.2%}")
        
        # Log Metrics
        mlflow.log_metric("test_accuracy", acc)
        print(classification_report(y_test, y_pred))

        # 8. SAVE ARTIFACTS
        # We save the WINNING model
        mlflow.sklearn.log_model(final_model, "best_model")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")
        
        return run.info.run_id

if __name__ == "__main__":
    if ffmpeg_bin_path.exists():
        # This explicitly adds the folder to the DLL search path for the current process
        os.add_dll_directory(str(ffmpeg_bin_path))
        
        # Optional: Update PATH for subprocess calls (like cmd commands)
        os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]
    else:
        print(f"Warning: FFmpeg bin directory not found at {ffmpeg_bin_path}")
    run_optimization_pipeline()