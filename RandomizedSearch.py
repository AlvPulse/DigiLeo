# randomized_search.py
import random
import copy
import mlflow
import numpy as np
import os
import pathlib
from config.config import ExperimentConfig
from Train_pipeline import train_model

# Helper to sample from the search space defined in Config
def sample_config(base_cfg):
    """
    Creates a new ExperimentConfig object with random parameters
    sampled from base_cfg.search_space.
    """
    new_cfg = copy.deepcopy(base_cfg)
    space = base_cfg.search_space
    
    # Sample Preprocessing Params
    if 'feature_type' in space:
        new_cfg.feature_type = random.choice(space['feature_type'])
    
    if 'n_mfcc' in space:
        new_cfg.n_mfcc = random.choice(space['n_mfcc'])
        
    if 'n_mels' in space:
        new_cfg.n_mels = random.choice(space['n_mels'])
        
    if 'window_samples' in space:
        new_cfg.window_samples = random.choice(space['window_samples'])
        
    # Sample Model Selection
    if 'model_type' in space:
        new_cfg.model_type = random.choice(space['model_type'])
        
    # Sample Model Hyperparameters
    # We clear the default model_params and build fresh
    params = {}
    
    if new_cfg.model_type == 'rf':
        if 'rf_n_estimators' in space:
            params['n_estimators'] = random.choice(space['rf_n_estimators'])
        if 'rf_max_depth' in space:
            params['max_depth'] = random.choice(space['rf_max_depth'])
            
    elif new_cfg.model_type == 'svm':
        if 'svm_C' in space:
            params['C'] = random.choice(space['svm_C'])
        if 'svm_kernel' in space:
            params['kernel'] = random.choice(space['svm_kernel'])
            
    # Assign the specific model params
    new_cfg.model_params = params
    
    return new_cfg

def run_randomized_search(n_iter=5):
    # Setup FFmpeg if needed (Windows)
    ffmpeg_bin_path = pathlib.Path(r"C:\program files\ffmpeg\bin")
    if ffmpeg_bin_path.exists():
        os.add_dll_directory(str(ffmpeg_bin_path))
        os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]

    print(f"🔎 Starting Randomized Search ({n_iter} iterations)...")
    
    base_cfg = ExperimentConfig()
    
    # We can group these under a parent run for better MLflow organization
    mlflow.set_experiment("Drone_Randomized_Search")
    
    with mlflow.start_run(run_name="Hyperopt_Parent") as parent_run:
        print(f"Parent Run ID: {parent_run.info.run_id}")
        
        for i in range(n_iter):
            print(f"\n--- Iteration {i+1}/{n_iter} ---")
            
            # 1. Sample Config
            trial_cfg = sample_config(base_cfg)
            print(f"Settings: Feat={trial_cfg.feature_type}, Model={trial_cfg.model_type}, Params={trial_cfg.model_params}")
            
            # 2. Run Pipeline
            # We pass the parent_run_id if we want nesting (optional, but good for grouping)
            try:
                train_model(trial_cfg, experiment_name="Drone_Randomized_Search", parent_run_id=parent_run.info.run_id) 
            except Exception as e:
                print(f"❌ Trial failed: {e}")
                
    print("\n✅ Search Complete. Check MLflow UI for results.")

if __name__ == "__main__":
    run_randomized_search(n_iter=5)
