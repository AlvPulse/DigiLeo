# models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
# You can add XGBoost/LightGBM here later if installed
# from xgboost import XGBClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_type, params):
        """
        Factory method to initialize any model dynamically.
        
        Args:
            model_type (str): 'rf', 'svm', 'log_reg', 'gmm'
            params (dict): Dictionary of hyperparameters (e.g. {'n_estimators': 100})
        """
        model_type = model_type.lower()
        
        if model_type == 'rf':
            # Set defaults but allow overrides from params
            default_params = {
                'n_estimators': 100, 
                'max_depth': 10, 
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': 42
            }
            # Update defaults with whatever user passed in config
            default_params.update(params)
            return RandomForestClassifier(**default_params)
            
        elif model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True, # CRITICAL: Required for predict_proba
                'class_weight': 'balanced',
                'random_state': 42
            }
            default_params.update(params)
            return SVC(**default_params)
            
        elif model_type == 'log_reg':
            default_params = {
                'solver': 'lbfgs',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
            
        elif model_type == 'gmm':
            # GMM is a special case (Generative), but we can wrap it if needed
            # For now, let's treat it as a standard estimator setup
            default_params = {
                'n_components': 16,
                'covariance_type': 'diag',
                'random_state': 42
            }
            default_params.update(params)
            # Note: GMM doesn't have a simple 'fit(X, y)' for classification
            # It usually requires custom handling (one GMM per class).
            # If you want strictly generic pipeline, GMM is the odd one out.
            raise NotImplementedError("GMM requires custom training logic separate from classifiers.")
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")