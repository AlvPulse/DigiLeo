# models.py
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import copy
from src.dl_models import PyTorchClassifier, AudioCNN, AudioDNN, SaraCNN

class ModelFactory:
    @staticmethod
    def get_model(model_type, params):
        """
        Factory method to initialize any model dynamically.
        
        Args:
            model_type (str): 'rf', 'svm', 'log_reg', 'cnn', 'dnn', 'ensemble', 'sara_cnn'
            params (dict): Dictionary of hyperparameters
        """
        model_type = model_type.lower()
        
        if model_type == 'rf':
            default_params = {
                'n_estimators': 100, 
                'max_depth': 10, 
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': 42
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
            
        elif model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True,
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

        elif model_type == 'cnn':
            # Params for wrapper: epochs, batch_size, lr
            wrapper_params = {k: v for k, v in params.items() if k in ['epochs', 'batch_size', 'lr', 'device']}
            # Params for model architecture: n_features, n_timesteps (passed dynamically often)
            model_arch_params = {k: v for k, v in params.items() if k not in wrapper_params}
            
            return PyTorchClassifier(AudioCNN, model_arch_params, **wrapper_params)

        elif model_type == 'sara_cnn':
            wrapper_params = {k: v for k, v in params.items() if k in ['epochs', 'batch_size', 'lr', 'device']}
            model_arch_params = {k: v for k, v in params.items() if k not in wrapper_params}
            return PyTorchClassifier(SaraCNN, model_arch_params, **wrapper_params)

        elif model_type == 'dnn':
            wrapper_params = {k: v for k, v in params.items() if k in ['epochs', 'batch_size', 'lr', 'device']}
            model_arch_params = {k: v for k, v in params.items() if k not in wrapper_params}

            return PyTorchClassifier(AudioDNN, model_arch_params, **wrapper_params)

        elif model_type == 'ensemble':
            # Expects 'estimators' list in params, where each item is (name, model_type, model_params)
            # Example: [('rf', 'rf', {...}), ('svm', 'svm', {...})]
            estimators_config = params.get('estimators', [])
            estimators = []

            for name, m_type, m_params in estimators_config:
                estimators.append((name, ModelFactory.get_model(m_type, m_params)))

            voting = params.get('voting', 'soft')
            return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
