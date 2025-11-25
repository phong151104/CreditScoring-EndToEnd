"""
Model Training Module
Handles training of various machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def train_model(X_train, y_train, X_test, y_test, model_type, params=None):
    """
    Train a machine learning model based on the specified type and parameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_type : str
        Type of model to train
    params : dict
        Model parameters
        
    Returns:
    --------
    model : object
        Trained model object
    metrics : dict
        Dictionary of evaluation metrics
    """
    if params is None:
        params = {}

    model = None
    
    try:
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 200),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', -1),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(
                iterations=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=0,
                allow_writing_files=False
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return model, metrics
        
    except Exception as e:
        raise Exception(f"Error training {model_type}: {str(e)}")


def cross_validate_model(X, y, model_type, params=None, cv_folds=5):
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_type : str
        Type of model to train
    params : dict
        Model parameters
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    cv_results : dict
        Dictionary containing CV scores and statistics
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    if params is None:
        params = {}
    
    model = None
    
    try:
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 200),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', -1),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(
                iterations=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=0,
                allow_writing_files=False
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Stratified K-Fold for classification
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores for multiple metrics
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        cv_results = {
            'cv_folds': cv_folds,
            'accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores.tolist()
            },
            'precision': {
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'scores': precision_scores.tolist()
            },
            'recall': {
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'scores': recall_scores.tolist()
            },
            'f1': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores.tolist()
            },
            'auc': {
                'mean': auc_scores.mean(),
                'std': auc_scores.std(),
                'scores': auc_scores.tolist()
            }
        }
        
        return cv_results
        
    except Exception as e:
        raise Exception(f"Error in cross-validation for {model_type}: {str(e)}")


def hyperparameter_tuning(X, y, model_type, method="Grid Search", cv_folds=5):
    """
    Perform hyperparameter tuning on a model.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_type : str
        Type of model to tune
    method : str
        Tuning method: "Grid Search", "Random Search", or "Bayesian Optimization"
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    tuning_results : dict
        Dictionary containing best parameters and scores
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Define parameter grids for each model type
        if model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [100, 200, 500]
            }
            param_distributions = {
                'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                'max_iter': [100, 200, 300, 500]
            }
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 10, -1],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
            param_grid = {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'iterations': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'depth': [3, 4, 5, 6, 7, 8],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Choose search method
        if method == "Grid Search":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
        elif method == "Random Search":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=20,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        elif method == "Bayesian Optimization":
            # For Bayesian, we use RandomizedSearchCV as a fallback
            # (full Bayesian requires optuna or scikit-optimize)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=30,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported tuning method: {method}")
        
        # Fit the search
        search.fit(X, y)
        
        # Get results
        cv_results_df = pd.DataFrame(search.cv_results_)
        
        # Top 5 results
        top_results = cv_results_df.nsmallest(5, 'rank_test_score')[[
            'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
        ]].to_dict('records')
        
        tuning_results = {
            'method': method,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'top_results': top_results,
            'total_fits': len(cv_results_df)
        }
        
        return tuning_results
        
    except Exception as e:
        raise Exception(f"Error in hyperparameter tuning for {model_type}: {str(e)}")
