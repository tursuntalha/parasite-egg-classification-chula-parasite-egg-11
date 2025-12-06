# -*- coding: utf-8 -*-
"""
Makine öğrenmesi modelleri - RandomForest, XGBoost, SVM, LogisticRegression
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from config import ML_CONFIG, MODELS_DIR


def train_random_forest(X_train, y_train, params: dict = None):
    """RandomForest modeli eğit."""
    if params is None:
        params = ML_CONFIG["random_forest"]
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params: dict = None):
    """XGBoost modeli eğit."""
    if params is None:
        params = ML_CONFIG["xgboost"]
    
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, params: dict = None):
    """SVM modeli eğit (olasılık çıktısı ile)."""
    if params is None:
        params = ML_CONFIG["svm"]
    
    model = SVC(**params, probability=True)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, params: dict = None):
    """Logistic Regression modeli eğit."""
    if params is None:
        params = ML_CONFIG["logistic_regression"]
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def grid_search_ml(model_type: str, X_train, y_train, cv: int = 5):
    """
    Grid Search ile hiperparametre optimizasyonu.
    
    Args:
        model_type: 'random_forest', 'xgboost', 'svm', 'logistic_regression'
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 50],
            'criterion': ['gini', 'entropy']
        },
        'xgboost': {
            'max_depth': [3, 6, 9],
            'n_estimators': [150, 200, 300],
            'subsample': [0.6, 0.8, 1.0]
        },
        'svm': {
            'kernel': ['linear', 'rbf'],
            'C': [0.5, 1.0, 2.0],
            'gamma': ['auto', 'scale']
        },
        'logistic_regression': {
            'penalty': ['l1', 'l2'],
            'solver': ['lbfgs', 'liblinear'],
            'C': [0.1, 1.0, 10.0]
        }
    }
    
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'svm': SVC(probability=True),
        'logistic_regression': LogisticRegression(max_iter=1000)
    }
    
    model = models[model_type]
    param_grid = param_grids[model_type]
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"En iyi parametreler ({model_type}): {grid_search.best_params_}")
    print(f"En iyi skor: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_all_ml_models(X_train, y_train, use_grid_search: bool = False):
    """Tüm ML modellerini eğit."""
    models = {}
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if use_grid_search:
        models['random_forest'], _ = grid_search_ml('random_forest', X_train_scaled, y_train)
        models['xgboost'], _ = grid_search_ml('xgboost', X_train_scaled, y_train)
        models['svm'], _ = grid_search_ml('svm', X_train_scaled, y_train)
        models['logistic_regression'], _ = grid_search_ml('logistic_regression', X_train_scaled, y_train)
    else:
        models['random_forest'] = train_random_forest(X_train_scaled, y_train)
        models['xgboost'] = train_xgboost(X_train_scaled, y_train)
        models['svm'] = train_svm(X_train_scaled, y_train)
        models['logistic_regression'] = train_logistic_regression(X_train_scaled, y_train)
    
    models['scaler'] = scaler
    return models


def save_ml_model(model, model_name: str):
    """ML modelini kaydet."""
    filepath = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, filepath)
    print(f"Model kaydedildi: {filepath}")


def load_ml_model(model_name: str):
    """ML modelini yükle."""
    filepath = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    return joblib.load(filepath)


def predict_proba_ml(model, X, scaler=None):
    """ML modeli ile olasılık tahmini yap."""
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)
