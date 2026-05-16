"""
Unified model training and evaluation module for respiratory disease detection.
Includes Random Forest, LSTM, SVM, and XGBoost models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates model performance with various metrics."""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_probs=None):
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities (optional)
            
        Returns:
            dict: Evaluation metrics
        """
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred) * 100
        recall = recall_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred) * 100
        specificity = (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])) * 100
        g_mean = np.sqrt((recall / 100) * (specificity / 100)) * 100
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'g_mean': g_mean,
            'confusion_matrix': conf_matrix
        }
        
        if y_probs is not None:
            auc = roc_auc_score(y_true, y_probs) * 100
            metrics['auc'] = auc
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """Print evaluation metrics."""
        print(f"\n{'='*50}")
        print(f"{model_name} - Evaluation Metrics")
        print(f"{'='*50}")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key.upper()}: {value:.2f}%")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


class RandomForestModel:
    """Random Forest classifier for disease detection."""
    
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath):
        """Save model."""
        joblib.dump(self.model, filepath.replace('.pkl', '_rf.pkl'))
        joblib.dump(self.scaler, filepath.replace('.pkl', '_rf_scaler.pkl'))
    
    def load(self, filepath):
        """Load model."""
        self.model = joblib.load(filepath.replace('.pkl', '_rf.pkl'))
        self.scaler = joblib.load(filepath.replace('.pkl', '_rf_scaler.pkl'))


class LSTMModel:
    """LSTM neural network for sequence classification."""
    
    def __init__(self, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
    
    def build(self, input_shape):
        """Build LSTM model."""
        self.model = Sequential([
            LSTM(128, input_shape=(1, input_shape), activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='tanh'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, validation_split=0.2):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X_train)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.model.fit(
            X_reshaped, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def predict(self, X_test):
        """Make predictions."""
        X_scaled = self.scaler.transform(X_test)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        probs = self.model.predict(X_reshaped).flatten()
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X_test)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        return self.model.predict(X_reshaped).flatten()
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath.replace('.pkl', '_lstm.h5'))
        joblib.dump(self.scaler, filepath.replace('.pkl', '_lstm_scaler.pkl'))
    
    def load(self, filepath):
        """Load model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath.replace('.pkl', '_lstm.h5'))
        self.scaler = joblib.load(filepath.replace('.pkl', '_lstm_scaler.pkl'))


class SVMModel:
    """Support Vector Machine classifier."""
    
    def __init__(self, kernel='rbf', random_state=42):
        self.model = SVC(kernel=kernel, probability=True, random_state=random_state)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath):
        """Save model."""
        joblib.dump(self.model, filepath.replace('.pkl', '_svm.pkl'))
        joblib.dump(self.scaler, filepath.replace('.pkl', '_svm_scaler.pkl'))
    
    def load(self, filepath):
        """Load model."""
        self.model = joblib.load(filepath.replace('.pkl', '_svm.pkl'))
        self.scaler = joblib.load(filepath.replace('.pkl', '_svm_scaler.pkl'))


class XGBoostModel:
    """XGBoost classifier with hyperparameter optimization."""
    
    def __init__(self, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.random_state = random_state
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=50):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, valid_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[valid_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
                
                model = XGBClassifier(eval_metric='logloss', random_state=self.random_state, **params)
                model.fit(X_tr, y_tr, verbose=0)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def train(self, X_train, y_train, optimize=False, n_trials=50):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        if optimize:
            print("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X_scaled, y_train, n_trials)
            self.model = XGBClassifier(
                eval_metric='logloss',
                random_state=self.random_state,
                **best_params
            )
        else:
            self.model = XGBClassifier(eval_metric='logloss', random_state=self.random_state)
        
        self.model.fit(X_scaled, y_train, verbose=0)
    
    def predict(self, X_test):
        """Make predictions."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath):
        """Save model."""
        joblib.dump(self.model, filepath.replace('.pkl', '_xgb.pkl'))
        joblib.dump(self.scaler, filepath.replace('.pkl', '_xgb_scaler.pkl'))
    
    def load(self, filepath):
        """Load model."""
        self.model = joblib.load(filepath.replace('.pkl', '_xgb.pkl'))
        self.scaler = joblib.load(filepath.replace('.pkl', '_xgb_scaler.pkl'))


class EnsembleModel:
    """Ensemble model combining multiple models."""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestModel(),
            'svm': SVMModel(),
            'xgb': XGBoostModel(),
        }
        self.weights = {'rf': 0.33, 'svm': 0.33, 'xgb': 0.34}
    
    def train(self, X_train, y_train, optimize_xgb=False):
        """Train all models."""
        # Balance data
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        print("Training Random Forest...")
        self.models['rf'].train(X_balanced, y_balanced)
        
        print("Training SVM...")
        self.models['svm'].train(X_balanced, y_balanced)
        
        print("Training XGBoost...")
        self.models['xgb'].train(X_balanced, y_balanced, optimize=optimize_xgb)
    
    def predict(self, X_test):
        """Make ensemble predictions."""
        probs = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_test)
            probs.append(prob * self.weights[name])
        
        ensemble_prob = np.sum(probs, axis=0)
        return (ensemble_prob > 0.5).astype(int)
    
    def predict_proba(self, X_test):
        """Get ensemble prediction probabilities."""
        probs = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_test)
            probs.append(prob * self.weights[name])
        
        return np.sum(probs, axis=0)
    
    def save(self, directory):
        """Save all models."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model.save(os.path.join(directory, f"{name}_model.pkl"))
    
    def load(self, directory):
        """Load all models."""
        for name in self.models.keys():
            self.models[name].load(os.path.join(directory, f"{name}_model.pkl"))
