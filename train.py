"""
Main training script for respiratory disease detection.
Orchestrates the complete pipeline from data preparation to model evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import modules
from data_preparation import DataPreparation, DataLoader, create_sample_data
from feature_extraction import AudioFeatureExtractor
from model_training import (
    RandomForestModel, LSTMModel, SVMModel, XGBoostModel,
    EnsembleModel, ModelEvaluator
)


def setup_directories(base_dir=None):
    """Create necessary project directories."""
    if base_dir is None:
        base_dir = Path.cwd()
    
    dirs = {
        'data': base_dir / 'data',
        'results': base_dir / 'results',
        'models': base_dir / 'results' / 'models',
        'plots': base_dir / 'results' / 'plots'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC', 'PD'],
                yticklabels=['HC', 'PD'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(all_metrics, save_path=None):
    """Plot metrics comparison across models."""
    metrics_list = []
    
    for model_name, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                metrics_list.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
    
    df = pd.DataFrame(metrics_list)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score (%)')
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_individual_models(X_train, X_test, y_train, y_test, dirs):
    """Train individual models and evaluate."""
    all_metrics = {}
    
    models = {
        'Random Forest': RandomForestModel(),
        'SVM': SVMModel(),
        'XGBoost': XGBoostModel(),
    }
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        model.train(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)
        
        metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
        all_metrics[model_name] = metrics
        
        ModelEvaluator.print_metrics(metrics, model_name)
        
        # Save confusion matrix plot
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            model_name,
            dirs['plots'] / f"{model_name.lower().replace(' ', '_')}_cm.png"
        )
        
        # Save model
        model.save(dirs['models'] / f"{model_name.lower().replace(' ', '_')}_model.pkl")
    
    return all_metrics


def train_lstm_model(X_train, X_test, y_train, y_test, dirs):
    """Train LSTM model."""
    print(f"\n{'='*60}")
    print("Training LSTM...")
    print(f"{'='*60}")
    
    lstm = LSTMModel(epochs=50, batch_size=32)
    lstm.build(X_train.shape[1])
    lstm.train(X_train, y_train)
    
    y_pred = lstm.predict(X_test)
    y_probs = lstm.predict_proba(X_test)
    
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    ModelEvaluator.print_metrics(metrics, "LSTM")
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        "LSTM",
        dirs['plots'] / "lstm_cm.png"
    )
    
    lstm.save(dirs['models'] / "lstm_model.pkl")
    
    return {'LSTM': metrics}


def train_ensemble_model(X_train, X_test, y_train, y_test, dirs):
    """Train ensemble model."""
    print(f"\n{'='*60}")
    print("Training Ensemble Model...")
    print(f"{'='*60}")
    
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train, optimize_xgb=False)
    
    y_pred = ensemble.predict(X_test)
    y_probs = ensemble.predict_proba(X_test)
    
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    ModelEvaluator.print_metrics(metrics, "Ensemble")
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        "Ensemble",
        dirs['plots'] / "ensemble_cm.png"
    )
    
    ensemble.save(dirs['models'])
    
    return {'Ensemble': metrics}


def save_results_summary(all_metrics, filepath):
    """Save metrics summary to file."""
    summary = []
    
    for model_name, metrics in all_metrics.items():
        row = {'Model': model_name}
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                row[metric_name] = f"{value:.2f}%"
        summary.append(row)
    
    df = pd.DataFrame(summary)
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")
    print("\n" + df.to_string(index=False))


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train respiratory disease detection models")
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--dataset-type', type=str, default='sample',
                       choices=['sample', 'csv'],
                       help='Type of dataset to use')
    parser.add_argument('--train-lstm', action='store_true',
                       help='Include LSTM model in training')
    parser.add_argument('--train-ensemble', action='store_true',
                       help='Train ensemble model')
    parser.add_argument('--optimize-xgb', action='store_true',
                       help='Optimize XGBoost hyperparameters')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RESPIRATORY DISEASE DETECTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    # Setup directories
    dirs = setup_directories()
    print(f"Project directories created: {dirs['results']}")
    
    # Load or create data
    print("\nLoading data...")
    
    if args.dataset_type == 'sample':
        X, y = create_sample_data(n_samples=1000, n_features=100)
        print("Using sample data (1000 samples, 100 features)")
    else:
        # Try to load from CSV
        loader = DataLoader(args.data_dir)
        try:
            X = loader.load_csv('features.csv')
            y = X['Label'].astype(int)
            X = X.drop(columns=['Label', 'Filename'], errors='ignore')
            print(f"Loaded data from CSV: {X.shape[0]} samples, {X.shape[1]} features")
        except FileNotFoundError:
            print("CSV files not found, using sample data")
            X, y = create_sample_data()
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = DataPreparation.prepare_for_training(
        pd.concat([X, y], axis=1).rename(columns={0: 'Label'}) if isinstance(y, pd.Series) else X,
        test_size=0.3
    )
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    all_metrics = {}
    
    # Individual models
    individual_metrics = train_individual_models(X_train, X_test, y_train, y_test, dirs)
    all_metrics.update(individual_metrics)
    
    # Optional: LSTM model
    if args.train_lstm:
        lstm_metrics = train_lstm_model(X_train, X_test, y_train, y_test, dirs)
        all_metrics.update(lstm_metrics)
    
    # Optional: Ensemble model
    if args.train_ensemble:
        ensemble_metrics = train_ensemble_model(X_train, X_test, y_train, y_test, dirs)
        all_metrics.update(ensemble_metrics)
    
    # Save results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    results_file = dirs['results'] / 'model_results.csv'
    save_results_summary(all_metrics, results_file)
    
    # Plot comparison
    plot_metrics_comparison(all_metrics, dirs['plots'] / 'metrics_comparison.png')
    print(f"\nMetrics comparison plot saved to {dirs['plots'] / 'metrics_comparison.png'}")
    
    # Best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.2f}%)")
    
    print(f"\n✓ Training complete! Results saved to {dirs['results']}")
    print(f"  - Models: {dirs['models']}")
    print(f"  - Plots: {dirs['plots']}")
    print(f"  - Results: {results_file}")


if __name__ == "__main__":
    main()
