"""
Test and demonstration script for the Respiratory Disease Detection project.
Shows how to use all components of the pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from data_preparation import DataPreparation, DataLoader, create_sample_data
from feature_extraction import AudioFeatureExtractor
from model_training import (
    RandomForestModel, SVMModel, XGBoostModel, LSTMModel,
    EnsembleModel, ModelEvaluator
)
from evaluate import ComprehensiveEvaluation, PerformanceAnalyzer


def test_data_preparation():
    """Test data preparation module."""
    print("\n" + "="*60)
    print("TEST: Data Preparation Module")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y = create_sample_data(n_samples=500, n_features=50)
    print(f"   ✓ Created {X.shape[0]} samples with {X.shape[1]} features")
    
    # Combine X and y
    data = X.copy()
    data['Label'] = y
    
    # Test low variance removal
    print("\n2. Testing low variance feature removal...")
    X_filtered = DataPreparation.remove_low_variance_features(X, threshold=0.01)
    print(f"   ✓ Removed {X.shape[1] - X_filtered.shape[1]} low variance features")
    
    # Test train-test split
    print("\n3. Testing train-test split...")
    X_train, X_test, y_train, y_test = DataPreparation.prepare_for_training(data)
    print(f"   ✓ Train set: {X_train.shape}")
    print(f"   ✓ Test set: {X_test.shape}")
    
    # Test data balancing
    print("\n4. Testing SMOTE balancing...")
    X_balanced, y_balanced = DataPreparation.balance_data(X_train, y_train)
    print(f"   ✓ Balanced train set: {X_balanced.shape}")
    print(f"   ✓ Original class distribution: {np.bincount(y_train)}")
    print(f"   ✓ Balanced class distribution: {np.bincount(y_balanced)}")
    
    print("\n✓ Data Preparation tests passed!")
    return X_train, X_test, y_train, y_test


def test_model_training(X_train, X_test, y_train, y_test):
    """Test model training and evaluation."""
    print("\n" + "="*60)
    print("TEST: Model Training and Evaluation")
    print("="*60)
    
    # Balance data for training
    X_train_balanced, y_train_balanced = DataPreparation.balance_data(X_train, y_train)
    
    all_metrics = {}
    
    # Test Random Forest
    print("\n1. Testing Random Forest Model...")
    rf_model = RandomForestModel()
    rf_model.train(X_train_balanced, y_train_balanced)
    y_pred = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    all_metrics['Random Forest'] = metrics
    print(f"   ✓ Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   ✓ AUC: {metrics.get('auc', 'N/A')}")
    
    # Test SVM
    print("\n2. Testing SVM Model...")
    svm_model = SVMModel()
    svm_model.train(X_train_balanced, y_train_balanced)
    y_pred = svm_model.predict(X_test)
    y_probs = svm_model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    all_metrics['SVM'] = metrics
    print(f"   ✓ Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   ✓ AUC: {metrics.get('auc', 'N/A')}")
    
    # Test XGBoost
    print("\n3. Testing XGBoost Model...")
    xgb_model = XGBoostModel()
    xgb_model.train(X_train_balanced, y_train_balanced, optimize=False)
    y_pred = xgb_model.predict(X_test)
    y_probs = xgb_model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    all_metrics['XGBoost'] = metrics
    print(f"   ✓ Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   ✓ AUC: {metrics.get('auc', 'N/A')}")
    
    # Test Ensemble
    print("\n4. Testing Ensemble Model...")
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train, optimize_xgb=False)
    y_pred = ensemble.predict(X_test)
    y_probs = ensemble.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    all_metrics['Ensemble'] = metrics
    print(f"   ✓ Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   ✓ AUC: {metrics.get('auc', 'N/A')}")
    
    # Test LSTM (optional - takes longer)
    print("\n5. Testing LSTM Model (reduced epochs)...")
    lstm = LSTMModel(epochs=10, batch_size=32)
    lstm.build(X_train.shape[1])
    lstm.train(X_train_balanced, y_train_balanced)
    y_pred = lstm.predict(X_test)
    y_probs = lstm.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_probs)
    all_metrics['LSTM'] = metrics
    print(f"   ✓ Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   ✓ AUC: {metrics.get('auc', 'N/A')}")
    
    print("\n✓ Model Training tests passed!")
    return all_metrics


def test_model_comparison(all_metrics):
    """Test model comparison and evaluation."""
    print("\n" + "="*60)
    print("TEST: Model Comparison and Analysis")
    print("="*60)
    
    # Compare models
    print("\n1. Comparing models...")
    comparison = ComprehensiveEvaluation.cross_model_comparison(all_metrics)
    
    # Find best model
    best_model = comparison.iloc[0]['Model']
    best_acc = comparison.iloc[0]['accuracy']
    print(f"\n✓ Best Model: {best_model} (Accuracy: {best_acc:.2f}%)")


def test_model_persistence(X_train, y_train, X_test, y_test):
    """Test model saving and loading."""
    print("\n" + "="*60)
    print("TEST: Model Persistence (Save/Load)")
    print("="*60)
    
    # Create temp directory for testing
    test_dir = Path('test_models')
    test_dir.mkdir(exist_ok=True)
    
    X_train_balanced, y_train_balanced = DataPreparation.balance_data(X_train, y_train)
    
    try:
        # Test Random Forest persistence
        print("\n1. Testing Random Forest save/load...")
        rf = RandomForestModel()
        rf.train(X_train_balanced, y_train_balanced)
        y_pred_before = rf.predict(X_test)
        
        rf.save(test_dir / 'rf_model.pkl')
        print("   ✓ Model saved")
        
        rf_loaded = RandomForestModel()
        rf_loaded.load(test_dir / 'rf_model.pkl')
        y_pred_after = rf_loaded.predict(X_test)
        
        assert np.array_equal(y_pred_before, y_pred_after), "Predictions don't match after loading!"
        print("   ✓ Model loaded and predictions match")
        
        # Test Ensemble persistence
        print("\n2. Testing Ensemble save/load...")
        ensemble = EnsembleModel()
        ensemble.train(X_train, y_train, optimize_xgb=False)
        y_pred_before = ensemble.predict(X_test)
        
        ensemble.save(test_dir)
        print("   ✓ Ensemble saved")
        
        ensemble_loaded = EnsembleModel()
        ensemble_loaded.load(test_dir)
        y_pred_after = ensemble_loaded.predict(X_test)
        
        assert np.array_equal(y_pred_before, y_pred_after), "Predictions don't match after loading!"
        print("   ✓ Ensemble loaded and predictions match")
        
        print("\n✓ Model persistence tests passed!")
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_performance_analysis(y_test, y_probs):
    """Test performance analysis utilities."""
    print("\n" + "="*60)
    print("TEST: Performance Analysis")
    print("="*60)
    
    # Analyze prediction confidence
    print("\n1. Analyzing prediction confidence...")
    confidence_analysis = PerformanceAnalyzer.analyze_prediction_confidence(y_test, y_probs)
    for key, value in confidence_analysis.items():
        print(f"   {key}: {value:.4f}")
    
    # Analyze threshold effect
    print("\n2. Analyzing classification threshold effects...")
    threshold_df = PerformanceAnalyzer.analyze_threshold_effect(y_test, y_probs)
    print(threshold_df.to_string(index=False))
    
    print("\n✓ Performance analysis tests passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RESPIRATORY DISEASE DETECTION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Data Preparation
        X_train, X_test, y_train, y_test = test_data_preparation()
        
        # Test 2: Model Training
        all_metrics = test_model_training(X_train, X_test, y_train, y_test)
        
        # Test 3: Model Comparison
        test_model_comparison(all_metrics)
        
        # Test 4: Model Persistence
        test_model_persistence(X_train, y_train, X_test, y_test)
        
        # Test 5: Performance Analysis
        # Use SVM predictions for confidence analysis
        svm = SVMModel()
        X_train_balanced, y_train_balanced = DataPreparation.balance_data(X_train, y_train)
        svm.train(X_train_balanced, y_train_balanced)
        y_probs = svm.predict_proba(X_test)
        test_performance_analysis(y_test, y_probs)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print("\nThe project is ready for use. Next steps:")
        print("1. Prepare your audio data using audio_preprocessing.py")
        print("2. Extract features using feature_extraction.py")
        print("3. Train models using: python train.py")
        print("4. Make predictions using: python inference.py")
        print("\nFor more information, see README.md")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
