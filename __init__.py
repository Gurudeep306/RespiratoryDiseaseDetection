"""
Respiratory Disease Detection Package
A comprehensive machine learning pipeline for disease detection from audio.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Import main modules
from audio_preprocessing import AudioPreprocessor
from feature_extraction import AudioFeatureExtractor
from data_preparation import DataPreparation, DataLoader
from model_training import (
    RandomForestModel,
    SVMModel,
    XGBoostModel,
    LSTMModel,
    EnsembleModel,
    ModelEvaluator
)
from evaluate import (
    EvaluationVisualizer,
    ComprehensiveEvaluation,
    PerformanceAnalyzer
)

__all__ = [
    'AudioPreprocessor',
    'AudioFeatureExtractor',
    'DataPreparation',
    'DataLoader',
    'RandomForestModel',
    'SVMModel',
    'XGBoostModel',
    'LSTMModel',
    'EnsembleModel',
    'ModelEvaluator',
    'EvaluationVisualizer',
    'ComprehensiveEvaluation',
    'PerformanceAnalyzer'
]
