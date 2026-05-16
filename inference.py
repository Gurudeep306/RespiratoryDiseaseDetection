"""
Inference module for making predictions on new audio files.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from feature_extraction import AudioFeatureExtractor
from model_training import RandomForestModel, SVMModel, XGBoostModel, EnsembleModel


class AudioPredictor:
    """Predicts disease status from audio files."""
    
    def __init__(self, model_dir):
        """
        Initialize predictor.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.feature_extractor = AudioFeatureExtractor()
        self.ensemble = EnsembleModel()
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        try:
            self.ensemble.load(str(self.model_dir))
            print("✓ Ensemble model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ensemble model: {e}")
            print("  Loading individual models...")
            self.load_individual_models()
    
    def load_individual_models(self):
        """Load individual models if ensemble fails."""
        models = {
            'rf': RandomForestModel(),
            'svm': SVMModel(),
            'xgb': XGBoostModel(),
        }
        
        for name, model in models.items():
            try:
                model.load(str(self.model_dir / f"{name}_model.pkl"))
                self.ensemble.models[name] = model
            except:
                print(f"Warning: Could not load {name} model")
    
    def predict_audio(self, audio_path, return_features=False):
        """
        Predict disease status from audio file.
        
        Args:
            audio_path (str): Path to audio file
            return_features (bool): Whether to return extracted features
            
        Returns:
            dict: Prediction results
        """
        print(f"\nProcessing: {audio_path}")
        
        # Extract features
        print("  Extracting features...")
        features = self.feature_extractor.extract_all_features(audio_path)
        
        if not features:
            return {
                'file': audio_path,
                'error': 'Failed to extract features',
                'prediction': None,
                'confidence': None
            }
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        try:
            prediction = self.ensemble.predict(X)[0]
            confidence = self.ensemble.predict_proba(X)[0]
        except Exception as e:
            return {
                'file': audio_path,
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': None
            }
        
        result = {
            'file': Path(audio_path).name,
            'prediction': 'PD (Parkinson\'s)' if prediction == 1 else 'HC (Healthy)',
            'confidence': f"{max(confidence, 1-confidence)*100:.2f}%",
            'pd_probability': f"{confidence*100:.2f}%",
            'hc_probability': f"{(1-confidence)*100:.2f}%"
        }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def batch_predict(self, audio_dir, output_csv=None, recursive=False):
        """
        Predict disease status for multiple audio files.
        
        Args:
            audio_dir (str): Directory containing audio files
            output_csv (str): Path to save results CSV (optional)
            recursive (bool): Search subdirectories
            
        Returns:
            pd.DataFrame: Predictions for all files
        """
        audio_dir = Path(audio_dir)
        
        # Find audio files
        if recursive:
            audio_files = list(audio_dir.glob("**/*.wav")) + \
                         list(audio_dir.glob("**/*.mp3"))
        else:
            audio_files = list(audio_dir.glob("*.wav")) + \
                         list(audio_dir.glob("*.mp3"))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return pd.DataFrame()
        
        print(f"\nFound {len(audio_files)} audio files")
        
        results = []
        for audio_file in audio_files:
            result = self.predict_audio(str(audio_file))
            results.append(result)
        
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        
        return df
    
    def print_results(self, results_df):
        """Print results in a formatted table."""
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Summary statistics
        if 'prediction' in results_df.columns:
            pd_count = (results_df['prediction'] == 'PD (Parkinson\'s)').sum()
            hc_count = (results_df['prediction'] == 'HC (Healthy)').sum()
            
            print(f"\nSummary:")
            print(f"  Total samples: {len(results_df)}")
            print(f"  PD predictions: {pd_count} ({pd_count/len(results_df)*100:.1f}%)")
            print(f"  HC predictions: {hc_count} ({hc_count/len(results_df)*100:.1f}%)")


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(
        description="Make predictions on respiratory disease status"
    )
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--audio-dir', type=str, help='Directory with multiple audio files')
    parser.add_argument('--model-dir', type=str, default='results/models',
                       help='Directory containing trained models')
    parser.add_argument('--output', type=str, help='Path to save results CSV')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories for audio files')
    
    args = parser.parse_args()
    
    if not args.audio and not args.audio_dir:
        print("Error: Provide either --audio or --audio-dir")
        print("\nUsage examples:")
        print("  python inference.py --audio path/to/audio.wav --model-dir results/models")
        print("  python inference.py --audio-dir path/to/audio/dir --model-dir results/models")
        return
    
    print("\n" + "="*60)
    print("RESPIRATORY DISEASE DETECTION - INFERENCE")
    print("="*60 + "\n")
    
    # Initialize predictor
    try:
        predictor = AudioPredictor(args.model_dir)
    except Exception as e:
        print(f"Error loading models: {e}")
        print(f"Make sure models are saved in: {args.model_dir}")
        return
    
    # Make predictions
    if args.audio:
        if not Path(args.audio).exists():
            print(f"Error: Audio file not found: {args.audio}")
            return
        
        result = predictor.predict_audio(args.audio)
        result_df = pd.DataFrame([result])
        predictor.print_results(result_df)
        
        if args.output:
            result_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    elif args.audio_dir:
        if not Path(args.audio_dir).exists():
            print(f"Error: Directory not found: {args.audio_dir}")
            return
        
        results_df = predictor.batch_predict(args.audio_dir, args.output, args.recursive)
        predictor.print_results(results_df)


if __name__ == "__main__":
    main()
