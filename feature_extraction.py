"""
Feature extraction module for audio analysis in respiratory disease detection.
Extracts statistical, spectral, and wavelet features from audio files.
"""

import os
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
import pywt
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extracts audio features for disease detection."""
    
    def __init__(self, sr=22050):
        """
        Initialize feature extractor.
        
        Args:
            sr (int): Sample rate for audio loading
        """
        self.sr = sr
    
    @staticmethod
    def compute_statistics(feature):
        """
        Compute statistical features.
        
        Args:
            feature (np.array): Input feature vector
            
        Returns:
            dict: Statistical measures
        """
        flat_feature = np.nan_to_num(np.asarray(feature).flatten())
        
        if len(flat_feature) == 0:
            return {k: 0.0 for k in ["mean", "std", "skew", "kurtosis", "min", "max", "ptp"]}
        
        return {
            "mean": float(np.mean(flat_feature)),
            "std": float(np.std(flat_feature)),
            "skew": float(skew(flat_feature)),
            "kurtosis": float(kurtosis(flat_feature)),
            "min": float(np.min(flat_feature)),
            "max": float(np.max(flat_feature)),
            "ptp": float(np.ptp(flat_feature))
        }
    
    def extract_mfcc_features(self, audio_path, n_mfcc=13):
        """Extract MFCC (Mel-frequency cepstral coefficient) features."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except:
            return {}
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        stats = {}
        
        for i, coeff in enumerate(mfcc):
            coeff_stats = self.compute_statistics(coeff)
            for stat_name, stat_value in coeff_stats.items():
                stats[f"mfcc_{i}_{stat_name}"] = stat_value
        
        return stats
    
    def extract_spectral_features(self, audio_path):
        """Extract spectral features."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except:
            return {}
        
        stats = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_stats = self.compute_statistics(spectral_centroid)
        for stat_name, stat_value in sc_stats.items():
            stats[f"spectral_centroid_{stat_name}"] = stat_value
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        sr_stats = self.compute_statistics(spectral_rolloff)
        for stat_name, stat_value in sr_stats.items():
            stats[f"spectral_rolloff_{stat_name}"] = stat_value
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_stats = self.compute_statistics(zcr)
        for stat_name, stat_value in zcr_stats.items():
            stats[f"zcr_{stat_name}"] = stat_value
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_stats = self.compute_statistics(rms)
        for stat_name, stat_value in rms_stats.items():
            stats[f"rms_{stat_name}"] = stat_value
        
        return stats
    
    def extract_wavelet_features(self, audio_path, wavelet='db4', level=3):
        """Extract wavelet transform features."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except:
            return {}
        
        stats = {}
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        for i, coeff in enumerate(coeffs):
            coeff_stats = self.compute_statistics(coeff)
            prefix = "wav_approx" if i == 0 else f"wav_detail_{i-1}"
            for stat_name, stat_value in coeff_stats.items():
                stats[f"{prefix}_{stat_name}"] = stat_value
        
        return stats
    
    def extract_temporal_features(self, audio_path):
        """Extract temporal features."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except:
            return {}
        
        stats = {}
        
        # Temporal statistics
        temp_stats = self.compute_statistics(y)
        for stat_name, stat_value in temp_stats.items():
            stats[f"temporal_{stat_name}"] = stat_value
        
        return stats
    
    def extract_all_features(self, audio_path):
        """
        Extract all features from audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: All extracted features
        """
        features = {}
        features.update(self.extract_mfcc_features(audio_path))
        features.update(self.extract_spectral_features(audio_path))
        features.update(self.extract_wavelet_features(audio_path))
        features.update(self.extract_temporal_features(audio_path))
        
        return features
    
    def batch_extract(self, input_dir, output_csv=None):
        """
        Extract features from all audio files in directory.
        
        Args:
            input_dir (str): Directory containing audio files
            output_csv (str): Path to save CSV output (optional)
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        audio_files = list(Path(input_dir).glob("*.wav")) + \
                     list(Path(input_dir).glob("*.mp3"))
        
        all_features = []
        
        for audio_file in audio_files:
            print(f"Processing {audio_file.name}...")
            features = self.extract_all_features(str(audio_file))
            features['filename'] = audio_file.name
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Features saved to {output_csv}")
        
        return df
