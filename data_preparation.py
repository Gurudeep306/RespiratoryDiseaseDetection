"""
Data preparation and management module for the project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DataPreparation:
    """Handles data loading, cleaning, and preparation."""
    
    @staticmethod
    def load_and_combine_datasets(dataset_paths):
        """
        Load and combine multiple datasets.
        
        Args:
            dataset_paths (dict): Dictionary with 'pd' and 'hc' paths
            
        Returns:
            pd.DataFrame: Combined dataset with labels
        """
        dfs = []
        
        if 'pd' in dataset_paths and Path(dataset_paths['pd']).exists():
            pd_data = pd.read_csv(dataset_paths['pd'])
            pd_data['Label'] = 1  # Parkinson's Disease
            dfs.append(pd_data)
        
        if 'hc' in dataset_paths and Path(dataset_paths['hc']).exists():
            hc_data = pd.read_csv(dataset_paths['hc'])
            hc_data['Label'] = 0  # Healthy Controls
            dfs.append(hc_data)
        
        if not dfs:
            raise ValueError("No datasets found at provided paths")
        
        combined_data = pd.concat(dfs, ignore_index=True)
        return combined_data
    
    @staticmethod
    def clean_dataset(data):
        """
        Clean the dataset.
        
        Args:
            data (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        # Drop completely empty columns
        cleaned_data = data.dropna(axis=1, how='all')
        
        # Handle missing values
        for column in cleaned_data.columns:
            if column == 'Label' or column == 'Filename':
                continue
            
            if cleaned_data[column].isnull().any():
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                else:
                    try:
                        cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)
                    except:
                        cleaned_data[column].fillna(0, inplace=True)
        
        # Handle object columns that might contain string representations of arrays
        for column in cleaned_data.columns:
            if cleaned_data[column].dtype == 'object':
                try:
                    # Try to clean string representations like '[value]'
                    cleaned_data[column] = cleaned_data[column].astype(str)\
                        .str.replace('[', '', regex=False)\
                        .str.replace(']', '', regex=False)\
                        .astype(float)
                except:
                    # If can't convert, encode as category
                    try:
                        cleaned_data[column] = pd.factorize(cleaned_data[column])[0]
                    except:
                        cleaned_data[column] = 0
        
        return cleaned_data
    
    @staticmethod
    def remove_low_variance_features(X, threshold=0.01):
        """
        Remove features with low variance.
        
        Args:
            X (pd.DataFrame): Features
            threshold (float): Variance threshold
            
        Returns:
            pd.DataFrame: Features after removing low variance
        """
        variance = X.var()
        return X.loc[:, variance > threshold]
    
    @staticmethod
    def prepare_for_training(data, test_size=0.3, random_state=42, remove_low_var=True):
        """
        Prepare data for model training.
        
        Args:
            data (pd.DataFrame): Combined and cleaned dataset
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
            remove_low_var (bool): Whether to remove low variance features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Separate features and labels
        X = data.drop(columns=['Label'], errors='ignore')
        
        # Remove filename if present
        if 'Filename' in X.columns:
            X = X.drop(columns=['Filename'])
        
        y = data['Label']
        
        # Remove low variance features
        if remove_low_var:
            X = DataPreparation.remove_low_variance_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def balance_data(X_train, y_train, random_state=42):
        """
        Balance training data using SMOTE.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training labels
            random_state (int): Random state
            
        Returns:
            tuple: (X_train_balanced, y_train_balanced)
        """
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        return X_balanced, y_balanced
    
    @staticmethod
    def scale_features(X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            X_test (pd.DataFrame or np.ndarray): Test features (optional)
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, None, scaler


class DataLoader:
    """Loads and manages project data."""
    
    def __init__(self, data_dir=None):
        """
        Initialize data loader.
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir or Path.cwd()
    
    def load_csv(self, filename):
        """Load CSV file."""
        filepath = Path(self.data_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)
    
    def save_csv(self, df, filename):
        """Save DataFrame to CSV."""
        filepath = Path(self.data_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
    
    def find_csv_files(self, pattern="*.csv"):
        """Find CSV files in data directory."""
        return list(Path(self.data_dir).glob(pattern))


def create_sample_data(n_samples=1000, n_features=50):
    """
    Create sample data for testing.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        
    Returns:
        tuple: (X, y)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.5, n_samples)
    
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), pd.Series(y)
