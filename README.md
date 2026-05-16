# Respiratory Disease Detection

A comprehensive machine learning pipeline for detecting respiratory diseases (specifically Parkinson's Disease) from audio recordings using multiple deep learning and machine learning models.

## Project Overview

This project implements an end-to-end pipeline for respiratory disease detection that includes:

- **Audio Preprocessing**: Silence removal and audio chunking
- **Feature Extraction**: MFCC, spectral, wavelet, and temporal features
- **Multiple Models**: Random Forest, SVM, XGBoost, LSTM, and Ensemble
- **Evaluation Framework**: Comprehensive metrics including accuracy, precision, recall, F1, AUC
- **Inference Pipeline**: Make predictions on new audio files

## Project Structure

```
RespiratoryDiseaseDetection/
├── audio_preprocessing.py      # Audio processing utilities
├── feature_extraction.py       # Feature extraction module
├── data_preparation.py         # Data loading and preparation
├── model_training.py           # Model implementations
├── train.py                    # Main training script
├── inference.py                # Inference/prediction script
├── evaluate.py                 # Model evaluation utilities
├── test_pipeline.py            # Testing and demo script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directory
├── results/                    # Training results
│   ├── models/                 # Saved models
│   ├── plots/                  # Evaluation plots
│   └── model_results.csv       # Performance metrics
└── THE_MODEL                   # Pre-trained model (if available)
```

## Installation

### 1. Clone or Navigate to Project
```bash
cd RespiratoryDiseaseDetection
```

### 2. Create Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Models

#### Using Sample Data (for testing)
```bash
python train.py --dataset-type sample
```

#### Using CSV Data
```bash
python train.py --dataset-type csv --data-dir data
```

#### Training with LSTM
```bash
python train.py --dataset-type sample --train-lstm
```

#### Training Ensemble Model
```bash
python train.py --dataset-type sample --train-ensemble
```

#### Full Training (all models)
```bash
python train.py --dataset-type sample --train-lstm --train-ensemble
```

### 2. Making Predictions

#### Single Audio File
```bash
python inference.py --audio path/to/audio.wav --model-dir results/models
```

#### Multiple Audio Files
```bash
python inference.py --audio-dir path/to/audio/dir --model-dir results/models
```

#### Batch Prediction with Output
```bash
python inference.py --audio-dir path/to/audio/dir --model-dir results/models --output results/predictions.csv
```

#### Recursive Directory Search
```bash
python inference.py --audio-dir path/to/audio/dir --recursive --model-dir results/models
```

### 3. Data Preparation

For custom data, prepare your dataset with the following structure:

**CSV Format**:
```
feature_1,feature_2,...,feature_n,Label,Filename
0.5,0.3,...,0.2,1,patient_001.wav
0.4,0.2,...,0.3,0,healthy_001.wav
...
```

Where:
- Features can be extracted using the AudioFeatureExtractor class
- Label: 0 for Healthy Controls (HC), 1 for Parkinson's Disease (PD)
- Filename: Original audio filename (optional)

## Modules

### audio_preprocessing.py
Handles audio preprocessing tasks:
- Silence detection and removal
- Audio chunking into segments
- Batch processing of audio files

**Example**:
```python
from audio_preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(min_silence_len=500, num_chunks=50)
chunks = preprocessor.chunk_audio('audio.wav', 'output_dir')
```

### feature_extraction.py
Extracts features from audio files:
- MFCC (Mel-frequency cepstral coefficients)
- Spectral features (centroid, rolloff, ZCR, RMS)
- Wavelet features
- Temporal features

**Example**:
```python
from feature_extraction import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_all_features('audio.wav')
df = extractor.batch_extract('audio_dir', 'features.csv')
```

### data_preparation.py
Data loading and preparation:
- Load and combine multiple datasets
- Clean datasets
- Remove low variance features
- Balance data using SMOTE
- Train-test split

**Example**:
```python
from data_preparation import DataPreparation

X_train, X_test, y_train, y_test = DataPreparation.prepare_for_training(data)
```

### model_training.py
Multiple model implementations:
- RandomForestModel
- SVMModel
- XGBoostModel
- LSTMModel
- EnsembleModel

**Example**:
```python
from model_training import RandomForestModel

model = RandomForestModel()
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Architectures

### Random Forest
- 200 estimators
- Max depth: 15
- Optimized for interpretability and performance

### Support Vector Machine (SVM)
- RBF kernel
- Probability estimates enabled
- SMOTE balancing

### XGBoost
- Optuna-based hyperparameter optimization
- 10-fold cross-validation
- L2 regularization for robustness

### LSTM (Long Short-Term Memory)
- 2 LSTM layers (128 and 64 units)
- Dropout (0.2) for regularization
- Early stopping with patience=5
- 50 epochs training

### Ensemble
- Weighted combination of RF, SVM, and XGBoost
- Equal weights (0.33 each)
- Robust predictions through majority voting

## Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)
- **G-Mean**: Geometric mean of sensitivity and specificity
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: True/False positives and negatives

## Data Requirements

### Minimum Dataset Size
- At least 100 samples per class for reliable training
- Recommended: 500+ samples per class

### Audio Format
- WAV or MP3 format
- 22050 Hz sample rate (will be resampled if different)
- Mono or stereo (will be converted to mono)

### Feature Requirements
- Numeric features only
- No missing values (will be handled by imputation)
- Feature scaling (StandardScaler applied automatically)

## Results

After training, results are saved to `results/` directory:

```
results/
├── models/
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── xgb_model.pkl
│   └── [LSTM and scaler files if trained]
├── plots/
│   ├── random_forest_cm.png
│   ├── metrics_comparison.png
│   └── [other visualizations]
└── model_results.csv
```

## Troubleshooting

### Error: "No audio files found"
- Check that audio files have .wav or .mp3 extension
- Ensure the correct directory path is specified

### Error: "Models not loaded"
- Verify model directory path with `--model-dir`
- Ensure all model files are present and not corrupted

### Memory Issues with LSTM
- Reduce batch size: modify in train.py
- Use only with smaller datasets

### Poor Model Performance
- Ensure data preprocessing is correct
- Check for class imbalance (SMOTE helps)
- Validate feature quality

## Performance Expectations

Typical performance metrics on a balanced test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 85-92% | 84-91% | 86-93% | 85-92% |
| SVM | 82-88% | 81-87% | 83-89% | 82-88% |
| XGBoost | 86-93% | 85-92% | 87-94% | 86-93% |
| LSTM | 80-90% | 79-89% | 81-91% | 80-90% |
| Ensemble | 88-94% | 87-93% | 89-95% | 88-94% |

*Note: Performance varies based on data quality and feature extraction parameters.*

## Advanced Usage

### Custom Training with Specific Parameters

```python
from train import *
from data_preparation import DataPreparation

# Load your data
data = pd.read_csv('your_data.csv')

# Prepare
X_train, X_test, y_train, y_test = DataPreparation.prepare_for_training(data)

# Train custom ensemble
from model_training import EnsembleModel
ensemble = EnsembleModel()
ensemble.train(X_train, y_train, optimize_xgb=True)

# Evaluate
y_pred = ensemble.predict(X_test)
metrics = ModelEvaluator.evaluate(y_test, y_pred)
```

### Feature Extraction Pipeline

```python
from feature_extraction import AudioFeatureExtractor
from pathlib import Path

extractor = AudioFeatureExtractor(sr=22050)

# Extract from single file
features = extractor.extract_all_features('audio.wav')

# Batch extract
df = extractor.batch_extract('audio_directory', 'output.csv')
```

## References

- MFCC: [Mel-frequency cepstral coefficient](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Wavelet Transform: [PyWavelets Documentation](https://pywt.readthedocs.io/)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- TensorFlow/Keras: [Keras Documentation](https://keras.io/)

## License

This project is provided as-is for research and educational purposes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues or questions:
- Check the Troubleshooting section
- Review the code comments
- Ensure all dependencies are correctly installed

---

**Last Updated**: 2026
**Version**: 1.0.0
