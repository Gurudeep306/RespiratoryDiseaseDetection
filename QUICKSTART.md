# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. **Install Dependencies** (2 minutes)

```bash
# Navigate to project directory
cd RespiratoryDiseaseDetection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. **Test the System** (1 minute)

```bash
# Verify everything works
python test_pipeline.py
```

Expected output:
```
✓ ALL TESTS PASSED SUCCESSFULLY!
```

### 3. **Train Models** (1 minute)

```bash
# Train with sample data (quick demo)
python train.py --dataset-type sample

# Or with your CSV data
python train.py --dataset-type csv --data-dir data
```

Models will be saved to `results/models/`

### 4. **Make Predictions** (1 minute)

```bash
# Single audio file
python inference.py --audio path/to/audio.wav --model-dir results/models

# Multiple files
python inference.py --audio-dir path/to/audio/folder --model-dir results/models --output results/predictions.csv
```

## 📊 What Gets Created

After training, you'll have:

```
results/
├── models/              # Trained models (.pkl, .h5 files)
├── plots/              # Evaluation plots (PNG images)
└── model_results.csv   # Performance metrics table
```

## 📁 Prepare Your Data

### Format 1: CSV with Features
If you already have extracted features:
```
feature_1,feature_2,...,Label,Filename
0.5,0.3,...,1,patient_001.wav
0.4,0.2,...,0,healthy_001.wav
```

### Format 2: Raw Audio Files
```bash
# Extract features from audio directory
python -c "
from feature_extraction import AudioFeatureExtractor
extractor = AudioFeatureExtractor()
df = extractor.batch_extract('path/to/audio/dir', 'features.csv')
"
```

## 🎯 Common Commands

### Training Variations

```bash
# Basic training (Random Forest, SVM, XGBoost)
python train.py

# Include LSTM neural network
python train.py --train-lstm

# Train ensemble model
python train.py --train-ensemble

# Full: all models with optimization
python train.py --train-lstm --train-ensemble

# Using custom data
python train.py --dataset-type csv --data-dir /path/to/data
```

### Inference Variations

```bash
# Single file with verbose output
python inference.py --audio file.wav --model-dir results/models

# Batch process with results export
python inference.py --audio-dir folder/ --model-dir results/models --output results.csv

# Recursive search through subdirectories
python inference.py --audio-dir data/ --recursive --model-dir results/models
```

## 📈 Expected Performance

| Model | Typical Accuracy |
|-------|------------------|
| Random Forest | 85-92% |
| SVM | 82-88% |
| XGBoost | 86-93% |
| LSTM | 80-90% |
| Ensemble | 88-94% |

*Performance varies based on data quality and features*

## 🔧 Troubleshooting

### ModuleNotFoundError
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### CUDA/GPU Error
```bash
# Use CPU version
# Add to your scripts:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Memory Error
```bash
# Reduce batch size in train.py
# Or use smaller feature set
```

### No Models Found
```bash
# Train first before inference
python train.py --dataset-type sample
```

## 📚 Next Steps

1. **Read Full Documentation**: See `README.md` for complete details
2. **Explore Code**: Check `*.py` files for implementation details
3. **Customize Models**: Modify hyperparameters in `model_training.py`
4. **Add Features**: Extend `feature_extraction.py` with custom features

## 🐍 Python API Usage

```python
# Quick example of using the package programmatically
from respiratory_disease_detection import *

# Load data
data = DataLoader().load_csv('data.csv')

# Prepare
X_train, X_test, y_train, y_test = DataPreparation.prepare_for_training(data)

# Train
model = RandomForestModel()
model.train(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = ModelEvaluator.evaluate(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.2f}%")

# Save
model.save('my_model.pkl')
```

## 📞 Support

- Check README.md for detailed documentation
- Review code comments for implementation details
- Run test_pipeline.py to verify setup
- Check results/plots/ for visualizations

---

**Tip**: Start with sample data to ensure everything works, then move to your actual data.
