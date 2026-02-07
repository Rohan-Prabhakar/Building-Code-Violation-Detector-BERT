# Building Code Violation Detector

Fine-tuned RoBERTa model for classifying NYC building code violations by category (8 classes) and severity (3 levels).

See it in action: https://construction-code-violation-detector-bert-bkvmmvsnynas4vdk9dic.streamlit.app/ |
Fine Tuned Model-ViolationBERT: https://huggingface.co/Rohan1103/ViolationBERT

## Results

| Task | Macro F1 | Accuracy |
|---|---|---|
| Category | 0.8977 | 0.9620 |
| Severity | 0.8637 | 0.8685 |

## Project Structure

```
building-violation-detector/
├── 00_download_data.py          # Download NYC DOB data via API
├── 01_Dataset_Preparation.py    # Clean, label, split, tokenize
├── 02_Model_Training.py         # RoBERTa fine-tuning + hyperparameter search
├── 03_Evaluation.py             # Test evaluation, error analysis, inference
├── data/
│   ├── raw/                     # Raw CSVs from API
│   ├── processed/               # Cleaned data + tokenized datasets
│   └── splits/                  # train.csv, val.csv, test.csv
├── checkpoints/                 # Model checkpoints
├── figures/                     # All generated plots
├── results/                     # JSON results files
├── logs/                        # Training logs
├── requirements.txt
├── README.md
└── documentation.md             # Full writeup
```

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/building-violation-detector.git
cd building-violation-detector

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the scripts in order:

```bash
# Step 1: Download data from NYC Open Data API
python 00_download_data.py

# Step 2: Preprocess, clean, split, and tokenize
python 01_Dataset_Preparation.py

# Step 3: Train model (requires GPU)
python 02_Model_Training.py

# Step 4: Evaluate on test set + run inference demo
python 03_Evaluation.py
```

## Quick Inference

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model (see 03_Evaluation.py for full ViolationClassifier class)
ckpt = torch.load('checkpoints/final_model.pt', map_location='cpu')

# Predict
text = "FAILURE TO MAINTAIN BUILDING WALL BRICKS FALLING FROM FACADE"
# Returns: Category=Construction, Severity=HIGH
```

## Dataset

NYC DOB ECB Violations from NYC Open Data

- Source: https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad
- 300K records downloaded, 236K after cleaning
- 8 violation categories, 3 severity levels
- Public domain

## Model

- Base: RoBERTa-base (125M params)
- Architecture: Dual classification heads (category + severity)
- Training: Full fine-tuning, AdamW, weighted cross-entropy
- Best config: lr=1e-5, dropout=0.1, batch_size=32, 3 epochs

## Environment

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Trained on Kaggle (T4 GPU)
- Training time: ~4 hours for 3 hyperparameter configs
