# Building Code Violation Detector

Fine-tuned RoBERTa model for classifying NYC building code violations by category (8 classes) and severity (3 levels).

**See it in action:** [Live Demo](https://construction-code-violation-detector-bert-bkvmmvsnynas4vdk9dic.streamlit.app/)

**Fine-Tuned Model (ViolationBERT):** [HuggingFace](https://huggingface.co/Rohan1103/ViolationBERT)

## Results

### Three-Way Comparison

| Model | Category F1 | Severity F1 | Combined F1 |
|---|---|---|---|
| Random Baseline | 0.0046 | 0.1530 | 0.0788 |
| Zero-Shot (BART-MNLI, 400M) | 0.2334 | 0.3277 | 0.2805 |
| **ViolationBERT (Ours, 125M)** | **0.8977** | **0.8637** | **0.8807** |

A 400M parameter generic model loses to our 125M fine-tuned model, proving domain adaptation is necessary for construction NLP.

## Project Structure

```
building-violation-detector/
├── app/                         # Streamlit deployment files
├── checkpoints/                 # Model checkpoints (final_model.pt)
├── data/
│   ├── raw/                     # Raw CSVs from API
│   ├── processed/               # Cleaned data + tokenized datasets
│   └── splits/                  # train.csv, val.csv, test.csv
├── figures/                     # All generated plots
├── logs/                        # Training logs
├── results/                     # JSON results files
├── data prep and training.ipynb # 00 Download + 01 Dataset Prep + 02 Training
├── evaluation.ipynb             # 03 Evaluation + Error Analysis + Inference
├── readme.md
└── requirements.txt
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

Run the notebooks in order for retraining:

```
1. data prep and training.ipynb   # Downloads data, preprocesses, trains model
2. evaluation.ipynb               # Evaluates on test set, error analysis, inference demo
```

For the Streamlit demo of current fine tuned model:
```bash
pip install streamlit
streamlit run app/app.py
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
