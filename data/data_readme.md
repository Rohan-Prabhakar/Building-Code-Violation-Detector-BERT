# Data Files Note

The data/ files are not included in this repository due to size.
They are generated automatically when you run the first notebook.

## How to Generate

Run "data prep and training.ipynb" which will:
- Download raw data from NYC Open Data API into data/raw/
- Clean and preprocess into data/processed/
- Create train/val/test splits in data/splits/
- Tokenize and save as data/processed/tokenized_datasets.pt

## Directories Generated

```
data/
  raw/
    ecb_violations.csv
    safety_violations.csv
    hpd_violations.csv
  processed/
    violations_cleaned.csv
    tokenized_datasets.pt
    label_maps.json
  splits/
    train.csv
    val.csv
    test.csv
```

## Already Included in Repo

- checkpoints/ (final_model.pt, best_config_*.pt)
- figures/ (all generated plots)
- results/ (hp_results.json, evaluation_results.json)
- logs/ (training.log)

## Pre-trained Model

Also hosted on HuggingFace: https://huggingface.co/Rohan1103/ViolationBERT

The Streamlit app downloads it automatically from HuggingFace on first run.

## Note on Reproducibility

The API pulls live data from NYC Open Data which updates weekly. A fresh run may pull slightly different records than the original training run. Exact metrics may vary slightly but overall performance should be comparable. The pre-trained model on HuggingFace reflects the original training run results.

## Data Source

NYC DOB ECB Violations (public domain)

https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad
