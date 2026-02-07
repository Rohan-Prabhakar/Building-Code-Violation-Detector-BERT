DATA FILES NOTE
===============

The data/ files are not included in this repository due to size.
It is generated automatically when you run the first notebook.

HOW TO GENERATE:
  Run "data prep and training.ipynb" - this will:
    - Download raw data from NYC Open Data API into data/raw/
    - Clean and preprocess into data/processed/
    - Create train/val/test splits in data/splits/
    - Tokenize and save as data/processed/tokenized_datasets.pt

DIRECTORIES GENERATED:
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

ALREADY INCLUDED IN REPO:
  checkpoints/               - final_model.pt, best_config_*.pt
  figures/                   - All generated plots (01 through 09)
  results/                   - hp_results.json, evaluation_results.json
  logs/                      - training.log

PRE-TRAINED MODEL:
  Also hosted on HuggingFace: https://huggingface.co/Rohan1103/ViolationBERT
  The Streamlit app downloads it automatically from HuggingFace on first run.

NOTE ON REPRODUCIBILITY:
  The API pulls live data from NYC Open Data which updates weekly.
  A fresh run may pull slightly different records than the original training run.
  Exact metrics may vary slightly but overall performance should be comparable.
  The pre-trained model on HuggingFace reflects the original training run results.

DATA SOURCE:
  NYC DOB ECB Violations (public domain)
  https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad
