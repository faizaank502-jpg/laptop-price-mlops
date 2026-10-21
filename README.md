# Laptop Price Prediction – MLOps Project

Predict laptop price from its specs (RAM, CPU, GPU, storage, screen, etc.).

## How to reproduce
1. Create the environment:  `conda env create -f environment.yml`
2. Activate it:              `conda activate laptop-mlops`
3. Put `laptop_price.csv` (Kaggle: muhammetvarl/laptop-price) in `data/raw/`
4. Clean data:               `python src/data_prep.py`
5. Train model:              `python src/train.py`

Results are saved in `reports/metrics.json`, model in `models/`.
Random seed is fixed (42), so results are the same every run.
WRONG LINE ADDED BY MISTAKE
