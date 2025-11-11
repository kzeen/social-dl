# Social DL: Predicting Social Roles from Tweets

## Overview
Social DL is a deep learning project for the INF554 / CSC_51054_EP Data Challenge 2025.
The goal is to predict whether a user is an Influencer (1) or an Observer (0) from the text of a single tweet.

We start from simple baselines such as TF-IDF + Logistic Regression and extend to modern deep learning
approaches based on Transformer language models (e.g. BERT, RoBERTa).

## Repository structure

- `src/` — Python package with data loading, preprocessing, models, and training scripts.
- `data/` — Local data folder (JSONL files and sample submissions from Kaggle).
- `notebooks/` — Jupyter notebooks for EDA and experiments.
- `scripts/` — Small shell helpers to run common experiments.
- `reports/` — Project report, figures, and slides.

## Quick start

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/social-dl.git
   cd social-dl
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the challenge data files under `data/`:

   - `train.jsonl`
   - `kaggle_test.jsonl`
   - `dummy.csv`
   - `logistic_predictions.csv`

5. Run a baseline experiment:

   ```bash
   python -m src.train --model logistic --data_dir data --output_dir outputs/logistic
   ```

6. Run a Transformer experiment:

   ```bash
   python -m src.train --model bert --data_dir data --output_dir outputs/bert --epochs 3
   ```

## Deliverables

For the class project we will use this repo to keep:

- Reproducible training code.
- Experiment configuration and logs.
- The LaTeX source of the final 3-page report and appendix.
- The script used to generate the final Kaggle submission CSV.

## Authors

- Neil (Nil) Biescas Rue
- <Teammate 1>
- <Teammate 2>
