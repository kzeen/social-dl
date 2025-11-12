# Social DL: Predicting Social Roles from Tweets

## Overview

Social DL is a deep learning project for the INF554 / CSC_51054_EP Data Challenge 2025.
The goal is to predict whether a user is an Influencer (1) or an Observer (0) from the text of a single tweet.

We start from simple baselines such as TF-IDF + Logistic Regression and extend to modern deep learning
approaches based on Transformer language models (e.g. BERT, RoBERTa).

## Repository structure

-   `src/` — Python package with data loading, preprocessing, models, and training scripts.
-   `data/` — Local data folder (JSONL files and sample submissions from Kaggle).
-   `notebooks/` — Jupyter notebooks for EDA and experiments.
-   `scripts/` — Small shell helpers to run common experiments.
-   `reports/` — Project report, figures, and slides.

## Quick start

1. Clone the repo:

    ```bash
    git clone https://github.com/kzeen/social-dl.git
    cd social-dl
    ```

2. Run the setup script (Optionally creates a virtual environment and installs dependencies):

    ```bash
    chmod +x setup.sh # (If needed) Make the script executable (first time only)
    ./setup.sh
    ```

    You'll be prompted:

    ```bash
    Do you want to create a virtual environment (.venv)? (y/n)
    ```

    - Type **y/Y** to create and use a local virtual environment.
    - Type **n** to install dependencies globally (or in your active environment).

    After setup, activate the environment manually if needed:

    ```bash
    source .venv/bin/activate # macOS / Linux
    source .venv/Scripts/activate # Windows (Git Bash or Powershell)
    ```

3. (Optional) Reinstall dependencies again:

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

-   Reproducible training code.
-   Experiment configuration and logs.
-   The LaTeX source of the final 3-page report and appendix.
-   The script used to generate the final Kaggle submission CSV.

## Authors

-   Karl Zeeny
-   Prakhar Tiwari
-   Sarah Rouphael
