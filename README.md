# Social DL: Predicting Social Roles from Tweets

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/csc-51054-ep-data-challenge-2025)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**CSC_51054_EP Data Challenge 2025** | Kaggle Team: **Social DL**

A deep learning solution for classifying Twitter users as Influencers or Observers based on individual tweet content and metadata. Achieved **84.4% accuracy** on the Kaggle test set using a hybrid approach combining CamemBERT contextual embeddings with engineered features and gradient boosting ensembles.

---

## Overview

This project tackles the challenge of predicting social roles from short-form social media text. Influencers exhibit asymmetrical follower-to-following ratios and broadcast-oriented behavior, while Observers maintain reciprocal, conversational patterns. The task requires inferring these roles from minimal context: a single tweet (140-280 characters) with associated metadata.

### Key Features

-   **Hybrid Architecture**: Combines contextual language understanding (CamemBERT) with behavioral signals (79 engineered features)
-   **Weighted Multi-Layer Pooling**: Extracts embeddings from last 4 CamemBERT layers with learned weights [1.0, 2.0, 3.0, 4.0]
-   **Bilingual Feature Engineering**: Detects 60+ promotional phrases in French and English
-   **Heterogeneous Ensemble**: Combines XGBoost, LightGBM, and CatBoost for robust predictions

### Results

| Model                        | Accuracy  |
| ---------------------------- | --------- |
| Dummy Classifier (Baseline)  | 53.0%     |
| Logistic Regression (TF-IDF) | 63.0%     |
| XGBoost (Metadata Only)      | 82.8%     |
| **CamemBERT + Ensemble**     | **84.0%** |
| **Stacked Ensemble (Final)** | **84.4%** |

---

## Dataset

-   **Training**: 154,914 tweets from 38,560 users with 194 raw features
-   **Test**: 103,380 tweets from 25,890 users
-   **Class Distribution**: ~53% Observers, ~47% Influencers

Each data point represents a single tweet with associated user metadata and temporal information. The challenge treats tweets independently without leveraging user-level aggregation.

---

## Methodology

### 1. Feature Engineering (79 Features)

**Text Content** (12): URL presence, hashtag/mention counts, emoji density, punctuation ratios, capitalization, elongated words

**Promotional Language** (2): Binary/count detection of 60+ bilingual promotional phrases ("giveaway", "nouvelle vidéo", "code promo", etc.)

**Temporal** (7): Hour, day of week, weekend/business hours flags, time of day segments, month

**Behavioral** (58): Account age, tweet frequency, encoded categorical metadata

### 2. Text Representation

CamemBERT-base embeddings (768-d) extracted via weighted pooling of last 4 layers:

```
e = Σ(i=9→12) wᵢ · MeanPool(Hᵢ)
```

Concatenated with metadata features → **847-dimensional input**

### 3. Ensemble Architecture

-   **XGBoost**: 900 trees, depth=8, lr=0.05 → 84.8% val
-   **LightGBM**: 512 leaves, lr=0.015, feat_frac=0.8 → 85.1% val
-   **CatBoost**: 1500 iter, depth=8, lr=0.05 → 84.3% val

Final prediction: Average of probability outputs

---

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/kzeen/social-dl.git
cd social-dl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n social-dl python=3.9
conda activate social-dl

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Setup

Place competition data files in the `data/` directory:

-   `train.jsonl`
-   `kaggle_test.jsonl`

### 2. Run Training Pipeline

Open and execute the main notebook:

```bash
jupyter notebook notebooks/Embedded_ensemble.ipynb
```

The notebook performs:

1. Feature extraction and engineering
2. CamemBERT embedding generation (GPU accelerated)
3. Hyperparameter tuning via grid search
4. Ensemble training on full dataset
5. Test set prediction generation

### 3. Generate Submission

Output is saved as `final_submission.csv` in the notebook directory, ready for Kaggle upload.

---

## Repository Structure

**Current Structure:**

```
social-dl/
├── data/                      # Competition data files
│   ├── train.jsonl
│   ├── kaggle_test.jsonl
│   └── README.md
├── notebooks/                 # Jupyter notebooks
│   ├── Baseline Model and submission/
│   │   ├── baseline.ipynb    # Logistic regression baseline
│   │   ├── dummy.csv
│   │   └── logistic_regression.csv
│   ├── Embedded_ensemble.ipynb  # **Main pipeline**
│   ├── Data.ipynb            # Exploratory analysis
│   ├── distilbert_finetune.ipynb
│   └── svm.ipynb
├── .venv/                    # Virtual environment
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Dependencies

Core libraries:

-   `torch` - PyTorch deep learning framework
-   `transformers` - Hugging Face transformers (CamemBERT)
-   `xgboost`, `lightgbm`, `catboost` - Gradient boosting implementations
-   `scikit-learn` - Preprocessing and metrics
-   `pandas`, `numpy` - Data manipulation

See `requirements.txt` for complete list.

---

## Hardware Requirements

-   **GPU Recommended**: NVIDIA GPU with 4GB+ VRAM for CamemBERT inference
-   **RAM**: 16GB+ recommended for full dataset processing
-   **Disk**: ~2GB for data + models

---

## Team

-   **Karl Zeeny** - karl.zeeny@polytechnique.edu
-   **Prakhar Tiwari** - prakhar.tiwari@polytechnique.edu
-   **Sarah Rouphael** - sarah.rouphael@polytechnique.edu

École Polytechnique | Master in Visual and Creative AI

---

## References

-   Martin et al. (2020). "CamemBERT: A Tasty French Language Model." _ACL_.
-   Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." _KDD_.
-   Jawahar et al. (2019). "What Does BERT Learn About the Structure of Language?" _ACL_.

---

## License

This project is for academic purposes as part of the CSC_51054_EP course at École Polytechnique.
