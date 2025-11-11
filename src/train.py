import argparse
import os
from typing import Literal

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .data_loader import load_datasets
from .preprocess import train_val_split, extract_xy, TEXT_COLUMN, LABEL_COLUMN
from .models import build_logistic_pipeline


ModelType = Literal["logistic", "bert"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_logistic(data_dir: str, output_dir: str) -> None:
    train_df, _ = load_datasets(data_dir)
    train_df = train_df.rename(columns={"tweet": TEXT_COLUMN, "label": LABEL_COLUMN})

    train_split, val_split = train_val_split(train_df)
    X_train, y_train = extract_xy(train_split)
    X_val, y_val = extract_xy(val_split)

    model = build_logistic_pipeline()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy (logistic): {acc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "logistic_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved logistic model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "bert"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model == "logistic":
        train_logistic(args.data_dir, args.output_dir)
    else:
        raise NotImplementedError("BERT training is not implemented yet.")


if __name__ == "__main__":
    main()
