import argparse
import os

import joblib
import pandas as pd

from .data_loader import load_datasets
from .preprocess import TEXT_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_path", type=str, default="outputs/predictions.csv")
    args = parser.parse_args()

    _, test_df = load_datasets(args.data_dir)
    test_df = test_df.rename(columns={"tweet": TEXT_COLUMN})

    model = joblib.load(args.checkpoint)

    X_test = test_df[TEXT_COLUMN].astype(str).tolist()
    preds = model.predict(X_test)

    if "ID" in test_df.columns:
        ids = test_df["ID"].tolist()
    else:
        ids = list(range(len(preds)))

    out_df = pd.DataFrame({"ID": ids, "Prediction": preds})
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
