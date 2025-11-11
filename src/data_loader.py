import json
import os
from typing import List, Dict, Tuple

import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "kaggle_test.jsonl")

    train_df = load_jsonl(train_path)
    test_df = load_jsonl(test_path)

    return train_df, test_df
