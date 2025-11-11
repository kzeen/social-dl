from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def train_val_split(
    df: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    return train_df, val_df


def extract_xy(
    df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
):
    X = df[text_column].astype(str).tolist()
    y = df[label_column].tolist()
    return X, y
