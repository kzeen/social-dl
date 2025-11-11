from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_pipeline(max_features: int = 50000) -> Pipeline:
    clf: Any = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", clf),
        ]
    )
    return pipeline
