"""
Week 1 — Day 7 (part 1): train TF-IDF + LinearSVC, save pipeline for the Streamlit app.

Matches `day6_tfidf_train.ipynb` hyperparameters. Requires `aclImdb/` under the project root.

Usage (from project root):
  uv run python day7_train_save.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from imdb_text import clean_text, load_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "aclImdb"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "imdb_tfidf_linearsvc.joblib"


def main() -> None:
    train_dir = DATA_ROOT / "train"
    test_dir = DATA_ROOT / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(
            f"Missing {train_dir} or {test_dir}. Extract aclImdb next to this script."
        )

    train_texts, train_labels = load_data(str(train_dir))
    test_texts, test_labels = load_data(str(test_dir))

    train_texts = [clean_text(t) for t in tqdm(train_texts, desc="clean train")]
    test_texts = [clean_text(t) for t in tqdm(test_texts, desc="clean test")]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(random_state=42)),
        ]
    )

    t0 = time.perf_counter()
    model.fit(train_texts, train_labels)
    fit_s = time.perf_counter() - t0
    print(f"Fit time: {fit_s:.1f}s")

    y_pred = model.predict(test_texts)
    acc = accuracy_score(test_labels, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(test_labels, y_pred, digits=4))
    print("Confusion matrix [rows=true 0,1 | cols=pred 0,1]:")
    print(confusion_matrix(test_labels, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
