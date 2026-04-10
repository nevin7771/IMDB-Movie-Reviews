"""
Week 1 — Day 7 (part 2): classify pasted reviews with the saved Day 6/7 model.

Creates `models/imdb_tfidf_linearsvc.joblib` via the in-app button or:
  uv run python day7_train_save.py

Usage (from project root):
  uv run streamlit run day7_streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import streamlit as st

from imdb_text import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "imdb_tfidf_linearsvc.joblib"
ACL_TRAIN = PROJECT_ROOT / "aclImdb" / "train"
ACL_TEST = PROJECT_ROOT / "aclImdb" / "test"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def _data_ready() -> bool:
    return ACL_TRAIN.is_dir() and ACL_TEST.is_dir()


def _render_setup() -> None:
    st.warning("No saved model found yet.")
    st.markdown(
        "1. Place the **`aclImdb`** folder in the project root (next to `day6_tfidf_train.ipynb`). "
        "Extract [Stanford’s ACL IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) if you don’t have it."
    )
    st.markdown(
        "2. Train once — **terminal:** `uv run python day7_train_save.py` **or** use the button below."
    )

    if not _data_ready():
        st.error(
            f"Expected **`{ACL_TRAIN}`** and **`{ACL_TEST}`**. "
            "Unpack `aclImdb` so those folders exist, then try again."
        )

    if st.button("Train and save model", type="primary", disabled=not _data_ready()):
        train_script = PROJECT_ROOT / "day7_train_save.py"
        with st.spinner("Training TF-IDF + LinearSVC (typically under ~2 minutes on CPU)…"):
            proc = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            st.error("Training failed. Output:")
            st.code((proc.stdout or "") + (proc.stderr or ""), language="text")
        else:
            load_model.clear()
            st.success("Model saved. Reloading…")
            st.rerun()


st.set_page_config(page_title="IMDB sentiment", page_icon="🎬", layout="centered")
st.title("IMDB review sentiment")
st.caption("TF-IDF + LinearSVC — same pipeline as `day6_tfidf_train.ipynb`.")

if not MODEL_PATH.is_file():
    _render_setup()
    st.stop()

model = load_model()

default_sample = (
    "This film was a waste of time — wooden acting and a predictable plot."
)

text = st.text_area("Paste a movie review (English):", value=default_sample, height=160)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        margin = float(model.decision_function([cleaned])[0])
        label = "Positive" if pred == 1 else "Negative"
        st.success(f"**Prediction: {label}**")
        st.metric("LinearSVC margin (higher → more confident positive)", f"{margin:.4f}")
