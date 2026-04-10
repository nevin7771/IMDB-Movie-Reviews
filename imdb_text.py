"""Shared load/clean helpers for IMDB sentiment (Day 5–7)."""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from tqdm.auto import tqdm


def load_data(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    for label in ["pos", "neg"]:
        folder = os.path.join(path, label)
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]

        for file in tqdm(files, desc=f"{os.path.basename(path)}/{label}"):
            fp = os.path.join(folder, file)
            with open(fp, encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if label == "pos" else 0)

    return texts, labels


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
