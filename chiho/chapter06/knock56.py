"""56. WordSimilarity-353 の評価

The WordSimilarity-353 Test Collection の評価データを読み込み、
単語ベクトルで計算した類似度と人手評価の間の
スピアマン相関係数を計算する。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
VECTOR_PATH = DATA_DIR / "GoogleNews-vectors-negative300.bin"
WORDSIM_PATH = SCRIPT_DIR / "wordsim353" / "combined.csv"


def load_wordsim_data(path: Path) -> list[tuple[str, str, float]]:
    """Load WordSimilarity-353 data."""
    data: list[tuple[str, str, float]] = []
    delimiter = "," if path.suffix == ".csv" else "\t"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            word1, word2, human_score = row[0].strip(), row[1].strip(), float(row[2])
            data.append((word1, word2, human_score))
    return data


def rankdata(values: list[float]) -> np.ndarray:
    """Assign average ranks for tied values."""
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array)
    ranks = np.empty(len(values_array), dtype=float)

    i = 0
    while i < len(values_array):
        j = i
        while j + 1 < len(values_array) and values_array[order[j + 1]] == values_array[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman's rank correlation coefficient."""
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def main() -> None:
    wordvector = KeyedVectors.load_word2vec_format(VECTOR_PATH, binary=True)
    wordsim_path = WORDSIM_PATH if WORDSIM_PATH.exists() else WORDSIM_PATH.with_suffix(".tab")
    data = load_wordsim_data(wordsim_path)

    model_scores: list[float] = []
    human_scores: list[float] = []
    skipped = 0

    for word1, word2, human_score in data:
        if word1 not in wordvector or word2 not in wordvector:
            skipped += 1
            continue
        model_scores.append(wordvector.similarity(word1, word2))
        human_scores.append(human_score)

    correlation = spearman_correlation(model_scores, human_scores)

    print(f"pairs used: {len(model_scores)}")
    print(f"pairs skipped: {skipped}")
    print(f"spearman correlation: {correlation:.6f}")


if __name__ == "__main__":
    main()
