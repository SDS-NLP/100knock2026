"""57. k-meansクラスタリング

国名に関する単語ベクトルを抽出し、k-meansクラスタリングを
クラスタ数 k=5 で実行する。
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from gensim.models import KeyedVectors
from sklearn.cluster import KMeans


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
VECTOR_PATH = DATA_DIR / "GoogleNews-vectors-negative300.bin"
QUESTIONS_PATH = SCRIPT_DIR / "questions-words.txt"


def load_country_words(path: Path) -> list[str]:
    """Extract unique country names from the capital-common-countries section."""
    countries: list[str] = []
    in_section = False

    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()

            if stripped_line.startswith(": "):
                in_section = stripped_line == ": capital-common-countries"
                continue

            if not in_section or not stripped_line:
                continue

            _, country1, _, country2 = stripped_line.split()
            for country in (country1, country2):
                if country not in countries:
                    countries.append(country)

    return countries


def main() -> None:
    wordvector = KeyedVectors.load_word2vec_format(VECTOR_PATH, binary=True)
    countries = load_country_words(QUESTIONS_PATH)

    country_vectors = [wordvector[country] for country in countries if country in wordvector]
    country_names = [country for country in countries if country in wordvector]

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(country_vectors)

    clusters: dict[int, list[str]] = defaultdict(list)
    for country, label in zip(country_names, labels):
        clusters[int(label)].append(country)

    for cluster_id in sorted(clusters):
        print(f"cluster {cluster_id}:")
        for country in sorted(clusters[cluster_id]):
            print(f"  {country}")


if __name__ == "__main__":
    main()
