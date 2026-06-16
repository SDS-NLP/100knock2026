"""58. Ward法によるクラスタリング

国名に関する単語ベクトルに対し、Ward法による階層型クラスタリングを実行し、
結果をデンドログラムとして可視化する。
"""

from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
VECTOR_PATH = DATA_DIR / "GoogleNews-vectors-negative300.bin"
QUESTIONS_PATH = SCRIPT_DIR / "questions-words.txt"
OUTPUT_PATH = SCRIPT_DIR / "knock58_dendrogram.png"
CACHE_DIR = SCRIPT_DIR / ".cache"

CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage


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

    country_names = [country for country in countries if country in wordvector]
    country_vectors = [wordvector[country] for country in country_names]

    Z = linkage(country_vectors, method="ward")

    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=country_names, leaf_rotation=90)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)

    print(f"saved dendrogram to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
