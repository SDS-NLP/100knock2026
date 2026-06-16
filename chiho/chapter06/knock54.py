"""54. アナロジーデータでの実験

単語アナロジーの評価データをダウンロードし、
各事例について vec(word2) - vec(word1) + vec(word3) を計算し、
最も類似度が高い単語とその類似度を記録する。
"""

import os
from pathlib import Path

from gensim.models import KeyedVectors


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
DATA_PATH = DATA_DIR / "GoogleNews-vectors-negative300.bin"
QUESTIONS_PATH = SCRIPT_DIR / "questions-words.txt"
OUTPUT_PATH = SCRIPT_DIR / "knock54_results.txt"

wordvector = KeyedVectors.load_word2vec_format(DATA_PATH, binary=True)

current_section = None
results = []
section_counts = {}

with QUESTIONS_PATH.open() as questions_file:
    for line in questions_file:
        stripped_line = line.strip()

        if stripped_line.startswith(": "):
            current_section = stripped_line[2:]
            section_counts.setdefault(current_section, 0)
            print(f"processing section: {current_section}", flush=True)
            continue

        if not stripped_line or current_section is None:
            continue

        section_counts[current_section] += 1
        word1, word2, word3, word4 = stripped_line.split()
        predicted_word, similarity = wordvector.most_similar(
            positive=[word2, word3],
            negative=[word1],
            topn=1,
        )[0]
        results.append(
            "\t".join(
                [
                    current_section,
                    word1,
                    word2,
                    word3,
                    word4,
                    predicted_word,
                    str(similarity),
                ]
            )
        )
        if section_counts[current_section] % 500 == 0:
            print(
                f"  {current_section}: {section_counts[current_section]} examples processed",
                flush=True,
            )

OUTPUT_PATH.write_text("\n".join(results) + "\n")
print("\n".join(results))
