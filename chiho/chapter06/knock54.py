# 54. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし、国と首都に関する事例（: capital-common-countriesセクション）
# に対して、vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し、そのベクトルと類似度が
# 最も高い単語と、その類似度を求めよ。求めた単語と類似度は、各事例と一緒に記録せよ。

import os
from pathlib import Path

from gensim.models import KeyedVectors


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
data_path = DATA_DIR / "GoogleNews-vectors-negative300.bin"
questions_path = SCRIPT_DIR / "questions-words.txt"
output_path = SCRIPT_DIR / "knock54_results.txt"

wordvector = KeyedVectors.load_word2vec_format(data_path, binary=True)

in_target_section = False
results = []

with questions_path.open() as questions_file:
    for line in questions_file:
        stripped_line = line.strip()

        if stripped_line.startswith(": "):
            in_target_section = stripped_line == ": capital-common-countries"
            continue

        if not in_target_section or not stripped_line:
            continue

        word1, word2, word3, word4 = stripped_line.split()
        predicted_word, similarity = wordvector.most_similar(
            positive=[word2, word3],
            negative=[word1],
            topn=1,
        )[0]
        results.append(
            f"{word1}\t{word2}\t{word3}\t{word4}\t{predicted_word}\t{similarity}"
        )

output_path.write_text("\n".join(results) + "\n")
print("\n".join(results))
