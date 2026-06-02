# 53. 加法構成性によるアナロジー
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き、
# ”Athens”のベクトルを足したベクトルを計算し、
# そのベクトルと類似度の高い10語とその類似度を出力せよ。

import os
from pathlib import Path
from gensim.models import KeyedVectors

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
data_path = DATA_DIR / "GoogleNews-vectors-negative300.bin"

wordvector = KeyedVectors.load_word2vec_format(data_path, binary=True)

positive_words = ["Spain", "Athens"]
negative_words = ["Madrid"]

results = wordvector.most_similar(
    positive=positive_words,
    negative=negative_words,
    topn=10,
)

for word, similarity in results:
    print(word, similarity)