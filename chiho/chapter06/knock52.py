#52. 類似度の高い単語10件
#“United States”とコサイン類似度が高い10語と、その類似度を出力せよ。

import os
from pathlib import Path

from gensim.models import KeyedVectors


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
data_path = DATA_DIR / "GoogleNews-vectors-negative300.bin"

wordvector = KeyedVectors.load_word2vec_format(data_path, binary=True)

for word, similarity in wordvector.most_similar("United_States", topn=10):
    print(word, similarity)