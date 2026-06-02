#“United States”と”U.S.”のコサイン類似度を計算せよ。

from gensim.models import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

vector1 = np.array(model["United_States"])
vector2 = np.array(model["U.S."])

cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)) #コサイン類似度

print(cos_sim)