#事前学習済み単語埋め込みを活用し、V×dの単語埋め込み行列Eを作成せよ。ここで、Vは単語埋め込みの語彙数、dは単語埋め込みの次元数である。ただし、単語埋め込み行列の先頭の行ベクトルE[0,:]は、将来的にパディング（<PAD>）トークンの埋め込みベクトルとして用いたいので、ゼロベクトルとして予約せよ。ゆえに、Eの2行目以降に事前学習済み単語埋め込みを読み込むことになる。
#もし、Google Newsデータセットの学習済み単語ベクトル（300万単語・フレーズ、300次元）を全て読み込んだ場合、V=3000001, d=300になるはずである（ただ、300万単語の中には、殆ど用いられない稀な単語も含まれるので、語彙を削減した方がメモリの節約になる）。
#また、単語埋め込み行列の構築と同時に、単語埋め込み行列の各行のインデックス番号（トークンID）と、単語（トークン）への双方向の対応付けを保持せよ。

from gensim.models import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

v = len(model.index_to_key) + 1 #埋め込み行列の行数(語彙+1)
d = model.vector_size #埋め込み行列の列数

embeddings = np.zeros((v, d), dtype = np.float32) #サイズが(v,d)のゼロ行列

word_to_id = {"<PAD>": 0} #wordからidを対応づける辞書
id_to_word = {0: "<PAD>"} #idからwordを対応づける辞書

for i, word in enumerate(model.index_to_key, start = 1): #i=1スタートで単語とそのインデックスを取り出す
    
    embeddings[i] = model[word] #wordの埋め込みを埋め込み行列のi行目へ
    word_to_id[word] = i #wordにインデックスiを対応
    id_to_word[i] = word #インデックスiにwordを対応

if __name__ == "__main__":
    
    print("embedding_shape:", embeddings.shape)
    print("United States:", word_to_id["United_States"])
    print("ID_410:", id_to_word[410])