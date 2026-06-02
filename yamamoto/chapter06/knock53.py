#“Spain”の単語ベクトルから”Madrid”のベクトルを引き、”Athens”のベクトルを足したベクトルを計算し、そのベクトルと類似度の高い10語とその類似度を出力せよ。

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

spain = model["Spain"]
madrid = model["Madrid"]
athens = model["Athens"]

vector = spain - madrid + athens

ranking = model.most_similar(vector, topn = 10) #"United States"と類似度が大きい上位10件
    
for word, similarity in ranking:
        
    print(word, "：", similarity)