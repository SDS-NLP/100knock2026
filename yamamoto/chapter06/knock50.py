#Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ、300次元）をダウンロードし、”United States”の単語ベクトルを表示せよ。ただし、”United States”は内部的には”United_States”と表現されていることに注意せよ。

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format( #Word2Vec形式のファイルを読み込む
    "GoogleNews-vectors-negative300.bin.gz", binary = True #今回のデータはバイナリ形式
)
    
vector = model["United_States"]

print("United States =")
print(vector)