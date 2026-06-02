import gensim.downloader as api

# Google Newsデータセットで学習済みのWord2Vecモデル（300次元）をロード
model = api.load('word2vec-google-news-300')

# "United_States" の単語ベクトルを取得
word = "United_States"
if word in model:
    vector = model[word]
    
    print(vector[:10])  # 300次元すべて表示すると長いため、先頭の10次元を抜粋
    
else:
    print(f"'{word}' はモデルの語彙に存在しません。")

"""[-0.03613281 -0.04833984  0.23535156  0.17480469 -0.14648438 -0.07421875
 -0.1015625  -0.07714844  0.109375   -0.05712891]"""