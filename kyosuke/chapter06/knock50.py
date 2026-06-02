import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'

print("巨大なモデルをメモリに読み込んでいます...（※PCのスペックにより1〜3分ほどかかります）")

model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

word = "United_States"
if word in model:
    vec = model[word]
    print(f"【{word} の単語ベクトル】")
    print(vec[:10]) 
    print(f"...\n(次元数: {vec.shape})")
else:
    print(f"エラー: {word} は辞書に存在しません。")