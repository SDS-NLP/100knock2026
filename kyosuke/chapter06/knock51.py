import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

word1 = "United_States"
word2 = "U.S."

if word1 in model and word2 in model:
    similarity = model.similarity(word1, word2)
    print(f"{similarity:.4f}")
else:
    print("指定された単語が辞書に存在しません。")