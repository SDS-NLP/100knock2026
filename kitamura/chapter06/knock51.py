import gensim.downloader as api

model = api.load('word2vec-google-news-300')

word1 = "United_States"
word2 = "U.S."

similarity = model.similarity(word1, word2)
print(similarity)

# 0.73107743