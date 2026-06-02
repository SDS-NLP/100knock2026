import gensim.downloader as api

word = "United_States"
model = api.load('word2vec-google-news-300')
similar_words = model.most_similar(word)

rank = 0
for i in similar_words:
    rank += 1
    print(f"{rank}・・・{i}")


"""1・・・('Unites_States', 0.7877248525619507)
2・・・('Untied_States', 0.7541369795799255)
3・・・('United_Sates', 0.74007248878479)
4・・・('U.S.', 0.7310774326324463)
5・・・('theUnited_States', 0.6404393911361694)
6・・・('America', 0.6178410053253174)
7・・・('UnitedStates', 0.6167312264442444)
8・・・('Europe', 0.6132988333702087)
9・・・('countries', 0.6044804453849792)
10・・・('Canada', 0.6019070148468018)"""