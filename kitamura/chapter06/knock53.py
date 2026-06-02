import gensim.downloader as api
model = api.load("word2vec-google-news-300")

result = model.most_similar(positive=["Spain", "Athens"],
                            negative=["Madrid"])

rank = 0
for i in result:
    rank+=1
    print(f"{rank}:{i}")

"""1:('Greece', 0.6898480653762817)
2:('Aristeidis_Grigoriadis', 0.560684859752655)
3:('Ioannis_Drymonakos', 0.5552908778190613)
4:('Greeks', 0.5450686812400818)
5:('Ioannis_Christou', 0.5400863289833069)
6:('Hrysopiyi_Devetzi', 0.5248444676399231)
7:('Heraklio', 0.5207759737968445)
8:('Athens_Greece', 0.5168809294700623)
9:('Lithuania', 0.5166866183280945)
10:('Iraklion', 0.5146791338920593"""