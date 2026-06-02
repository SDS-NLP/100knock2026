import gensim.downloader as api
import re
model = api.load("word2vec-google-news-300")

file = "questions-words.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()


pattern = r"^: capital-common-countries\n(.*?)(?=\n:)"

capital_common_countries = re.findall(pattern, text, re.DOTALL)

line_list = capital_common_countries[0].strip().split("\n")

print(line_list[:2])

result = []

for line in line_list:
    words = line.split()
    word1, word2, word3, word4 = words

    predict = model.most_similar(positive=[word2, word3], negative=[word1])

    pred_word, pred_simi  = predict[0]
    result.append([line, pred_word, pred_simi])


for i in result[:2]:
    print(i)


"""
['Athens Greece Baghdad Iraq', 'Athens Greece Bangkok Thailand']
['Athens Greece Baghdad Iraq', 'Athens Greece Bangkok Thailand']
['Athens Greece Baghdad Iraq', 'Iraqi', 0.6351870894432068]
['Athens Greece Bangkok Thailand', 'Thailand', 0.7137669920921326]"""