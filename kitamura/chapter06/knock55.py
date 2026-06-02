import gensim.downloader as api
import re
from tqdm import tqdm

model = api.load("word2vec-google-news-300")

file = "questions-words.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()


pattern_gram = r": gram[^\n]*\n(.*?)(?=\n:|\Z)"
gram_block = re.findall(pattern_gram, text, re.DOTALL)
gram_list = []
for block in gram_block:
    gram_list.extend(block.strip().split("\n"))

pattern_semantic = r": (?!gram)[^\n]*\n(.*?)(?=\n:|\Z)"
semantic_blocks = re.findall(pattern_semantic, text, re.DOTALL)
semantic_list = []
for block in semantic_blocks:
    semantic_list.extend(block.strip().split("\n"))


syntactic_correct = 0
syntactic_total = len(gram_list)

for line in tqdm(gram_list):
    words = line.split()
    word1, word2, word3, word4 = words

    try:
        predict = model.most_similar(positive=[word2, word3], negative=[word1])
        pred_word = predict[0][0]

    except KeyError:
        pred_word = ""

    if word4 == pred_word:
        syntactic_correct += 1

    
semantic_correct = 0
semantic_total = len(semantic_list)

for line in tqdm(semantic_list):
    words = line.split()
    word1, word2, word3, word4 = words

    try:
        predict = model.most_similar(positive=[word2, word3], negative=[word1])
        pred_word = predict[0][0]

    except KeyError:
        pred_word = ""

    if word4 == pred_word:
        semantic_correct += 1

syntactic_accuracy = syntactic_correct / syntactic_total
semantic_accuracy = semantic_correct / semantic_total

print(f"syntactic:{syntactic_accuracy}")
print(f"semantic:{semantic_accuracy}")

# syntactic:0.7400468384074942
# semantic:0.7308602999210734