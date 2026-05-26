import re 
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def extract_nouns(text):
    pattern = r"(《.*?》)"
    text = re.sub(pattern, "", text)

    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    nouns = []
    while node:
        features = node.feature.split(",")

        if features[0] == "名詞":
            nouns.append(node.surface)


        node = node.next

    return " ".join(nouns)


file = "kokoro.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

pattern1 = r"(《.*?》)"
pattern2 = r"[一二三四五六七八九十百]+(?:\n|\s)"
text = re.sub(pattern1, "", text)
chapters = re.split(pattern2, text)    

print(len(chapters))

corpus = []
for chapter in chapters:
    corpus.append(extract_nouns(chapter))


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

target_idx = 56 
tfidf_vector = tfidf_matrix[target_idx-1].toarray()[0]
feature_names = vectorizer.get_feature_names_out()

words_in_target = corpus[target_idx].split()
total_words_in_target = len(words_in_target)
word_counts = pd.Series(words_in_target).value_counts()

# データをまとめる
data = []
for word_idx, tfidf_score in enumerate(tfidf_vector):
    if tfidf_score > 0:
        word = feature_names[word_idx]
        count = word_counts.get(word, 0)
        tf = count / total_words_in_target if total_words_in_target > 0 else 0
        idf = vectorizer.idf_[word_idx]
        
        data.append({
            '単語': word,
            'TF': tf,
            'IDF': idf,
            'TF-IDF': tfidf_score
        })

df = pd.DataFrame(data)
if not df.empty:
    top20_df = df.sort_values(by='TF-IDF', ascending=False).head(20)
    print(top20_df.to_string(index=False))