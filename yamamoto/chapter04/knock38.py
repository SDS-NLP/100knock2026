#日本に関する記事における名詞のTF・IDFスコアを求め、TF・IDFスコア上位20語とそのTF, IDF, TF・IDFを表示せよ。

import re
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer

with open("kokoro.txt", "r", encoding = "utf-8-sig") as file:
    
    text = file.read()    

parts = re.split(r"(?:^|\n+)[一二三四五六七八九十]+\n+", text, flags = re.MULTILINE)

chapters = [] 

for line in parts:
    
    line = line.strip()
    
    if line != "":
        
        chapters.append(line)

tagger = MeCab.Tagger()

documents = [] #章ごとに分かち書きされた本文を格納するリスト

for chapter in chapters: #各章を取得
    
    words = [] #表層形だけを格納するリスト
    
    result = tagger.parse(chapter)
    res_lines = result.splitlines()
    
    for line in res_lines[:-1]:
    
        surface, feature = line.split("\t")
        features = feature.split(",")
        
        if features[0] == "名詞":
            
            words.append(surface)
    
    document = " ".join(words) #分かち書きされた本文を結合
    documents.append(document)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents) #各章に対してTF-IDFを計算し行列を作成

names = vectorizer.get_feature_names_out() #tfidf_matrixから語彙のリストを取得
scores = tfidf_matrix.toarray()[0] #今回は1章を対象としてスコアを取り出す

word_scores = {} #単語とそのスコアを格納する辞書

for word, score in zip(names, scores): #zip()：複数のリストから同時に要素を取り出す
    
    word_scores[word] = score #{word:score}

sorted_words = sorted(word_scores.items(), key = lambda x:x[1], reverse = True) #scoreの降順でソート

for word, score in sorted_words[:20]:
    
    print(word, score)