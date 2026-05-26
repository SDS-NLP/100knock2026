#コーパスにおける名詞の出現頻度を求め、出現頻度の高い20語とその出現頻度を表示せよ。

import re
import MeCab

with open("kokoro.txt", "r", encoding = "utf-8-sig") as file: #utf-8-sig：ファイル先頭に入るBOM(\ufeff)を自動除去
    
    text = file.read()    

parts = re.split(r"(?:^|\n+)[一二三四五六七八九十]+\n+", text, flags = re.MULTILINE) #テキストを章番号ごとに分割

chapters = [] #各章の本文を格納するリスト

for line in parts:
    
    line = line.strip()
    
    if line != "": #行が空文字でないとき
        
        chapters.append(line)

tagger = MeCab.Tagger()

tokens = [] #表層形とその品詞情報を格納するリスト

for chapter in chapters:
    
    result = tagger.parse(chapter)
    res_lines = result.splitlines()
    
    for line in res_lines[:-1]:
    
        surface, feature = line.split("\t")
        features = feature.split(",")
        features.insert(0, surface)

        tokens.append(features) 

word_count = {}

for token in tokens:
    
    if token[1] == "名詞": #品詞が名詞のとき
        
        if token[0] in word_count:
            
            word_count[token[0]] += 1
        
        else:
            
            word_count[token[0]] = 1

sorted_count = sorted(word_count.items(), key = lambda x:x[1], reverse = True)

print(sorted_count[:20])