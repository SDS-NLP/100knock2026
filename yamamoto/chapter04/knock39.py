#コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、両対数グラフをプロットせよ。
#zipfの法則：データを頻度の大きい順にランキングしたとき、順位と頻度は反比例する
#データが自然言語のとき、少数超頻出・大量低頻度という構造になる

import re
import MeCab
import matplotlib.pyplot as plt

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
    
    if token[1] != "記号":
        
        if token[0] in word_count:
            
            word_count[token[0]] += 1
        
        else:
            
            word_count[token[0]] = 1

sorted_count = sorted(word_count.items(), key = lambda x:x[1], reverse = True)

counts = [] #単語の出現頻度だけを格納するリスト

for word, count in sorted_count:
    
    counts.append(count) #出現頻度が多い方から順に格納

ranks = range(1, len(counts) + 1) #順位は1~単語数まで

plt.plot(ranks, counts) #順位をx軸, 出現頻度をy軸でプロット

plt.xscale("log")
plt.yscale("log")

plt.xlabel("rank")
plt.ylabel("frequency")

plt.show()