#Wikipedia記事のJSONファイルを読み込み、「イギリス」に関する記事本文を表示せよ。問題21-29では、ここで抽出した記事本文に対して実行せよ。

import json

list = []

with open("jawiki-country.json", "r", encoding = "utf-8") as file: #jsonファイルの読み込み
    
    for line in file:
        
        dict = json.loads(line) #json.loads()の返り値はdict型
        list.append(dict)

for dict in list:
    
    if dict["title"] == "イギリス": #タイトルがイギリスの記事のみテキストを出力
        
        print(dict["text"])