#記事中でカテゴリ名を宣言している行を抽出せよ。

import json
import re

with open("jawiki-country.json", "r", encoding = "utf-8") as file: #jsonファイルの読み込み
    
    for line in file:
        
        dict = json.loads(line) #json.loads()の返り値はdict型
        
        if dict["title"] == "イギリス":
            
            #"Category"だけだと行ごと抽出できないので、正規表現を用いて周りの文字ごと抜き出す
            category = re.findall(r'.*Category.*', dict["text"]) #.：任意の1文字, *：0回以上
            print(category)