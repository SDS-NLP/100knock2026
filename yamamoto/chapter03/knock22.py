#記事のカテゴリ名を（行単位ではなく名前で）抽出せよ。

import json
import re

with open("jawiki-country.json", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        dict = json.loads(line)
        
        if dict["title"] == "イギリス":
            
            #.*?とすることで、\]は1つ目の]に反応(.*のとき1つ目の]は.*に吸収されるが、.*?はなるべく少ない文字数になるように調整されるので1つ目の]は\]にマッチ)
            category = re.findall(r'.*Category(.*?)\]', dict["text"]) #()：グループ化, ?：0回または1回
            print(category)