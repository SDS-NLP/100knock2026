#記事から参照されているメディアファイルをすべて抜き出せ。

import json
import re

with open("jawiki-country.json", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        dict = json.loads(line)
        
        if dict["title"] == "イギリス":
            
            file = re.findall(r'[\w-]+\.[a-z]{3,}', dict["text"])
            print(file)