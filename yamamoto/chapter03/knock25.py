#記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し、辞書オブジェクトとして格納せよ。

import json
import re

template = {}

with open("jawiki-country.json", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        dict = json.loads(line)
        
        if dict["title"] == "イギリス":
            
            basic_info = re.search(r'^\{\{基礎情報.*?$(.*?)^\}\}$', dict["text"], re.S | re.M)
            list = re.findall(r'^\|([^=]+?)\s*=\s*(.+?)(?=\n^\||\n^\}\})', basic_info.group(1), re.M | re.S)
            
            for i in list:
                
                template[i[0]] = i[1]

print(template)