#記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ。

import json
import re

with open("jawiki-country.json", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        dict = json.loads(line)
        
        if dict["title"] == "イギリス":
            
            section = re.findall(r"(={2,})\s*([^\s=]*).*", dict["text"]) #("="が2つ以上), 空白が0個以上, (空白でも"="でもない任意の1文字が0個以上), 任意の文字が0個以上
            
            section_level = {} #セクション名とそのレベルを辞書型で格納
            
            for s in section:
                
                section_level[s[1]] = len(s[0]) - 1

print(section_level) 