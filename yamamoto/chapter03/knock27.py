#26の処理に加えて、テンプレートの値からMediaWikiの内部リンクマークアップを除去し、テキストに変換せよ。

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
                
remove_emphasis = {}

for key, value in template.items(): 
    
    remove_emphasis[key] = re.sub(r'\'{2,5}', '', value) 
print("removed", remove_emphasis)    
                
remove_link = {}

for key, value in remove_emphasis.items():
    
    remove_link[key] = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', "",  value)

print(remove_link)                        