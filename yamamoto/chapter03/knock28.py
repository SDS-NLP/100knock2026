#27の処理に加えて、テンプレートの値からMediaWikiマークアップを可能な限り除去し、国の基本情報を整形せよ。

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
                
remove_link = {}

for key, value in remove_emphasis.items():
    
    remove_link[key] = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', "",  value)

remove_markup = {}

for key, value in remove_link.items():
    
    remove_markup[key] = re.sub(r'\[https?:\/\/[^\s]+\s(.+?)\]', r'\1', value)
    remove_markup[key] = re.sub(r'</?(ref|br).*?>', '', remove_markup[key])
    remove_markup[key] = re.sub(r'\{\{lang\|[^|]+\|([^}]+)\}\}', r'\1', remove_markup[key]) #{{lang|**|text}} ⇨ text
    remove_markup[key] = re.sub(r'\{\{0\}\}', '', remove_markup[key])
    remove_markup[key] = re.sub(r'\{\{[^}]*\}\}', '', remove_markup[key])



print(remove_markup)    