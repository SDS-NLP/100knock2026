#25の処理時に、テンプレートの値からMediaWikiの強調マークアップ（弱い強調、強調、強い強調のすべて）を除去してテキストに変換せよ。

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

for key, value in template.items(): #辞書型リストのすべてのkeyとvalueを取得
    
    remove_emphasis[key] = re.sub(r'\'{2,5}', '', value) #re.sub():置き換え　#2-5回連続する'を消す

print("removed", remove_emphasis)    