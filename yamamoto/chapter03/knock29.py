#テンプレートの内容を利用し、国旗画像のURLを取得せよ。

import json
import re
import urllib.request
import urllib.parse

template = {}

with open("jawiki-country.json", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        dict = json.loads(line)
        
        if dict["title"] == "イギリス":
            
            basic_info = re.search(r'^\{\{基礎情報.*?$(.*?)^\}\}$', dict["text"], re.S | re.M)
            list = re.findall(r'^\|([^=]+?)\s*=\s*(.+?)(?=\n^\||\n^\}\})', basic_info.group(1), re.M | re.S)
            
            for i in list:
                
                template[i[0]] = i[1]

flag_file = template["国旗画像"]

params = urllib.parse.urlencode({
    'action': 'query',
    'titles': f'File:{flag_file}',
    'prop': 'imageinfo',
    'iiprop': 'url',
    'format': 'json'
})

req = urllib.request.Request(
    f'https://en.wikipedia.org/w/api.php?{params}',
    headers={'User-Agent': 'knock29/1.0'}
)

with urllib.request.urlopen(req) as res:
    data = json.load(res)

pages = data['query']['pages']
page = next(iter(pages.values()))
print(page['imageinfo'][0]['url'])    