import json

with open('./100knock2026/jawiki-country.json/jawiki-country.json', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)  
        if article["title"] == "イギリス":
            uk_text = article["text"]
            break

print(uk_text) 
with open('uk.txt', 'w', encoding='utf-8') as f:
    f.write(uk_text)
    