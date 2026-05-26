# cd project

# python3 -m venv venv
# source venv/bin/activate

# python3 -m pip install mecab-python3

from pathlib import Path
import MeCab

#chapter04.txtを文字列として読み込む
file_path = Path(__file__).with_name("chapter04.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


#MeCabの解析器作成
mecab = MeCab.Tagger(r"-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic")
node = mecab.parseToNode(text)


# while node:
#     print(node.surface)
#     print(node.feature)
#     print("------")

#     node = node.next

while node:

    #品詞情報をカンマ区切りでリスト化
    features = node.feature.split(",")

    if features[0] == "動詞":
        print(features[6])
    
    node = node.next