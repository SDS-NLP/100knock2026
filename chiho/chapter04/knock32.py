from pathlib import Path
import MeCab

#chapter04.txtを文字列として読み込む
file_path = Path(__file__).with_name("chapter04.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


#MeCabの解析器作成
mecab = MeCab.Tagger(r"-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic")
node = mecab.parseToNode(text)

while node:
     
    if node.next and node.next.next:
          first = node.feature.split(",")[0]
          third = node.feature.split(",")[0]

          if first == "名詞" and node.next.surface == "の" and third == "名詞":
               print(node.surface + node.next.surface + node.next.next.surface)
    node = node.next