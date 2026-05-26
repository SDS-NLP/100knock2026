# pip install mecab-python3 unidic-lite
import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

phrases = []

while node and node.next and node.next.next:
    features = node.feature.split(",")
    next_features = node.next.feature.split(",")
    next_next_features = node.next.next.feature.split(",")

    if (
        features[0] == "名詞" 
        and node.next.surface == "の" 
        and next_next_features[0] == "名詞"
    ):
        phrases.append(node.surface + "の" + node.next.next.surface)

    node = node.next

print(phrases)