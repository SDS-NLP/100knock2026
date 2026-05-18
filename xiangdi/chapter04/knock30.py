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

verbs = []

while node:
    features = node.feature.split(",")
    pos = features[0]

    next_node = node.next

    if next_node:
        next_features = next_node.feature.split(",")

        if pos == "名詞" and next_features[7] == "為る":
            verbs.append(node.surface + next_node.surface)

        elif pos == "動詞" and features[7] != "為る":
            verbs.append(node.surface)
    
    node = node.next

for verb in verbs:
    print(verb)