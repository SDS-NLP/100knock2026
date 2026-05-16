import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

wakati = MeCab.Tagger()
nodes = wakati.parseToNode(text)
tokens = []
while nodes:
    if nodes.surface != "":
        features = nodes.feature.split(",")
        tokens.append(
            (nodes.surface, features[0], features[7] if len(features) > 8 else None)
        )
    nodes = nodes.next
print(tokens)

base_verbs = [(token, base) for token, pos, base in tokens if pos == "動詞"]
print(f"動詞の原型: {base_verbs}")
