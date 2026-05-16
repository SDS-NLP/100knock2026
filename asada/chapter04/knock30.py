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
        tokens.append((nodes.surface, nodes.feature.split(",")[0]))
    nodes = nodes.next
print(tokens)

verbs = [token for token, pos in tokens if pos == "動詞"]
print(f"動詞: {verbs}")
