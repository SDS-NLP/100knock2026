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
        tokens.append((nodes.surface, features[0]))
    nodes = nodes.next

nouns = [token for token, pos in tokens if pos == "名詞"]
noun_phrases = [
    f"{tokens[i][0]}の{tokens[i + 2][0]}"
    for i in range(len(tokens) - 2)
    if (
        tokens[i][1] == "名詞"
        and tokens[i + 1][0] == "の"
        and tokens[i + 2][1] == "名詞"
    )
]
print(noun_phrases)
