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
    if features[0] == "動詞":
        surface = node.surface          # 表層形（活用形）
        base = features[6]              # 原形
        conjugation = features[5]      # 活用形
        verbs.append((surface, base, conjugation))
    node = node.next

print(f"{'表層形':<10} {'原形':<10} {'活用形'}")
print("-" * 40)
for surface, base, conj in verbs:
    print(f"{surface:<10} {base:<10} {conj}")