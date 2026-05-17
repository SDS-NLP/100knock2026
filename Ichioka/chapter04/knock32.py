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

# 全トークンを(表層形, 品詞)のリストに変換
tokens = []
while node:
    features = node.feature.split(",")
    tokens.append((node.surface, features[0]))
    node = node.next

# 「名詞 → の → 名詞」のパターンを検索
results = []
for i in range(len(tokens) - 2):
    surf0, pos0 = tokens[i]
    surf1, pos1 = tokens[i+1]
    surf2, pos2 = tokens[i+2]
    if pos0 == "名詞" and surf1 == "の" and pos2 == "名詞":
        results.append(f"{surf0}の{surf2}")

print("抽出された名詞句：")
for r in results:
    print(f"  {r}")