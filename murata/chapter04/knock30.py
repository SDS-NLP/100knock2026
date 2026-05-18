import MeCab

def extract_verb(text):
    mecab = MeCab.Tagger()
    nodes = mecab.parseToNode(text)
    tokens = []
    while nodes:
        if nodes.feature.split(',')[0] == "動詞":
             tokens.append({"surface": nodes.surface, "base": nodes.feature.split(',')[6] if len(nodes.feature) > 7 else nodes.surface})
        nodes = nodes.next
    return tokens

if __name__ == "__main__":
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """ 
    result = extract_verb(text)
    for i in range(len(result)):
        print(result[i]["surface"])