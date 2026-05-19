import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def get_verbs(text):
    tagger = MeCab.Tagger()

    node = tagger.parseToNode(text)

    verbs = []

    while node:
        features = node.feature.split(",")
        if features[0] == "動詞":
            surface = node.surface

            verbs.append(surface)

        node = node.next

    return verbs


if __name__ == "__main__":
    verbs = get_verbs(text)
    for verb in verbs:
        print(verb)
