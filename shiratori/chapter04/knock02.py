import MeCab


def get_pair(text):

    tagger = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc")
    node = tagger.parseToNode(text)

    tokens = []

    while node:
        features = node.feature.split(",")
        tokens.append((node.surface, features[0]))

        node = node.next

    for i in range(len(tokens)):
        if tokens[i][1] == "名詞" and tokens[i + 1][0] == "の" and tokens[i + 2][1] == "名詞":
            print(tokens[i][0] + tokens[i + 1][0] + tokens[i + 2][0])


if __name__ == "__main__":
    text = """
        メロスは激怒した。
        必ず、かの邪智暴虐の王を除かなければならぬと決意した。
        メロスには政治がわからぬ。
        メロスは、村の牧人である。
        笛を吹き、羊と遊んで暮して来た。
        けれども邪悪に対しては、人一倍に敏感であった。
        """

    get_pair(text)
