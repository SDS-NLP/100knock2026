import MeCab


def get_pair(filename):
    text = """
        メロスは激怒した。
        必ず、かの邪智暴虐の王を除かなければならぬと決意した。
        メロスには政治がわからぬ。
        メロスは、村の牧人である。
        笛を吹き、羊と遊んで暮して来た。
        けれども邪悪に対しては、人一倍に敏感であった。
        """

    tagger = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc")
    node = tagger.parseToNode(text)

    while node:
        features = node.feature.split(",")
        if features[0] == "動詞":
            print(f"動詞:{node.surface}, 原型:{features[6]}")

        node = node.next


if __name__ == "__main__":
    filename = "chapter04/kokoro.txt"
    get_pair(filename)
