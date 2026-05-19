import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def get_verbs_dict(text):
    tagger = MeCab.Tagger()

    node = tagger.parseToNode(text)

    dicts = []

    while node:
        features = node.feature.split(",")
        if features[0] == "動詞":
            dict_form = features[7]

            dicts.append(dict_form)

        node = node.next

    return dicts


if __name__ == "__main__":
    verbs = get_verbs_dict(text)
    for verb in verbs:
        print(verb)
