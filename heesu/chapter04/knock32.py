import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def extract_noun_no_noun(text):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    tokens = []

    while node:
        if node.surface:
            features = node.feature.split(",")
            pos = features[0]
            tokens.append({"surface": node.surface, "pos": pos})
        node = node.next

    extracted_phrases = []

    for i in range(len(tokens) - 2):
        word1 = tokens[i]
        word2 = tokens[i + 1]
        word3 = tokens[i + 2]

        if (
            word1["pos"] == "名詞"
            and word2["surface"] == "の"
            and word2["pos"] == "助詞"
            and word3["pos"] == "名詞"
        ):
            phrase = f"{word1['surface']}{word2['surface']}{word3['surface']}"
            extracted_phrases.append(phrase)

    return extracted_phrases


if __name__ == "__main__":
    results = extract_noun_no_noun(text)

    for phrase in results:
        print(f"{phrase}")
