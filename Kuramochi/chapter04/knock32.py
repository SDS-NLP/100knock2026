from janome.tokenizer import Tokenizer

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def extract_noun_phrases(text):
    tokenizer = Tokenizer()
    tokens    = [token for token in tokenizer.tokenize(text)]
    # なぜtokensで出力すると、変になりtokens[0]で出力すると正常なのか？ --- IGNORE ---
    print(tokens[0:5]) # 範囲で出力すると変になる。

    noun_phrases = []
    for i in range(len(tokens) - 2):
        left  = tokens[i]
        mid   = tokens[i + 1]
        right = tokens[i + 2]

        left_pos  = left.part_of_speech.split(',')[0]
        right_pos = right.part_of_speech.split(',')[0]

        if left_pos == '名詞' and mid.surface == 'の' and right_pos == '名詞':
            noun_phrases.append(left.surface + mid.surface + right.surface)

    return noun_phrases


if __name__ == '__main__':
    for phrase in extract_noun_phrases(text):
        print(phrase)
