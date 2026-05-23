from janome.tokenizer import Tokenizer

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

def extract_noun_no_noun(text):
    tokenizer = Tokenizer()
    tokens    = list(tokenizer.tokenize(text))
    phrases   = []

    for i in range(len(tokens) - 2):
        first  = tokens[i]
        middle = tokens[i + 1]
        second = tokens[i + 2]

        first_pos  = first.part_of_speech.split(',')[0]
        middle_pos = middle.part_of_speech.split(',')[0]
        second_pos = second.part_of_speech.split(',')[0]

        if first_pos == '名詞' and middle.surface == 'の' and middle_pos == '助詞' and second_pos == '名詞':
            phrases.append(first.surface + middle.surface + second.surface)

    return phrases

if __name__ == "__main__":
    for phrase in extract_noun_no_noun(text):
        print(phrase)