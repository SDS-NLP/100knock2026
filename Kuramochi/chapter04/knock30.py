from janome.tokenizer import Tokenizer

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
def extract_verbs(text):
    t = Tokenizer()
    verbs = []
    for token in t.tokenize(text):
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech == '動詞':
            verbs.append(token.surface)
    return verbs

if __name__ == "__main__":
    verbs = extract_verbs(text)
    print("抽出された動詞:")
    for verb in verbs:
        print(verb)
