from janome.tokenizer import Tokenizer

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

def extract_verbs_with_base(text):
    t = Tokenizer()
    verbs = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if pos == '動詞':
            verbs.append((token.surface, token.base_form))
    return verbs

if __name__ == "__main__":
    for surface, base in extract_verbs_with_base(text):
        print(f"{surface}\t{base}")