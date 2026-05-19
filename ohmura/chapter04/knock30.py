from janome.tokenizer import Tokenizer

def extract_verbs_janome():
    text = """メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては, 人一倍に敏感であった。"""

    t = Tokenizer()
    verbs = []

    for token in t.tokenize(text):
        part_of_speech = token.part_of_speech.split(',')[0]
        
        if part_of_speech == '動詞':
            verbs.append(token.surface)

    print(", ".join(verbs))

if __name__ == "__main__":
    extract_verbs_janome()