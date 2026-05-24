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

    t     = Tokenizer()
    verbs = []

    for token in t.tokenize(text):
        print(token)                # テキストを形態素解析して、各トークンを処理
        pos = token.part_of_speech.split(',')[0]  # 品詞を取得
        if pos == '動詞':                          # 品詞が動詞であれば
            verbs.append((token.surface, token.base_form)) # 表層形と基本形を

    return verbs   

if __name__ == "__main__":
    verbs = extract_verbs(text)
    print(verbs)
    for verb in verbs:
        print(verb)
