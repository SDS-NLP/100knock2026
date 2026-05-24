from janome.tokenizer import Tokenizer
from knock30 import extract_verbs

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

if __name__ == "__main__":
    verbs = extract_verbs(text)
    for surface, base in verbs:
        print(f"{base}")