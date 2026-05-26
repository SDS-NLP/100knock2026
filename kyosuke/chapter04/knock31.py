import MeCab
import ipadic

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
parsed_text = tagger.parse(text)

for line in parsed_text.splitlines():
    if line == "EOS":
        break
    surface, feature = line.split("\t")
    features = feature.split(",")
    if features[0] == "動詞":
        print(surface,features[6])
