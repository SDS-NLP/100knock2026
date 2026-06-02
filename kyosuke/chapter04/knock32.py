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
word_list=[]

for line in parsed_text.splitlines():
    if line == "EOS":
        break
    surface, feature = line.split("\t")
    features = feature.split(",")
    word_list.append((surface, features))

for i in range(1, len(word_list)-1):
    if word_list[i][1][0] == "助詞" and word_list[i][1][1] == "連体化" and word_list[i-1][1][0] == "名詞" and word_list[i+1][1][0] == "名詞":
        print(word_list[i-1][0], word_list[i][0], word_list[i+1][0])
