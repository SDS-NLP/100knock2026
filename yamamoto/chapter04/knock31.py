#文章(text)に含まれる動詞と、その原型をすべて表示せよ。

import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

tagger = MeCab.Tagger()

result = tagger.parse(text)

res_lines = result.splitlines() #改行文字(\n)で分割

for line in res_lines[:-1]:
    
    surface, feature = line.split("\t")
    features = feature.split(",")
    
    if features[0] == "動詞":
        
        print(surface, features[6]) #MeCabではfeature[6]が原型