#文章(text)に含まれる動詞をすべて表示せよ。

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

result = tagger.parse(text) #形態素解析

res_lines = result.splitlines() #解析結果を行ごとに分割しリスト化

for line in res_lines[:-1]: #最終行はEOS(End of Sequence)なので除外
    
    surface, feature = line.split("\t") #res_linesを"\t"の部分で分割 ⇨ 表層形とその情報の2分割
    features = feature.split(",")
    
    if features[0] == "動詞":
        
        print(surface)