#文章(text)において、2つの名詞が「の」で連結されている名詞句をすべて抽出せよ。

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

res_lines = result.splitlines()

list = [] #表層形とその情報をまとめて格納するリスト

for line in res_lines[:-1]:
    
    features_list = [] 
    
    surface, feature = line.split("\t")
    features = feature.split(",")
    
    features.insert(0, surface) #featuresの0番目にsurfaceを挿入
    list.append(features) #リストに格納

for i in range(1, len(list)-1): #「の」があるのは２番目〜最後の1個前まで
    
    if list[i-1][1] == "名詞" and list[i][0] == "の" and list[i+1][1] == "名詞": #「の」の前後が名詞で挟まれているとき
        
        print(list[i-1][0], list[i][0], list[i+1][0]) #名詞句をなす表層形を出力