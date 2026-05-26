# 33. 係り受け解析
# 文章textに係り受け解析を適用し、係り元と係り先のトークン
# （形態素や文節などの単位）をタブ区切り形式ですべて抽出せよ。

import spacy 
import ginza  

nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
doc = nlp("メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。")

# 文節のリストを作成
bunsetu_list = list(ginza.bunsetu_spans(doc))

# 2. 各文節をループ処理し、文節同士の係り受けを抽出
for span in bunsetu_list:
    # 現在の文節の中心（主となる単語）が、どこに係っているかを取得
    head_token = span.root.head
    
    # 係り先の単語が含まれる「文節」を特定する
    head_span = span
    for b in bunsetu_list:
        if head_token in b:
            head_span = b
            break
            
    # 係り元と係り先をタブ区切りで出力
    print(f"{span.text}\t{head_span.text}")