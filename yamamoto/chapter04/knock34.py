#文章(text)において、「メロス」が主語であるときの述語を抽出せよ。

import CaboCha

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

parser = CaboCha.Parser() #係り受け解析器

tree = parser.parse(text) #係り受け木を構築

chunks = [] #全文節を格納するリスト

for i in range(tree.chunk_size()): #文節数
    
    chunk = tree.chunk(i) #文節を取得(ex."メロスは")
    surface = ""
    
    for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size): #chunk.token_pos：文節(chunk)の開始token位置
        
        token = tree.token(j) #token(形態素オブジェクト)を取得(ex."メロス", "は")
        
        if token.feature.split(",")[0] != "記号": #品詞(token.featureを","で分割したときの0番目)が"記号"でないとき
        
            surface += token.surface #tokenの表層形を文字列に結合, token.feature：品詞情報
    
    chunks.append(surface)

for i in range(tree.chunk_size()):
    
    chunk = tree.chunk(i)
    dst = chunk.link 
    
    if "メロス" in chunks[i]: #文節に主語となる「メロス」がある
        
        print(chunks[i], "\t", chunks[dst]) #係り受け先(術語)を出力