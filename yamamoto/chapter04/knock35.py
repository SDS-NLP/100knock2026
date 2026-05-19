#「メロスは激怒した。」の係り受け木を可視化せよ。

import CaboCha
from graphviz import Digraph

text = "メロスは激怒した。"

parser = CaboCha.Parser()

tree = parser.parse(text)

chunks = []

for i in range(tree.chunk_size()):
    
    chunk = tree.chunk(i)
    surface = ""
    
    for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
        
        token = tree.token(j)
        
        if token.feature.split(",")[0] != "記号":
        
            surface += token.surface
    
    chunks.append(surface)
    
graph = Digraph(format = "png") #png形式で有向グラフを作成

for i, chunk in enumerate(chunks): #enumerate()：リストのインデックスと値を同時に取得
    
    graph.node(str(i), chunk) #文節をi番目のノードに追加

for i in range(tree.chunk_size()):
    
    link = tree.chunk(i).link
    
    if link != -1: #最後の文節ではないとき
        
        graph.edge(str(i), str(link)) #文節(i番目)と係り受け先(link番目)の間にエッジを追加

graph.render("dependency_tree", view = True) #dependency_tree.pngを生成   