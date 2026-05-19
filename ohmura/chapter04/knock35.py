from graphviz import Digraph

def visualize_tree():
    dot = Digraph(node_attr={'fontname': 'AppleGothic'}) 
    
    dot.edge('メロスは', '激怒した。')
    
    dot.render('meros_tree', view=True)
    print("画像ファイル 'meros_tree.pdf' を生成しました。")

if __name__ == "__main__":
    visualize_tree()