import CaboCha
import pydot

def visualize_dependency_tree(text):
    parser = CaboCha.Parser()
    tree = parser.parse(text)
    graph = pydot.Dot(graph_type='digraph')
    graph.set_node_defaults(fontname='Noto Sans CJK JP')
    def get_chunk_text(chunk):
        chunk_text = ""
        for i in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            token = tree.token(i)
            if token.feature.split(',')[0] != "記号":
                chunk_text += token.surface
        return chunk_text
    
    for i in range(tree.chunk_size()):
        chunk = tree.chunk(i)
        
        if chunk.link != -1:
            src_text = get_chunk_text(chunk)
            dst_text = get_chunk_text(tree.chunk(chunk.link))
            if src_text and dst_text:
                edge = pydot.Edge(src_text, dst_text)
                graph.add_edge(edge)
        graph.write_png("dependency_tree.png")
if __name__=="__main__":
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """
    visualize_dependency_tree(text)