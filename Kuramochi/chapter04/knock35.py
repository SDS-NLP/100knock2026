import CaboCha
from graphviz import Digraph

text = "メロスは激怒した。"

parser = CaboCha.Parser()
tree   = parser.parse(text)

chunks = []

for i in range(tree.chunk_size()):
    chunk   = tree.chunk(i)
    surface = ''
    for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
        token = tree.token(j)
        if token.feature.split(',')[0] != '記号':
            surface += token.surface
    chunks.append(surface)

graph = Digraph(format='png')
for i, chunk in enumerate(chunks):
    graph.node(str(i), chunk)

for i in range(tree.chunk_size()):
    link = tree.chunk(i).link
    if link != -1:
        graph.edge(str(i), str(link))

if __name__ == '__main__':
    graph.render('dependency_tree', view=True)
