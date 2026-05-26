import CaboCha

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def extract_dependency_relations(text):
    parser = CaboCha.Parser()  # 係り受け解析器を生成
    tree   = parser.parse(text)

    chunks = []
    for i in range(tree.chunk_size()):
        chunk      = tree.chunk(i)
        chunk_text = ''
        for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            token       = tree.token(j)
            chunk_text += token.surface
        chunks.append(chunk_text)

    relations = []
    for i, chunk in enumerate(chunks):
        dst = tree.chunk(i).link
        if dst != -1:
            relations.append((chunk, chunks[dst]))

    return relations


if __name__ == '__main__':
    for src, dst in extract_dependency_relations(text):
        print(f"{src}\t{dst}")
