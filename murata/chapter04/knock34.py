import CaboCha

def parse_CaboCha(text):
    parser = CaboCha.Parser()
    tree = parser.parse(text)
    chunk_list = []
    for i in range(tree.chunk_size()):
        chunk = tree.chunk(i)
        chunk_text = ""
        for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            token = tree.token(j)
            chunk_text += token.surface
        link_id = chunk.link
        chunk_list.append({
            "text": chunk_text,
            "link": link_id
        })
    results = []
    for chunk in chunk_list:
        if chunk["link"] != -1:
            src_text = chunk["text"]
            dst_text = chunk_list[chunk["link"]]["text"]
            if src_text and dst_text and ("メロス" in src_text):
                results.append(f"{dst_text}")
    return results

if __name__=="__main__":
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """
    result = parse_CaboCha(text)
    for line in result:
    
        print(line)