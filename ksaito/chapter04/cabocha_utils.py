import CaboCha

def iter_sentences(path):
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            yield line

def parse_to_chunks(sentence, parser):
    tree = parser.parse(sentence)
    chunks = []

    for i in range(tree.size()):
        token = tree.token(i)
        if token.chunk is not None:
            chunks.append({
                "link": token.chunk.link,
                "tokens": [],
            })

        chunks[-1]["tokens"].append({
            "surface": token.surface,
            "feature": token.feature.split(",")
        })

    for chunk in chunks:
        chunk["surface"] = "".join(t["surface"] for t in chunk["tokens"])
    return chunks

if __name__ == "__main__":
    textfile_path= "data/run_melos.txt"
    parser = CaboCha.Parser()

    for sentence in iter_sentences(textfile_path):

        chunks = parse_to_chunks(sentence, parser)
        print(f"sentence = {sentence}")
        print(f"chunks = {chunks}")

