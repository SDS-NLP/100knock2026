from cabocha_utils import iter_sentences, parse_to_chunks
import CaboCha

parser = CaboCha.Parser()
for sentence in iter_sentences("data/run_melos.txt"):
    chunks = parse_to_chunks(sentence, parser)
    for chunk in chunks:
        if chunk["link"] == -1:
            continue

        src = chunk["surface"]
        if "メロス" not in src:
            continue
        dst = chunks[chunk["link"]]["surface"]
        print(f"{src}\t{dst}")
    print("-" * 60)
