def extract_noun_of_noun(morphs):
    for idx, morph in enumerate(morphs):
        if morph["surface"] != "の" or idx == 0 or idx == (len(morphs) - 1):
            continue
        noun1, of, noun2 = morphs[idx - 1], morphs[idx], morphs[idx + 1]
        if (
            noun1["pos"] == "名詞"
            and of["surface"] == "の"
            and noun2["pos"] == "名詞"
        ):
            yield noun1["surface"] + of["surface"] + noun2["surface"]

if __name__ == "__main__":
    from morph_utils import load_text, parse_to_morphs

    text = load_text("data/run_melos.txt")
    morphs = parse_to_morphs(text)

    for noun_of_noun in extract_noun_of_noun(morphs):
        print(noun_of_noun)
