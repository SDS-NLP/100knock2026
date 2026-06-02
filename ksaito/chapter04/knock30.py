def extract_verbs(morphs):
    for morph in morphs:
        if morph["pos"] != "動詞":
            continue

        yield morph["surface"]


if __name__ == "__main__":
    from morph_utils import load_text, parse_to_morphs

    textfile_path = "data/run_melos.txt"
    text = load_text(textfile_path)
    morphs = parse_to_morphs(text)
    
    for verb in extract_verbs(morphs):
        print(verb)

