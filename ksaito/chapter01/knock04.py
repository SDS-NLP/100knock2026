def build_ch_to_index(sentence, indexes_single_ch: set) -> dict:
    ch_to_index = {}
    words = sentence.split()

    for i, word in enumerate(words):
        if i + 1 in indexes_single_ch:
            ch_to_index[word[0]] = i + 1
            continue
        
        ch_to_index[word[:2]] = i + 1

    return ch_to_index

if __name__ == "__main__":
    sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can"
    indexes_single_ch = set([1, 5, 6, 7, 8, 9, 15, 16, 19])
    print(build_ch_to_index(sentence, indexes_single_ch))
