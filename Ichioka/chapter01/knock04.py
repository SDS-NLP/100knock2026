def build_element_symbol(sentence: str, single_char_positions: set) -> dict:
    words = sentence.split()
    symbol_map = {}

    for position, word in enumerate(words, start=1):
        clean_word = word.strip(".")
        if position in single_char_positions:
            symbol = clean_word[0]
        else:
            symbol = clean_word[:2]
        symbol_map[position] = symbol

    return symbol_map

sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
single_char_positions = {1, 5, 6, 7, 8, 9, 15, 16, 19}

result = build_element_symbol(sentence, single_char_positions)
print(result)