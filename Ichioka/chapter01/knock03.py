def get_word_lengths(text: str) -> list[int]:
    words = text.split()
    word_lengths = []

    for word in words:
        clean_word = word.strip(",.")
        word_lengths.append(len(clean_word))

    return word_lengths


text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print(get_word_lengths(text))