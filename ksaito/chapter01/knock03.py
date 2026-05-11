def word_lens(sentence: str) -> list[str]:
    words = sentence.replace(",", "").replace(".", "").split()
    return [len(word) for word in words]

if __name__ == "__main__":
    sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    print(word_lens(sentence))
