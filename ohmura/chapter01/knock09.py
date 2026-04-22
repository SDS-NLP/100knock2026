import random
def typoglycemia(text):
    words = text.split()
    result = []

    for word in words:
        if len(word) <= 4:
            result.append(word)
        else:
            start = word[0]
            end = word[-1]
            middle = list(word[1:-1])

            random.shuffle(middle)

            new_word = start + "".join(middle) + end
            result.append(new_word)

    return " ".join(result)

target_text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(typoglycemia(target_text))