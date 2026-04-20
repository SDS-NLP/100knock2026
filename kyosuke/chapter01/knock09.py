import random
def typoglycemia(text):
    result = []
    words  = text.split()
    for word in words:
        if len(word) <= 4:
            result.append(word)
        else:
            middle = list(word[1:-1])
            random.shuffle(middle)
            new_word = word[0] + "".join(middle) + word[-1]
            result.append(new_word)
    return " ".join(result)
print(typoglycemia("I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."))