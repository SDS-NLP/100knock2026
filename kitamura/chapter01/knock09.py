import random
def random_word(text):
    text = text.split()
    word_list = []
    for word in text:
        if len(word) > 4:
            middle = list(word[1:-1])
            shuffled = random.sample(middle, len(middle))
            new_word = word[0] + "".join(shuffled) + word[-1]
            word_list.append(new_word)

        else:
            word_list.append(word)

    answer = " ".join(word_list)

    return answer


text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(random_word(text))