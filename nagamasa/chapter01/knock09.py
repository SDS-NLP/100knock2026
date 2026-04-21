import random

def more4_shuffle(text):
    word_list = text.split(" ")
    shuffled_list = []
    for word in word_list:
        if len(word) > 4:
            center = word[1:-1]
            char_list = list(center)
            random.shuffle(char_list)
            shuffled_word = "".join([word[0]] + char_list + [word[-1]])
            shuffled_list.append(shuffled_word)
        else:
            shuffled_list.append(word)
    
    return " ".join(shuffled_list)

if __name__ == "__main__":
    sample_text = "I couldn’t believe thatI could actually understand what I was reading : the phenomenal power of the human mind ."
    print(more4_shuffle(sample_text))