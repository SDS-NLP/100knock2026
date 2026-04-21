import random


def typoglycemia(text):
    word_list = text.split()
    ans = []
    for word in word_list:
        if len(word) <= 4:
            ans.append(word)
        else:
            middle = list(word[1:-1])
            random.shuffle(middle)
            ans.append(word[0] + "".join(middle) + word[-1])
    return " ".join(ans)


random.seed(0)
str1 = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
ans = typoglycemia(str1)

if __name__ == "__main__":
    print(ans)
