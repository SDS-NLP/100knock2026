import knock03

if __name__ == "__main__":

    text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    index = [1, 5, 6, 7, 8, 9, 15, 16, 19]

    words = knock03.split_words(text)
    d = {}

    for i, word in enumerate(words):
        if i + 1 in index:
            target_letters = word[:1]
        else:
            target_letters = word[:2]
        d[i + 1] = target_letters
    
    print(d)