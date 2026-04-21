string = "Hi He Lied Because Boron Could Not Oxidize Fluorine. \
New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = string.split(' ')

magic_numbers = [0, 4, 5, 6, 7, 8, 14, 15, 18]

answer = {}
for i, word in enumerate(words):
    if i in magic_numbers:
        c = word[0]
    else:
        c = word[:2]
    answer[c] = i

print(answer)
