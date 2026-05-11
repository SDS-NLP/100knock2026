text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = text.replace(".", "").split()

solo_indices = [1, 5, 6, 7, 8, 9, 15, 16, 19]

result = {}

for i, word in enumerate(words, 1):
    if i in solo_indices:
        key = word[0]
    else: 
        key =word[:2]

    result[key] = i

print(result)